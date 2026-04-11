"""
Camera capture and YOLO+ArUco inference pipeline.

Runs two background threads:
  - capture_thread: reads frames from the USB webcam
  - depth_capture_thread: reads frames from the Arducam on CAM1
  - inference_thread: runs YOLO + ArUco on each frame
"""

# CRITICAL: import torch BEFORE cv2 on Jetson to prevent CUDA heap corruption
import torch  # noqa: F401  (must be first)

import cv2
import numpy as np
from pathlib import Path
import subprocess
import threading
import time

from vision.config import (
    CAMERA_DEVICE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    DEPTH_CAMERA_ENABLED,
    DEPTH_CAMERA_FLIP,
    DEPTH_CAMERA_SENSOR_ID,
    DEPTH_CAMERA_SENSOR_MODE,
    DEPTH_FRAME_WIDTH,
    DEPTH_FRAME_HEIGHT,
    DEPTH_CAMERA_FPS,
    DEPTH_PREVIEW_WIDTH,
    DEPTH_PREVIEW_HEIGHT,
    DEPTH_CAMERA_FOV_H_DEG,
    DEPTH_CAMERA_HEIGHT_M,
    DEPTH_CAMERA_PITCH_DEG,
    DEPTH_YOLO_CLASSES,
    DEPTH_YOLO_EVERY_N_FRAMES,
    YOLO_ENABLED,
    YOLO_MODEL,
    YOLO_CLASSES,
    YOLO_MIN_CONFIDENCE,
)
from vision import aruco
from vision.depth_forward import build_csi_gstreamer_pipeline
from vision.depth_forward import build_v4l2_bayer_pipeline
from vision.depth_forward import estimate_forward_distance_m

if YOLO_ENABLED:
    from ultralytics import YOLO

# ── Load YOLO ────────────────────────────────────────────────────────
model = None
if YOLO_ENABLED:
    print("Loading YOLO model for workspace camera...", flush=True)
    model = YOLO(YOLO_MODEL)
    print("YOLO model loaded.", flush=True)
else:
    print("YOLO disabled.", flush=True)

# ── Shared state ─────────────────────────────────────────────────────
latest_frame   = None
latest_results = None
latest_aruco   = None
latest_depth_frame = None
latest_depth_info = None
frame_lock     = threading.Lock()
camera_ref     = None
depth_camera_ref = None
depth_snapshot_process = None
depth_snapshot_dir = Path("/tmp/cam0_snapshots")
depth_camera_source = f"CAM{DEPTH_CAMERA_SENSOR_ID}"
depth_camera_actual = {
    "width": None,
    "height": None,
    "fps": None,
}


def get_state():
    """Return a snapshot of (frame, yolo_results, aruco_data)."""
    with frame_lock:
        if latest_frame is not None:
            return latest_frame.copy(), latest_results, latest_aruco
        if latest_depth_frame is not None:
            return latest_depth_frame.copy(), None, None
        return None, None, None


def get_aruco():
    """Return the latest ArUco detection dict."""
    with frame_lock:
        return latest_aruco


def get_camera_ref():
    """Return the raw camera object for cleanup."""
    return camera_ref


def get_depth_info():
    """Return the latest forward-distance camera output."""
    with frame_lock:
        if latest_depth_info is not None:
            return dict(latest_depth_info)

    snapshot = _latest_depth_snapshot_path()
    if snapshot is None:
        return None

    return {
        "enabled": DEPTH_CAMERA_ENABLED,
        "camera_ready": True,
        "detected": False,
        "source": "CAM0",
        "frame_width": DEPTH_FRAME_WIDTH,
        "frame_height": DEPTH_FRAME_HEIGHT,
        "fps": float(DEPTH_CAMERA_FPS),
        "class": None,
        "confidence": None,
        "pixel_bbox": None,
        "forward_distance_m": None,
        "forward_distance_in": None,
    }


def get_depth_frame():
    """Return a copy of the latest CAM1 frame."""
    global latest_depth_info

    with frame_lock:
        if latest_depth_frame is None:
            pass
        else:
            return latest_depth_frame.copy()

    snapshot = _latest_depth_snapshot_path()
    if snapshot is None:
        return None

    try:
        data = snapshot.read_bytes()
        frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            with frame_lock:
                latest_depth_info = {
                    "enabled": DEPTH_CAMERA_ENABLED,
                    "camera_ready": True,
                    "detected": False,
                    "source": "CAM0",
                    "frame_width": DEPTH_FRAME_WIDTH,
                    "frame_height": DEPTH_FRAME_HEIGHT,
                    "fps": float(DEPTH_CAMERA_FPS),
                    "class": None,
                    "confidence": None,
                    "pixel_bbox": None,
                    "forward_distance_m": None,
                    "forward_distance_in": None,
                }
        return frame
    except Exception:
        return None


def release_cameras():
    """Release both camera handles cleanly."""
    global camera_ref, depth_camera_ref, depth_camera_source, depth_camera_actual, depth_snapshot_process

    if camera_ref is not None:
        camera_ref.release()
        camera_ref = None
    if depth_camera_ref is not None:
        depth_camera_ref.release()
        depth_camera_ref = None
    if depth_snapshot_process is not None:
        depth_snapshot_process.terminate()
        try:
            depth_snapshot_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            depth_snapshot_process.kill()
        depth_snapshot_process = None
    depth_camera_source = f"CAM{DEPTH_CAMERA_SENSOR_ID}"
    depth_camera_actual = {
        "width": None,
        "height": None,
        "fps": None,
    }


def _latest_depth_snapshot_path():
    if not depth_snapshot_dir.exists():
        return None
    files = sorted(depth_snapshot_dir.glob("frame-*.jpg"))
    return files[-1] if files else None


def start_depth_snapshot_stream():
    """Start a standalone GStreamer process that writes CAM0 JPEG snapshots."""
    global depth_snapshot_process, latest_depth_info, depth_camera_source, depth_camera_actual

    if depth_snapshot_process is not None and depth_snapshot_process.poll() is None:
        return

    depth_snapshot_dir.mkdir(parents=True, exist_ok=True)
    for old in depth_snapshot_dir.glob("frame-*.jpg"):
        old.unlink()

    flip_method = "2" if DEPTH_CAMERA_FLIP else "0"
    cmd = [
        "gst-launch-1.0",
        "-q",
        "nvarguscamerasrc",
        f"sensor-id={DEPTH_CAMERA_SENSOR_ID}",
        f"sensor-mode={DEPTH_CAMERA_SENSOR_MODE}",
        "!",
        f"video/x-raw(memory:NVMM),width=(int){DEPTH_FRAME_WIDTH},height=(int){DEPTH_FRAME_HEIGHT},format=(string)NV12,framerate=(fraction){DEPTH_CAMERA_FPS}/1",
        "!",
        "nvvidconv",
        f"flip-method={flip_method}",
        "!",
        f"video/x-raw,width=(int){DEPTH_PREVIEW_WIDTH},height=(int){DEPTH_PREVIEW_HEIGHT},format=(string)I420",
        "!",
        "nvjpegenc",
        "quality=85",
        "!",
        "multifilesink",
        f"location={depth_snapshot_dir}/frame-%05d.jpg",
        "max-files=3",
    ]

    depth_snapshot_process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    depth_camera_source = "CAM0"
    depth_camera_actual = {
        "width": DEPTH_FRAME_WIDTH,
        "height": DEPTH_FRAME_HEIGHT,
        "fps": float(DEPTH_CAMERA_FPS),
    }
    latest_depth_info = {
        "enabled": DEPTH_CAMERA_ENABLED,
        "camera_ready": False,
        "detected": False,
        "source": "CAM0",
        "frame_width": DEPTH_FRAME_WIDTH,
        "frame_height": DEPTH_FRAME_HEIGHT,
        "fps": float(DEPTH_CAMERA_FPS),
        "class": None,
        "confidence": None,
        "pixel_bbox": None,
        "forward_distance_m": None,
        "forward_distance_in": None,
    }


def _open_usb_camera():
    """Open the first working USB webcam."""
    camera = None
    for idx in ([CAMERA_DEVICE] + [i for i in range(5) if i != CAMERA_DEVICE]):
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                camera = cap
                print(f"Found working camera at /dev/video{idx}", flush=True)
                break
        cap.release()
    return camera


def _open_depth_camera():
    """Open the Arducam on CAM1 through Jetson Argus."""
    print("Trying Arducam via /dev/video2 Bayer pipeline...", flush=True)
    v4l2_pipeline = build_v4l2_bayer_pipeline(
        device="/dev/video2",
        width=DEPTH_FRAME_WIDTH,
        height=DEPTH_FRAME_HEIGHT,
        fps=DEPTH_CAMERA_FPS,
    )
    v4l2_camera = cv2.VideoCapture(v4l2_pipeline, cv2.CAP_GSTREAMER)
    if v4l2_camera.isOpened():
        ret, frame = v4l2_camera.read()
        if ret and frame is not None:
            return v4l2_camera, "/dev/video2"
        v4l2_camera.release()

    sensor_ids = []
    for sensor_id in (DEPTH_CAMERA_SENSOR_ID, 0):
        if sensor_id not in sensor_ids:
            sensor_ids.append(sensor_id)

    for sensor_id in sensor_ids:
        print(f"Trying Arducam via Argus sensor-id={sensor_id}...", flush=True)
        pipeline = build_csi_gstreamer_pipeline(
            sensor_id=sensor_id,
            width=DEPTH_FRAME_WIDTH,
            height=DEPTH_FRAME_HEIGHT,
            fps=DEPTH_CAMERA_FPS,
            sensor_mode=DEPTH_CAMERA_SENSOR_MODE,
        )
        argus_camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if argus_camera.isOpened():
            ret, frame = argus_camera.read()
            if ret and frame is not None:
                return argus_camera, f"CAM{sensor_id}"
            argus_camera.release()

    return None, f"CAM{DEPTH_CAMERA_SENSOR_ID}"


def _build_depth_info(depth_frame):
    """Build the forward-camera payload from the Arducam feed."""
    with frame_lock:
        source = depth_camera_source
        actual = dict(depth_camera_actual)

    info = {
        "enabled": DEPTH_CAMERA_ENABLED,
        "camera_ready": depth_frame is not None,
        "detected": False,
        "source": source,
        "frame_width": actual["width"],
        "frame_height": actual["height"],
        "fps": actual["fps"],
        "class": None,
        "confidence": None,
        "pixel_bbox": None,
        "forward_distance_m": None,
        "forward_distance_in": None,
    }
    return info


def _select_best_box(results, confidence_threshold):
    """Pick the most confident detection above the configured threshold."""
    if results is None or len(results) == 0:
        return None

    best = None
    best_conf = None
    names = getattr(results[0], "names", {})
    for box in results[0].boxes:
        conf = float(box.conf[0].item())
        if conf < confidence_threshold:
            continue

        cls_id = int(box.cls[0].item())
        if isinstance(names, dict):
            cls_name = names.get(cls_id, str(cls_id))
        elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            cls_name = str(names[cls_id])
        else:
            cls_name = str(cls_id)

        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].detach().cpu().numpy()]
        if best_conf is None or conf > best_conf:
            best_conf = conf
            best = {
                "class": cls_name,
                "confidence": conf,
                "pixel_bbox": [x1, y1, x2, y2],
            }

    return best


def _populate_depth_detection(depth_info, detection, frame_shape):
    """Fill the depth payload using the latest cup detection."""
    if detection is None or frame_shape is None:
        return depth_info

    frame_height, frame_width = frame_shape[:2]
    x1, y1, x2, y2 = detection["pixel_bbox"]
    distance_m = estimate_forward_distance_m(
        bbox_bottom_y=y2,
        frame_height=frame_height,
        frame_width=frame_width,
        camera_height_m=DEPTH_CAMERA_HEIGHT_M,
        camera_fov_h_deg=DEPTH_CAMERA_FOV_H_DEG,
        pitch_deg=DEPTH_CAMERA_PITCH_DEG,
    )
    depth_info.update(
        {
            "detected": True,
            "class": detection["class"],
            "confidence": round(float(detection["confidence"]), 4),
            "pixel_bbox": [int(v) for v in detection["pixel_bbox"]],
            "forward_distance_m": None if distance_m is None else round(float(distance_m), 4),
            "forward_distance_in": None if distance_m is None else round(float(distance_m) * 39.3701, 4),
        }
    )
    return depth_info


# ── Threads ──────────────────────────────────────────────────────────
def capture_thread():
    """Continuously capture frames from the USB webcam."""
    global latest_frame, camera_ref

    camera = _open_usb_camera()
    if camera is None:
        print("ERROR: No working USB webcam found on /dev/video0-4", flush=True)
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    camera_ref = camera

    if not camera.isOpened():
        print("ERROR: Camera lost after configuration", flush=True)
        return

    w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    aruco.configure_camera_model(w, h)
    print(f"Camera opened: {w}x{h} @ {fps}fps", flush=True)

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame = frame


def depth_capture_thread():
    """Continuously capture frames from the Arducam on CAM1."""
    global latest_depth_frame, depth_camera_ref, latest_depth_info, depth_camera_source, depth_camera_actual

    if not DEPTH_CAMERA_ENABLED:
        with frame_lock:
            latest_depth_info = {
                "enabled": False,
                "camera_ready": False,
                "detected": False,
                "source": f"CAM{DEPTH_CAMERA_SENSOR_ID}",
                "frame_width": None,
                "frame_height": None,
                "fps": None,
                "class": None,
                "confidence": None,
                "pixel_bbox": None,
                "forward_distance_m": None,
                "forward_distance_in": None,
            }
        print("Forward depth camera disabled.", flush=True)
        return

    camera, source = _open_depth_camera()
    if camera is None:
        depth_camera_source = source
        with frame_lock:
            latest_depth_info = {
                "enabled": True,
                "camera_ready": False,
                "detected": False,
                "source": source,
                "frame_width": None,
                "frame_height": None,
                "fps": None,
                "class": None,
                "confidence": None,
                "pixel_bbox": None,
                "forward_distance_m": None,
                "forward_distance_in": None,
            }
        print(f"WARNING: Could not open Arducam depth camera from {source}", flush=True)
        return

    depth_camera_ref = camera
    depth_camera_source = source
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or DEPTH_FRAME_WIDTH
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or DEPTH_FRAME_HEIGHT
    actual_fps = camera.get(cv2.CAP_PROP_FPS) or DEPTH_CAMERA_FPS
    depth_camera_actual = {
        "width": actual_width,
        "height": actual_height,
        "fps": round(float(actual_fps), 2),
    }
    print(
        f"Forward depth camera opened on {source}: "
        f"{actual_width}x{actual_height} @ {actual_fps}fps",
        flush=True,
    )

    while True:
        success, frame = camera.read()
        if not success:
            time.sleep(0.01)
            continue
        if DEPTH_CAMERA_FLIP:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if frame.shape[1] != DEPTH_PREVIEW_WIDTH or frame.shape[0] != DEPTH_PREVIEW_HEIGHT:
            frame = cv2.resize(frame, (DEPTH_PREVIEW_WIDTH, DEPTH_PREVIEW_HEIGHT))
        with frame_lock:
            latest_depth_frame = frame
            latest_depth_info = _build_depth_info(frame)


def inference_thread():
    """Run ArUco detection and YOLO inference on the latest frame."""
    global latest_results, latest_aruco, latest_depth_info

    depth_frame_counter = 0
    cached_depth_detection = None

    while True:
        frame = None
        depth_frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
            if latest_depth_frame is not None:
                depth_frame = latest_depth_frame.copy()

        if depth_frame is None and DEPTH_CAMERA_ENABLED:
            depth_frame = get_depth_frame()

        if frame is None:
            time.sleep(0.01)
            continue

        aruco_data = aruco.detect(frame)
        results = model(frame, verbose=False, classes=YOLO_CLASSES) if model is not None else None
        depth_info = _build_depth_info(depth_frame)

        if depth_frame is not None and model is not None:
            depth_frame_counter += 1
            if (
                cached_depth_detection is None
                or depth_frame_counter % max(1, int(DEPTH_YOLO_EVERY_N_FRAMES)) == 0
            ):
                depth_results = model(
                    depth_frame,
                    verbose=False,
                    classes=DEPTH_YOLO_CLASSES,
                )
                cached_depth_detection = _select_best_box(depth_results, YOLO_MIN_CONFIDENCE)

            depth_info = _populate_depth_detection(depth_info, cached_depth_detection, depth_frame.shape)
        else:
            cached_depth_detection = None

        with frame_lock:
            latest_results = results
            latest_aruco = aruco_data
            latest_depth_info = depth_info


def start():
    """Start the capture and inference threads."""
    if YOLO_ENABLED:
        threading.Thread(target=capture_thread, daemon=True).start()

        # Let the USB workspace camera settle before opening the CSI camera.
        for _ in range(50):
            with frame_lock:
                if latest_frame is not None:
                    break
            time.sleep(0.02)
    else:
        print("Workspace camera disabled in stream-only mode.", flush=True)
        start_depth_snapshot_stream()
        return

    start_depth_snapshot_stream()
    if YOLO_ENABLED:
        time.sleep(0.25)
        threading.Thread(target=inference_thread, daemon=True).start()
