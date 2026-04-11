"""
Standalone CAM0 capture worker.

Runs the Arducam in its own process: detects ArUco markers for distance
calibration, runs YOLO to find cups, and estimates the cup's physical
height using the ArUco-derived camera pose + webcam's cup X/Y position.

Publishes results to /tmp/cam0_stream/ for the main Flask process to read.
"""

# CRITICAL: import torch BEFORE cv2 on Jetson to prevent CUDA heap corruption
import torch  # noqa: F401

import json
import math
import os
from pathlib import Path
import signal
import time

import cv2
import numpy as np

from vision.config import (
    ARUCO_DICT,
    DEPTH_CAMERA_FPS,
    DEPTH_CAMERA_FLIP,
    DEPTH_CAMERA_FOV_H_DEG,
    DEPTH_CAMERA_HEIGHT_M,
    DEPTH_CAMERA_PITCH_DEG,
    DEPTH_CAMERA_SENSOR_ID,
    DEPTH_CAMERA_SENSOR_MODE,
    DEPTH_FRAME_HEIGHT,
    DEPTH_FRAME_WIDTH,
    DEPTH_HEIGHT_CALIBRATION,
    DEPTH_PREVIEW_HEIGHT,
    DEPTH_PREVIEW_WIDTH,
    YOLO_CLASSES,
    DEPTH_YOLO_EVERY_N_FRAMES,
    M_TO_IN,
    MARKER_SIZE_M,
    WORKSPACE_IDS,
    YOLO_MODEL,
    get_camera_model_for_fov,
)
from vision.depth_forward import estimate_object_height_m


cv2.setNumThreads(0)

STATE_DIR = Path("/tmp/cam0_stream")
FRAME_PATH = STATE_DIR / "latest.jpg"
STATUS_PATH = STATE_DIR / "status.json"
TILT_PATH = STATE_DIR / "tilt_override.json"
DISTANCE_PATH = STATE_DIR / "cup_distance.json"  # Written by webcam (cup position)

camera_ref = None
running = True

# ── ArUco setup ──────────────────────────────────────────────────────
_aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_aruco_params = cv2.aruco.DetectorParameters()
# Tune for shallow viewing angles (marker seen at an angle)
_aruco_params.adaptiveThreshWinSizeMin = 3
_aruco_params.adaptiveThreshWinSizeMax = 23
_aruco_params.adaptiveThreshWinSizeStep = 5
_aruco_params.minMarkerPerimeterRate = 0.01    # Detect smaller markers
_aruco_params.maxMarkerPerimeterRate = 4.0
_aruco_params.perspectiveRemovePixelPerCell = 6
_aruco_params.perspectiveRemoveIgnoredMarginPerCell = 0.2
_aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

# 3D points of one marker (Z=0 plane)
_half = MARKER_SIZE_M / 2.0
_MARKER_OBJ_POINTS = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)

# EMA smoothing factor for distance
_EMA_ALPHA = 0.3


def _write_status(payload):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = STATUS_PATH.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(tmp_path, STATUS_PATH)


def _read_tilt_override():
    try:
        if TILT_PATH.exists():
            data = json.loads(TILT_PATH.read_text(encoding="utf-8"))
            return float(data.get("pitch_deg", DEPTH_CAMERA_PITCH_DEG))
    except Exception:
        pass
    return DEPTH_CAMERA_PITCH_DEG


def _read_webcam_cup_position():
    """Read the cup's workspace position from the webcam.

    Returns (distance_m,) or None. The webcam writes this after computing
    the cup's 3D distance using ArUco workspace calibration.
    """
    try:
        if DISTANCE_PATH.exists():
            data = json.loads(DISTANCE_PATH.read_text(encoding="utf-8"))
            d = data.get("distance_m")
            if d is not None and d > 0:
                return float(d)
    except Exception:
        pass
    return None


def _detect_aruco(frame, cam_matrix, dist_coeffs):
    """Detect ArUco markers and return pose info for closest workspace marker.

    Returns: (distance_m, tvec, rvec, marker_id, corners) or all None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Enhance contrast for better detection at shallow angles
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    corners, ids, _ = cv2.aruco.detectMarkers(gray, _aruco_dict, parameters=_aruco_params)

    if ids is None or len(ids) == 0:
        return None, None, None, None, None

    best_dist = None
    best_tvec = None
    best_rvec = None
    best_id = None
    best_corners = None

    for i, marker_id in enumerate(ids.flatten()):
        if int(marker_id) not in WORKSPACE_IDS:
            continue

        marker_corners = corners[i].reshape(4, 2).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            _MARKER_OBJ_POINTS, marker_corners,
            cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue

        dist = float(np.linalg.norm(tvec))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_tvec = tvec.flatten()
            best_rvec = rvec.flatten()
            best_id = int(marker_id)
            best_corners = marker_corners

    return best_dist, best_tvec, best_rvec, best_id, best_corners


def _compute_distance_to_pixel(px, py, rvec, tvec, cam_matrix):
    """Compute distance from camera to a point on the workspace plane at pixel (px, py).

    Uses the ArUco marker's solvePnP result to define the workspace plane,
    then intersects the pixel's camera ray with that plane.

    This gives the ACTUAL distance to where the cup sits on the workspace,
    not the distance to the marker (which may be much farther).
    """
    # Convert rvec/tvec to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # The workspace plane passes through tvec with normal = R @ [0, 0, 1]
    # (the Z axis of the marker frame, transformed to camera frame)
    plane_normal = R @ np.array([0, 0, 1], dtype=np.float64)
    plane_point = tvec.flatten().astype(np.float64)

    # Build the ray from the camera through pixel (px, py)
    fx = cam_matrix[0, 0]
    fy = cam_matrix[1, 1]
    cx = cam_matrix[0, 2]
    cy = cam_matrix[1, 2]
    ray = np.array([
        (float(px) - cx) / fx,
        (float(py) - cy) / fy,
        1.0,
    ], dtype=np.float64)
    ray = ray / np.linalg.norm(ray)  # normalize

    # Ray-plane intersection: t = dot(plane_point, plane_normal) / dot(ray, plane_normal)
    denom = np.dot(ray, plane_normal)
    if abs(denom) < 1e-8:
        return None  # Ray is parallel to plane

    t = np.dot(plane_point, plane_normal) / denom
    if t <= 0:
        return None  # Intersection is behind the camera

    return float(t)  # This is the distance along the ray to the plane


def _offline_status():
    return {
        "camera_ready": False,
        "detected": False,
        "class": None,
        "confidence": None,
        "pixel_bbox": None,
        "workspace_found": False,
        "aruco_distance_m": None,
        "aruco_marker_id": None,
        "height_in": None,
        "source": f"CAM{DEPTH_CAMERA_SENSOR_ID}",
        "frame_width": DEPTH_PREVIEW_WIDTH,
        "frame_height": DEPTH_PREVIEW_HEIGHT,
        "fps": float(DEPTH_CAMERA_FPS),
        "updated_at": time.time(),
    }


def cleanup(signum=None, frame=None):
    global camera_ref, running
    running = False
    if camera_ref is not None:
        camera_ref.release()
        camera_ref = None
    _write_status(_offline_status())
    os._exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def gstreamer_pipeline(
    sensor_id=DEPTH_CAMERA_SENSOR_ID,
    sensor_mode=DEPTH_CAMERA_SENSOR_MODE,
    capture_width=DEPTH_FRAME_WIDTH,
    capture_height=DEPTH_FRAME_HEIGHT,
    display_width=DEPTH_PREVIEW_WIDTH,
    display_height=DEPTH_PREVIEW_HEIGHT,
    framerate=DEPTH_CAMERA_FPS,
    flip_method=2 if DEPTH_CAMERA_FLIP else 0,
):
    return (
        "nvarguscamerasrc sensor-id=%d sensor-mode=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
        % (
            sensor_id, sensor_mode,
            capture_width, capture_height, framerate,
            flip_method,
            display_width, display_height,
        )
    )


def main():
    global camera_ref

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _write_status(_offline_status())

    # ── Camera model ─────────────────────────────────────────────────
    cam_matrix, dist_coeffs = get_camera_model_for_fov(
        DEPTH_PREVIEW_WIDTH, DEPTH_PREVIEW_HEIGHT, DEPTH_CAMERA_FOV_H_DEG,
    )
    focal_length_px = cam_matrix[0, 0]
    print(f"[CAM0] Arducam: fx={focal_length_px:.1f}px, FOV={DEPTH_CAMERA_FOV_H_DEG}°", flush=True)

    # ── YOLO ─────────────────────────────────────────────────────────
    from ultralytics import YOLO

    print("[CAM0] Loading YOLO model...", flush=True)
    yolo_model = YOLO(YOLO_MODEL)
    yolo_model(
        np.zeros((DEPTH_PREVIEW_HEIGHT, DEPTH_PREVIEW_WIDTH, 3), dtype=np.uint8),
        verbose=False, classes=YOLO_CLASSES,
    )
    print("[CAM0] YOLO ready.", flush=True)

    # ── Open camera ──────────────────────────────────────────────────
    pipeline_str = gstreamer_pipeline()
    print(f"[CAM0] Opening GStreamer pipeline:\n  {pipeline_str}", flush=True)

    camera = cv2.VideoCapture(pipeline_str, cv2.CAP_GSTREAMER)
    camera_ref = camera
    if not camera.isOpened():
        print("[CAM0] ERROR: Could not open Arducam.", flush=True)
        _write_status(_offline_status())
        return

    print(f"[CAM0] Camera opened: {DEPTH_PREVIEW_WIDTH}x{DEPTH_PREVIEW_HEIGHT} "
          f"@ {DEPTH_CAMERA_FPS}fps", flush=True)

    # ── Main loop state ──────────────────────────────────────────────
    frame_count = 0
    last_bbox = None
    last_cls_name = None
    last_conf = None
    last_height_in = None
    last_raw_height_in = None
    last_all_boxes = []  # List of [x1,y1,x2,y2, cls_name, conf, height_in]
    smooth_aruco_dist = None    # EMA-smoothed distance (meters)
    last_aruco_id = None
    last_aruco_corners = None

    while running:
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        frame_count += 1
        frame_h, frame_w = frame.shape[:2]

        # ── ArUco detection (every frame — very cheap) ───────────────
        aruco_dist, aruco_tvec, aruco_rvec, aruco_id, aruco_corners = (
            _detect_aruco(frame, cam_matrix, dist_coeffs)
        )
        if aruco_dist is not None:
            # EMA smoothing to reduce flicker
            if smooth_aruco_dist is None:
                smooth_aruco_dist = aruco_dist
            else:
                smooth_aruco_dist = (
                    _EMA_ALPHA * aruco_dist + (1 - _EMA_ALPHA) * smooth_aruco_dist
                )
            last_aruco_id = aruco_id
            last_aruco_corners = aruco_corners

        # ── YOLO every N frames ──────────────────────────────────────
        if frame_count % DEPTH_YOLO_EVERY_N_FRAMES == 0:
            results = yolo_model(frame, verbose=False, classes=YOLO_CLASSES)
            boxes = results[0].boxes
            names = results[0].names

            if len(boxes) > 0:
                best_idx = int(boxes.conf.argmax())
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[best_idx].cpu().numpy()]
                cls_id = int(boxes.cls[best_idx].item())
                conf = float(boxes.conf[best_idx].item())
                cls_name = (names.get(cls_id, str(cls_id))
                            if isinstance(names, dict) else str(cls_id))

                last_bbox = [x1, y1, x2, y2]
                last_cls_name = cls_name
                last_conf = conf
                bbox_height_px = abs(y2 - y1)

                # ── Height estimation for the best bbox ──────────────
                cup_dist = None
                height_source = "none"

                if aruco_rvec is not None and aruco_tvec is not None:
                    cup_cx = (x1 + x2) // 2
                    cup_bottom_y = y2
                    cup_dist = _compute_distance_to_pixel(
                        cup_cx, cup_bottom_y,
                        aruco_rvec, aruco_tvec, cam_matrix,
                    )
                    height_source = "aruco-ray"

                if cup_dist is None:
                    webcam_dist = _read_webcam_cup_position()
                    if webcam_dist is not None and webcam_dist > 0:
                        cup_dist = webcam_dist
                        height_source = "webcam"

                if cup_dist is not None and cup_dist > 0:
                    raw_height_m = (bbox_height_px * cup_dist) / focal_length_px
                    height_m = raw_height_m * DEPTH_HEIGHT_CALIBRATION
                    last_height_in = round(height_m * M_TO_IN, 2)
                    last_raw_height_in = round(raw_height_m * M_TO_IN, 2)
                else:
                    pitch = _read_tilt_override()
                    height_m = estimate_object_height_m(
                        bbox_top_y=y1, bbox_bottom_y=y2,
                        frame_height=frame_h, frame_width=frame_w,
                        camera_height_m=DEPTH_CAMERA_HEIGHT_M,
                        camera_fov_h_deg=DEPTH_CAMERA_FOV_H_DEG,
                        pitch_deg=pitch,
                    )
                    if height_m is not None:
                        last_raw_height_in = round(height_m * M_TO_IN, 2)
                        height_m *= DEPTH_HEIGHT_CALIBRATION
                        last_height_in = round(height_m * M_TO_IN, 2)
                    else:
                        last_height_in = None
                        last_raw_height_in = None
                    height_source = "tilt"

                # Store all boxes for drawing
                last_all_boxes = []
                for i in range(len(boxes)):
                    bx1, by1, bx2, by2 = [int(v) for v in boxes.xyxy[i].cpu().numpy()]
                    b_cls_id = int(boxes.cls[i].item())
                    b_conf = float(boxes.conf[i].item())
                    b_cls_name = names.get(b_cls_id, str(b_cls_id)) if isinstance(names, dict) else str(b_cls_id)
                    
                    # For non-best objects, we don't calculate height to save CPU
                    b_height = last_height_in if i == best_idx else None
                    last_all_boxes.append({
                        "bbox": [bx1, by1, bx2, by2],
                        "cls_name": b_cls_name,
                        "conf": b_conf,
                        "height": b_height
                    })
            else:
                last_bbox = None
                last_cls_name = None
                last_conf = None
                last_height_in = None
                last_raw_height_in = None
                last_all_boxes = []

        # ── Draw annotations ─────────────────────────────────────────
        cv2.putText(frame, "CAM0 Depth View", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        # ArUco info
        if smooth_aruco_dist is not None:
            dist_in = smooth_aruco_dist * M_TO_IN
            cv2.putText(frame, f"ArUco #{last_aruco_id}: {dist_in:.1f}in ({smooth_aruco_dist:.3f}m)",
                        (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        else:
            cv2.putText(frame, "ArUco: not visible",
                        (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

        # ArUco marker outline
        if aruco_corners is not None:
            pts = aruco_corners.reshape(-1, 1, 2).astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)

        # YOLO bboxes
        for item in last_all_boxes:
            bx1, by1, bx2, by2 = item["bbox"]
            color = (0, 255, 0)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            label = f"{item['cls_name']} {item['conf']:.2f}"
            if item['height'] is not None:
                label += f" | h={item['height']:.1f}in"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (bx1, by1 - th - 8), (bx1 + tw, by1), color, -1)
            cv2.putText(frame, label, (bx1, by1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        # Debug: show raw vs calibrated height
        if last_bbox is not None:
            dbg = f"raw={last_raw_height_in}in cal={DEPTH_HEIGHT_CALIBRATION:.2f}"
            cv2.putText(frame, dbg, (12, frame_h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # ── Publish ──────────────────────────────────────────────────
        status = {
            "camera_ready": True,
            "detected": last_bbox is not None,
            "class": last_cls_name,
            "confidence": round(last_conf, 4) if last_conf is not None else None,
            "pixel_bbox": last_bbox,
            "workspace_found": smooth_aruco_dist is not None,
            "aruco_distance_m": round(smooth_aruco_dist, 4) if smooth_aruco_dist else None,
            "aruco_marker_id": last_aruco_id,
            "height_in": last_height_in,
            "source": f"CAM{DEPTH_CAMERA_SENSOR_ID}",
            "frame_width": frame_w,
            "frame_height": frame_h,
            "fps": float(DEPTH_CAMERA_FPS),
            "updated_at": time.time(),
        }

        ok_enc, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok_enc:
            tmp_frame = FRAME_PATH.with_suffix(".tmp")
            tmp_frame.write_bytes(buffer.tobytes())
            os.replace(tmp_frame, FRAME_PATH)
            _write_status(status)

        time.sleep(0.005)


if __name__ == "__main__":
    main()
