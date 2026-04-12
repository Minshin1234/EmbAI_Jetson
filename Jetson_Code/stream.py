"""
Dual-camera Flask app.

Uses the older standalone CAM0 Arducam logic in a separate worker process while
the main Flask process handles the USB webcam for ArUco workspace tracking.
"""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from flask import Flask, Response, jsonify, request


CAM0_STATE_DIR = Path("/tmp/cam0_stream")
CAM0_FRAME_PATH = CAM0_STATE_DIR / "latest.jpg"
CAM0_STATUS_PATH = CAM0_STATE_DIR / "status.json"
CAM0_TILT_PATH = CAM0_STATE_DIR / "tilt_override.json"
CAM0_DISTANCE_PATH = CAM0_STATE_DIR / "cup_distance.json"  # webcam → cam0

app = Flask(__name__)

camera_ref = None
cam0_worker = None
latest_frame = None
latest_results = None
latest_aruco = None
frame_lock = threading.Lock()
model = None
cv2 = None
np = None
vision_aruco = None
vision_focus_control = None
CAMERA_DEVICE = None
FRAME_HEIGHT = None
FRAME_WIDTH = None
JPEG_QUALITY = None
YOLO_CLASSES = None
vex_serial = None  # Lazy-init VexSerial instance


def cleanup(signum=None, frame=None):
    """Release cameras and stop the worker cleanly."""
    global camera_ref, cam0_worker
    if camera_ref is not None:
        camera_ref.release()
        camera_ref = None
    if cam0_worker is not None and cam0_worker.poll() is None:
        cam0_worker.terminate()
        try:
            cam0_worker.wait(timeout=2)
        except subprocess.TimeoutExpired:
            cam0_worker.kill()
    print("\nCamera released cleanly.", flush=True)
    os._exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def initialize_workspace_model():
    """Load the webcam YOLO model after the CAM1 worker is already running."""
    global model

    from ultralytics import YOLO

    print("Loading YOLO model for workspace webcam...", flush=True)
    model = YOLO("yolov8n.pt")
    print("Warming up YOLO model...", flush=True)
    model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False, classes=YOLO_CLASSES)
    print("YOLO warmup complete.", flush=True)


def initialize_runtime():
    """Load the main-process vision stack after the CAM0 worker is spawned."""
    global cv2, np, vision_aruco, vision_focus_control
    global CAMERA_DEVICE, FRAME_HEIGHT, FRAME_WIDTH, JPEG_QUALITY, YOLO_CLASSES

    # CRITICAL: import torch BEFORE cv2 on Jetson to prevent CUDA heap corruption
    import torch  # noqa: F401
    import cv2 as _cv2
    import numpy as _np

    _cv2.setNumThreads(0)
    print(f"--- Diagnostic Info ---", flush=True)
    print(f"Python version: {sys.version.split()[0]}", flush=True)
    print(f"OpenCV version: {_cv2.__version__}", flush=True)
    print(f"Numpy version:  {_np.__version__}", flush=True)
    print(f"Torch version:  {torch.__version__}", flush=True)
    print(f"------------------------", flush=True)

    import vision.aruco as _vision_aruco
    import vision.focus_control as _vision_focus_control
    from vision.config import (
        CAMERA_DEVICE as _CAMERA_DEVICE,
        FRAME_HEIGHT as _FRAME_HEIGHT,
        FRAME_WIDTH as _FRAME_WIDTH,
        JPEG_QUALITY as _JPEG_QUALITY,
        YOLO_CLASSES as _YOLO_CLASSES,
    )

    cv2 = _cv2
    np = _np
    vision_aruco = _vision_aruco
    vision_focus_control = _vision_focus_control
    CAMERA_DEVICE = _CAMERA_DEVICE
    FRAME_HEIGHT = _FRAME_HEIGHT
    FRAME_WIDTH = _FRAME_WIDTH
    JPEG_QUALITY = _JPEG_QUALITY
    YOLO_CLASSES = _YOLO_CLASSES


def _read_cam0_status():
    if not CAM0_STATUS_PATH.exists():
        return {
            "camera_ready": False,
            "detected": False,
            "class": None,
            "confidence": None,
            "pixel_bbox": None,
            "source": "CAM0",
            "frame_width": None,
            "frame_height": None,
            "fps": None,
        }
    try:
        return json.loads(CAM0_STATUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {
            "camera_ready": False,
            "detected": False,
            "class": None,
            "confidence": None,
            "pixel_bbox": None,
            "source": "CAM0",
            "frame_width": None,
            "frame_height": None,
            "fps": None,
        }


def _open_usb_camera():
    """Open the first working USB webcam, skipping CSI camera nodes."""
    global camera_ref
    # Skip /dev/video0 — it belongs to the CSI sensor (Argus/IMX519).
    # Probing it with V4L2 kills the Argus pipeline in the cam0_worker.
    csi_skip = {0}
    candidates = [CAMERA_DEVICE] + [i for i in range(1, 10) if i != CAMERA_DEVICE]
    for idx in candidates:
        if idx in csi_skip:
            continue
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            continue
        ok, test_frame = cap.read()
        if not ok or test_frame is None:
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        camera_ref = cap
        print(f"Found working webcam at /dev/video{idx}", flush=True)
        return cap
    return None


def webcam_capture_thread():
    """Continuously read the USB webcam."""
    global latest_frame

    camera = _open_usb_camera()
    if camera is None:
        print("ERROR: No working USB webcam found.", flush=True)
        return

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(camera.get(cv2.CAP_PROP_FPS) or 0.0)
    vision_aruco.configure_camera_model(width, height)
    print(f"Webcam opened: {width}x{height} @ {fps}fps", flush=True)

    while True:
        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue
        with frame_lock:
            latest_frame = frame


def webcam_inference_thread():
    """Run webcam YOLO + ArUco inference."""
    global latest_results, latest_aruco

    while True:
        frame = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        if model is None:
            time.sleep(0.01)
            continue

        aruco_data = vision_aruco.detect(frame)
        results = model(frame, verbose=False, classes=YOLO_CLASSES)

        # Share the cup's precise distance with the depth camera worker
        # so it can compute height without depending on tilt estimation
        try:
            if results is not None and len(results[0].boxes) > 0:
                best = results[0].boxes
                best_idx = int(best.conf.argmax())
                bx1, by1, bx2, by2 = [int(v) for v in best.xyxy[best_idx].cpu().numpy()]
                cup_cx = (bx1 + bx2) // 2
                cup_bottom_y = by2
                dist = vision_aruco.get_distance_to_point(cup_cx, cup_bottom_y)
                if dist is not None:
                    CAM0_STATE_DIR.mkdir(parents=True, exist_ok=True)
                    tmp = CAM0_DISTANCE_PATH.with_suffix(".tmp")
                    tmp.write_text(json.dumps({"distance_m": round(dist, 6)}), encoding="utf-8")
                    os.replace(tmp, CAM0_DISTANCE_PATH)
        except Exception:
            pass

        with frame_lock:
            latest_aruco = aruco_data
            latest_results = results


def _draw_workspace_objects(frame, results):
    """Draw webcam YOLO boxes with workspace-relative labels."""
    if results is None:
        return

    cam0_status = _read_cam0_status()
    depth_z_in = cam0_status.get("height_in")

    names = getattr(results[0], "names", {})
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].detach().cpu().numpy()]
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        if isinstance(names, dict):
            cls_name = names.get(cls_id, str(cls_id))
        elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            cls_name = str(names[cls_id])
        else:
            cls_name = str(cls_id)

        label = f"{cls_name} {conf:.2f}"
        ws_pos = vision_aruco.pixel_to_workspace((x1 + x2) // 2, y2)
        ws_pos_in = vision_aruco.workspace_to_inches(ws_pos)
        obj_z_m = vision_aruco.estimate_object_z((x1 + x2) // 2, y1, y2)
        if ws_pos_in is not None and depth_z_in is not None:
            label = f"{cls_name} ({ws_pos_in[0]:.1f}in, {ws_pos_in[1]:.1f}in, z={depth_z_in:.1f}in)"
        elif ws_pos_in is not None and obj_z_m is not None:
            label = f"{cls_name} ({ws_pos_in[0]:.1f}in, {ws_pos_in[1]:.1f}in, h={obj_z_m * 39.3701:.1f}in)"
        elif ws_pos_in is not None:
            label = f"{cls_name} ({ws_pos_in[0]:.1f}in, {ws_pos_in[1]:.1f}in)"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def _placeholder_frame(text="Waiting for webcam..."):
    """Generate a black placeholder frame with a message."""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(placeholder, text, (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    _, buf = cv2.imencode(".jpg", placeholder, [cv2.IMWRITE_JPEG_QUALITY, 50])
    return buf.tobytes()


def generate_webcam_frames():
    """Serve the annotated USB webcam feed."""
    placeholder = _placeholder_frame()

    while True:
        frame = None
        results = None
        aruco_data = None
        with frame_lock:
            if latest_frame is not None:
                frame = latest_frame.copy()
                results = latest_results
                aruco_data = latest_aruco

        if frame is None:
            # Serve a placeholder so the browser doesn't hang
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
            time.sleep(0.5)
            continue

        _draw_workspace_objects(frame, results)
        if aruco_data is not None:
            vision_aruco.draw(frame, aruco_data)

        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


def generate_cam0_frames():
    """Serve the latest JPEG published by the standalone CAM0 worker."""
    placeholder = _placeholder_frame("Waiting for Arducam...")
    last_bytes = None
    while True:
        try:
            if CAM0_FRAME_PATH.exists():
                last_bytes = CAM0_FRAME_PATH.read_bytes()
        except Exception:
            pass

        if last_bytes is None:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + placeholder + b"\r\n"
            time.sleep(0.5)
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + last_bytes + b"\r\n"
        time.sleep(0.02)


def _build_object_payload():
    objects = []
    results = None
    with frame_lock:
        results = latest_results

    cam0_status = _read_cam0_status()
    depth_z_in = cam0_status.get("height_in")

    if results is None:
        return objects

    names = getattr(results[0], "names", {})
    for box in results[0].boxes:
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].detach().cpu().numpy()]
        conf = float(box.conf[0].item())
        cls_id = int(box.cls[0].item())
        if isinstance(names, dict):
            cls_name = names.get(cls_id, str(cls_id))
        elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            cls_name = str(names[cls_id])
        else:
            cls_name = str(cls_id)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        obj_z_m = vision_aruco.estimate_object_z(cx, y1, y2)
        # Project bbox bottom-center onto the table plane for horizontal
        # position.  This gives the true base position of the cup.
        ws_pos = vision_aruco.pixel_to_workspace(cx, y2)
        ws_pos_in = vision_aruco.workspace_to_inches(ws_pos)
        objects.append(
            {
                "class": cls_name,
                "confidence": round(conf, 4),
                "pixel_bbox": [x1, y1, x2, y2],
                "workspace_pos_in": [round(v, 4) for v in ws_pos_in] if ws_pos_in else None,
                "height_in": round(obj_z_m * 39.3701, 4) if obj_z_m is not None else None,
                "z_from_depth_in": depth_z_in,
            }
        )
    return objects


def start_cam0_worker():
    """Launch the standalone CAM0 worker process."""
    global cam0_worker
    CAM0_STATE_DIR.mkdir(parents=True, exist_ok=True)
    cam0_worker = subprocess.Popen(
        [sys.executable, "-m", "vision.cam0_worker"],
        stderr=subprocess.PIPE,
    )
    print(f"[CAM0] Worker subprocess started (PID {cam0_worker.pid})", flush=True)

    def _cam0_watchdog():
        """Monitor the cam0 worker and log if it crashes."""
        global cam0_worker
        while True:
            if cam0_worker is None:
                break
            retcode = cam0_worker.poll()
            if retcode is not None:
                stderr_output = ""
                if cam0_worker.stderr:
                    try:
                        stderr_output = cam0_worker.stderr.read().decode("utf-8", errors="replace")
                    except Exception:
                        pass
                print(f"[CAM0] Worker CRASHED with exit code {retcode}", flush=True)
                if stderr_output:
                    print(f"[CAM0] stderr:\n{stderr_output}", flush=True)
                # Auto-restart
                time.sleep(2)
                print("[CAM0] Restarting worker...", flush=True)
                cam0_worker = subprocess.Popen(
                    [sys.executable, "-m", "vision.cam0_worker"],
                    stderr=subprocess.PIPE,
                )
                print(f"[CAM0] Worker restarted (PID {cam0_worker.pid})", flush=True)
            time.sleep(1)

    threading.Thread(target=_cam0_watchdog, daemon=True).start()


@app.after_request
def disable_cache(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def index():
    focus = vision_focus_control.get_focus_state()
    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <title>Jetson Camera Stream</title>
        <style>
          body {{ font-family: sans-serif; text-align: center; background: #111827; color: #e5e7eb; margin: 0; padding: 20px; }}
          .streams, .panels {{ display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; }}
          .card {{ background: #1f2937; border: 1px solid #374151; border-radius: 10px; padding: 14px; }}
          img {{ border: 2px solid #374151; border-radius: 8px; background: #000; }}
          .panel {{ min-width: 320px; text-align: left; font-family: monospace; font-size: 13px; }}
          .found {{ color: #34d399; }}
          .missing {{ color: #f87171; }}
          .focus-wrap {{ margin-top: 10px; }}
          input[type=range] {{ width: 100%; }}
        </style>
        <script>
          function updateFocus(val) {{
            fetch('/set_focus', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
              body: 'value=' + encodeURIComponent(val)
            }})
            .then(r => r.json())
            .then(data => {{
              const nextVal = data.focus && data.focus.value != null ? data.focus.value : val;
              document.getElementById('focus_label').innerText = nextVal;
            }});
          }}

          function updateTilt(val) {{
            fetch('/set_tilt', {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/x-www-form-urlencoded' }},
              body: 'value=' + encodeURIComponent(val)
            }})
            .then(r => r.json())
            .then(data => {{
              document.getElementById('tilt_label').innerText = data.pitch_deg != null ? data.pitch_deg : val;
            }});
          }}

          function updateStatus() {{
            fetch('/aruco_status')
              .then(r => r.json())
              .then(data => {{
                let ws = '';
                const ids = [0, 1, 2, 3];
                ids.forEach(id => {{
                  const pos = data.marker_workspace_in && data.marker_workspace_in[id];
                  const held = data.memory_ids && data.memory_ids.includes(id) ? ' [memory]' : '';
                  ws += pos
                    ? `<div class="found">WS-${{id}} (${{
                        pos[0].toFixed(2)
                      }}in, ${{pos[1].toFixed(2)}}in, z=${{pos[2].toFixed(2)}}in)${{held}}</div>`
                    : `<div class="missing">WS-${{id}} missing${{held}}</div>`;
                }});
                ws += `<div>Workspace: <span class="${{data.workspace_found ? 'found' : 'missing'}}">${{data.workspace_found ? 'LOCKED' : 'SEARCHING'}}</span></div>`;
                document.getElementById('workspace-info').innerHTML = ws;

                let objs = '';
                if (data.objects && data.objects.length > 0) {{
                  data.objects.forEach(obj => {{
                    const pos = obj.workspace_pos_in;
                    const z = obj.z_from_depth_in != null
                      ? `, z=${{obj.z_from_depth_in.toFixed(2)}}in`
                      : (obj.height_in != null ? `, h=${{obj.height_in.toFixed(2)}}in` : '');
                    objs += pos
                      ? `<div class="found">${{obj.class}}: (${{pos[0].toFixed(2)}}in, ${{pos[1].toFixed(2)}}in${{z}})</div>`
                      : `<div>${{obj.class}}: position unavailable</div>`;
                  }});
                }} else {{
                  objs = '<div class="missing">No webcam detections</div>';
                }}
                document.getElementById('objects-info').innerHTML = objs;

                const cam0 = data.cam0 || {{}};
                let cam0Html = `<div>Source: ${{cam0.source || 'CAM0'}}</div>`;
                cam0Html += `<div>Status: <span class="${{cam0.camera_ready ? 'found' : 'missing'}}">${{cam0.camera_ready ? 'LIVE' : 'OFFLINE'}}</span></div>`;
                cam0Html += `<div>Workspace: <span class="${{cam0.workspace_found ? 'found' : 'missing'}}">${{cam0.workspace_found ? 'LOCKED' : 'SEARCHING'}}</span></div>`;
                if (cam0.camera_tilt_deg != null) {{
                  cam0Html += `<div>Camera tilt: ${{cam0.camera_tilt_deg.toFixed(2)}} deg</div>`;
                }}
                const focus = data.focus || {{}};
                cam0Html += `<div>Focus bus: ${{focus.bus != null ? focus.bus : 'n/a'}}</div>`;
                cam0Html += `<div>Focus value: ${{focus.value != null ? focus.value : 'n/a'}}</div>`;
                if (focus.error) {{
                  cam0Html += `<div class="missing">Focus error: ${{focus.error}}</div>`;
                }}
                if (cam0.detected && cam0.pixel_bbox) {{
                  cam0Html += `<div class="found">Cup bbox: [${{cam0.pixel_bbox.join(', ')}}]</div>`;
                  if (cam0.height_in != null) {{
                    cam0Html += `<div class="found">Cup z: ${{cam0.height_in.toFixed(2)}}in</div>`;
                  }}
                }} else {{
                  cam0Html += `<div class="missing">No cup detected yet</div>`;
                }}
                document.getElementById('cam0-info').innerHTML = cam0Html;
              }});
          }}

          function goToTarget() {{
            const btn = document.getElementById('go-target-btn');
            btn.disabled = true;
            btn.innerText = 'Sending...';
            fetch('/go_to_target', {{ method: 'POST' }})
              .then(r => r.json())
              .then(data => {{
                let msg = '';
                if (data.error) {{
                  msg = 'Error: ' + data.error;
                }} else {{
                  msg = 'IK: [' + data.angles_deg.map(a => a.toFixed(1)).join(', ') + ']°';
                  if (data.serial_sent) msg += ' ✓ Sent';
                  else msg += ' ✗ Serial failed';
                }}
                document.getElementById('ik-result').innerText = msg;
                btn.disabled = false;
                btn.innerText = 'Go to Target';
              }})
              .catch(err => {{
                document.getElementById('ik-result').innerText = 'Request failed';
                btn.disabled = false;
                btn.innerText = 'Go to Target';
              }});
          }}

          setInterval(updateStatus, 250);
          updateStatus();
        </script>
      </head>
      <body>
        <h1>Jetson Camera Stream</h1>
        <p>USB webcam workspace tracking + standalone CAM0 Arducam stream</p>
        <div class="streams">
          <div class="card">
            <h3>Workspace Webcam</h3>
            <img src="/video_feed" width="640" height="480"/>
          </div>
          <div class="card">
            <h3>CAM0 Arducam</h3>
            <img src="/depth_feed" width="640" height="480"/>
            <div class="focus-wrap">
              <label for="focus_slider">Manual Focus</label>
              <input
                type="range"
                id="focus_slider"
                min="{int(focus['min'])}"
                max="{int(focus['max'])}"
                value="{int(focus['value'])}"
                onchange="updateFocus(this.value)"
                oninput="document.getElementById('focus_label').innerText = this.value"
              />
              <div>Value: <span id="focus_label">{int(focus['value'])}</span></div>
            </div>
            <div class="focus-wrap">
              <label for="tilt_slider">Camera Tilt (degrees)</label>
              <input
                type="range"
                id="tilt_slider"
                min="-30"
                max="90"
                step="0.5"
                value="0"
                onchange="updateTilt(this.value)"
                oninput="document.getElementById('tilt_label').innerText = this.value"
              />
              <div>Tilt: <span id="tilt_label">0</span>&deg;</div>
            </div>
          </div>
        </div>
        <div class="panels" style="margin-top: 18px;">
          <div class="card panel">
            <h3>Workspace</h3>
            <div id="workspace-info">Connecting...</div>
          </div>
          <div class="card panel">
            <h3>Webcam Objects</h3>
            <div id="objects-info">Connecting...</div>
          </div>
          <div class="card panel">
            <h3>CAM0 Status</h3>
            <div id="cam0-info">Connecting...</div>
          </div>
          <div class="card panel">
            <h3>Arm Control</h3>
            <button id="go-target-btn" onclick="goToTarget()" style="
              padding: 10px 24px; font-size: 15px; cursor: pointer;
              background: #2563eb; color: white; border: none; border-radius: 6px;
              margin-bottom: 8px;
            ">Go to Target</button>
            <div id="ik-result" style="margin-top: 6px; color: #94a3b8;">No command sent yet</div>
          </div>
        </div>
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(generate_webcam_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/depth_feed")
def depth_feed():
    return Response(generate_cam0_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/aruco_status")
def aruco_status():
    aruco_data = None
    with frame_lock:
        aruco_data = latest_aruco

    if aruco_data is None:
        aruco_out = {
            "workspace_found": False,
            "workspace_corners": {},
            "marker_workspace_in": {},
            "memory_ids": [],
        }
    else:
        marker_workspace_in = {}
        for key, value in aruco_data["marker_workspace"].items():
            converted = vision_aruco.workspace_to_inches(value)
            if converted is not None:
                marker_workspace_in[int(key)] = [round(v, 4) for v in converted]
        aruco_out = {
            "workspace_found": aruco_data["workspace_poly"] is not None,
            "workspace_corners": {
                int(k): [int(v) for v in val]
                for k, val in aruco_data["workspace_corners"].items()
            },
            "marker_workspace_in": marker_workspace_in,
            "memory_ids": [int(v) for v in aruco_data.get("memory_ids", [])],
        }

    return jsonify({
        **aruco_out,
        "objects": _build_object_payload(),
        "cam0": _read_cam0_status(),
        "focus": vision_focus_control.get_focus_state(),
    })


@app.route("/set_focus", methods=["POST"])
def set_focus():
    value = int(request.form.get("value", 0))
    return jsonify(vision_focus_control.set_focus_value(value))


@app.route("/set_tilt", methods=["POST"])
def set_tilt():
    """Write a tilt override that the CAM0 worker reads at runtime."""
    pitch = float(request.form.get("value", 0))
    CAM0_STATE_DIR.mkdir(parents=True, exist_ok=True)
    CAM0_TILT_PATH.write_text(json.dumps({"pitch_deg": pitch}), encoding="utf-8")
    return jsonify({"ok": True, "pitch_deg": pitch})


@app.route("/go_to_target", methods=["POST"])
def go_to_target():
    """Compute IK for the best detected object and send angles over serial."""
    global vex_serial

    # 1. Get the best detected object's workspace position
    objects = _build_object_payload()
    if not objects:
        return jsonify({"error": "No objects detected"})

    # Find the highest-confidence object with a valid workspace position
    best = None
    for obj in sorted(objects, key=lambda o: o["confidence"], reverse=True):
        if obj.get("workspace_pos_in") is not None:
            best = obj
            break

    if best is None:
        return jsonify({"error": "No object has a workspace position (ArUco not locked)"})

    # 2. Convert workspace position from inches to meters for IK
    ws_in = best["workspace_pos_in"]
    z_in = best.get("z_from_depth_in") or best.get("height_in") or 0.0
    
    # The Arducam detects the TOP of the cup. 
    # To grab the middle, we need to reach lower (deeper).
    # The user requested 2 inches more. Because our IK Z axis logic plunges the arm
    # deeper when z_in increases (due to the -z_in negation), adding 2 inches works perfectly!
    Z_REACH_OFFSET_IN = 2.0
    target_z_in = z_in + Z_REACH_OFFSET_IN

    # Negate Z: ikpy chain +Z is down.
    z_m = -target_z_in / 39.3701
    x_m = ws_in[0] / 39.3701
    y_m = ws_in[1] / 39.3701

    # Rotate workspace→IK frame:
    # Arm faces -Y in workspace. So Forward (+IK_X) = -WS_Y.
    # Arm's Left (+IK_Y) is -WS_X (since +WS_X is Right).
    ik_x_raw = -y_m  # forward
    ik_y_raw = -x_m  # left
    ik_z = z_m        # down

    # ── Angular alignment correction ──
    # The WS axes are rotated a few degrees relative to the IK frame,
    # causing the arm to drift LEFT for far targets.  A positive angle
    # rotates the IK target to correct this distance-dependent drift.
    import math
    WS_ROTATION_DEG = 8  # increase if still drifting left at far range
    _theta = math.radians(WS_ROTATION_DEG)
    _cos_t = math.cos(_theta)
    _sin_t = math.sin(_theta)
    ik_x = ik_x_raw * _cos_t - ik_y_raw * _sin_t
    ik_y = ik_x_raw * _sin_t + ik_y_raw * _cos_t

    # ── Constant lateral correction ──
    # Positive = shift target to arm's RIGHT (fixes leftward bias at close range)
    IK_Y_CORRECTION_M = 0.053   # +2.1 inches
    ik_y += IK_Y_CORRECTION_M

    # ── Forward reach correction ──
    # The arm undershoots (falls short of the cup). Nudge the target forward.
    IK_X_CORRECTION_M = 0.102   # +2.0 inches forward
    ik_x += IK_X_CORRECTION_M

    print(f"[IK] Workspace pos (in): x={ws_in[0]:.2f}, y={ws_in[1]:.2f}, z={z_in:.2f}", flush=True)
    print(f"[IK] IK target (m): x={ik_x:.4f}, y={ik_y:.4f}, z={ik_z:.4f} (rot={WS_ROTATION_DEG}° fwd={IK_X_CORRECTION_M:+.4f} lat={IK_Y_CORRECTION_M:+.4f})", flush=True)

    # 3. Connect to VEX serial if not already connected
    if vex_serial is None:
        from vex.control import VexSerial
        vex_serial = VexSerial()
        vex_serial.connect()
    elif not vex_serial.connected:
        vex_serial.connect()

    # 4. Request current joint positions from the V5 brain
    current_angles = None
    if vex_serial.connected:
        current_angles = vex_serial.request_status(timeout=1.0)
        if current_angles is not None:
            print(f"[IK] Current joint angles: {current_angles}", flush=True)
        else:
            print("[IK] No STATUS response — using default seed", flush=True)

    # 5. Compute IK (seeded with current angles if available)
    try:
        from vex.ik_solver import get_ik_angles
        angles_deg = get_ik_angles(ik_x, ik_y, ik_z, current_angles_deg=current_angles)
    except Exception as e:
        return jsonify({"error": f"IK failed: {str(e)}"})

    # 6. Raise-then-arc motion: go up first, then arc down to target
    #    with J4 (wrist pitch) locked at 90° (pointing straight down).
    RAISED_J1 = 0.0    # shoulder straight up
    RAISED_J2 = 0.0    # elbow straight
    RAISED_J3 = 0.0    # wrist yaw centered
    GRIP_DOWN = 90.0   # J4 = gripper pointing down
    DESCENT_STEPS = 8  # number of interpolation steps
    STEP_DELAY = 0.15  # seconds between steps

    target_J0 = angles_deg[0]  # turntable angle from IK
    final_angles = list(angles_deg)
    final_angles[4] = GRIP_DOWN  # force J4 to 90° in final pose too

    # Step A: raise the arm straight up, rotate turntable to face target
    raised_angles = [target_J0, RAISED_J1, RAISED_J2, RAISED_J3, GRIP_DOWN]
    print(f"[IK] Step 0/{DESCENT_STEPS}: RAISE {[round(a,1) for a in raised_angles]}", flush=True)

    serial_ok = False
    if vex_serial.connected:
        vex_serial.send_all_joints_deg(raised_angles)
    time.sleep(1.0)  # wait for arm to reach raised position

    # Step B: arc from raised position to final target in smooth steps
    for step in range(1, DESCENT_STEPS + 1):
        t = step / DESCENT_STEPS  # 0→1 interpolation factor
        step_angles = [
            raised_angles[j] + t * (final_angles[j] - raised_angles[j])
            for j in range(5)
        ]
        step_angles[4] = GRIP_DOWN  # keep J4 locked at 90° throughout
        print(f"[IK] Step {step}/{DESCENT_STEPS}: ARC {[round(a,1) for a in step_angles]}", flush=True)
        if vex_serial.connected:
            serial_ok = vex_serial.send_all_joints_deg(step_angles)
        time.sleep(STEP_DELAY)

    return jsonify({
        "ok": True,
        "target_class": best["class"],
        "target_pos_in": ws_in,
        "target_z_in": z_in,
        "target_pos_m": [round(x_m, 4), round(y_m, 4), round(z_m, 4)],
        "angles_deg": [round(a, 2) for a in final_angles],
        "current_angles_deg": current_angles,
        "serial_sent": serial_ok,
        "serial_connected": vex_serial.connected,
    })


@app.route("/calibrate", methods=["POST"])
def calibrate():
    """Compare FK-predicted gripper position vs actual gripper marker position."""
    from vision.config import M_TO_IN
    from vex.ik_solver import compute_fk_position
    from vision import aruco as aruco_mod

    # 1. Get current joint angles (from VEX or default calibration position)
    default_angles = [0, -34, 90, 0, 90]
    current_angles = None

    global vex_serial
    if vex_serial is not None and vex_serial.connected:
        current_angles = vex_serial.request_status(timeout=1.0)

    joint_angles = current_angles if current_angles else default_angles
    print(f"[CAL] Joints: {joint_angles}", flush=True)

    # 2. FK: compute where the gripper SHOULD be (in IK frame, meters)
    fk_pos_m = compute_fk_position(joint_angles)
    fk_pos_in = [p * M_TO_IN for p in fk_pos_m]
    print(f"[CAL] FK predicted (m):  {[round(p,4) for p in fk_pos_m]}", flush=True)
    print(f"[CAL] FK predicted (in): {[round(p,2) for p in fk_pos_in]}", flush=True)

    # 3. Read gripper marker workspace position
    gripper_actual_in = None
    if aruco_mod._last_gripper_ws is not None:
        gripper_actual_in = [round(v * M_TO_IN, 2) for v in aruco_mod._last_gripper_ws]

    # Grab arm marker position for reference
    arm_ws_in = None
    if aruco_mod._last_arm_pose_3d is not None:
        # arm_pose_3d is camera-space; use marker_workspace from last detect
        pass

    result = {
        "joint_angles_deg": joint_angles,
        "fk_predicted_m": [round(p, 4) for p in fk_pos_m],
        "fk_predicted_in": [round(p, 2) for p in fk_pos_in],
        "gripper_marker_in": gripper_actual_in,
    }

    if gripper_actual_in:
        delta = [round(fk_pos_in[i] - gripper_actual_in[i], 2) for i in range(3)]
        result["delta_in"] = delta
        print(f"[CAL] Gripper actual (in): {gripper_actual_in}", flush=True)
        print(f"[CAL] Delta (FK - actual): {delta} inches", flush=True)
    else:
        print("[CAL] Gripper marker (ID 5) not detected! Place ArUco ID 5 on the gripper.", flush=True)
        result["error"] = "Gripper marker (ID 5) not detected"

    return jsonify(result)


def main():
    # 1. Load main-process vision stack first (torch, cv2, aruco)
    initialize_runtime()

    # 2. Load workspace YOLO model (GPU-intensive — do this BEFORE cam0 worker)
    initialize_workspace_model()

    # 3. Start webcam capture threads
    threading.Thread(target=webcam_capture_thread, daemon=True).start()
    threading.Thread(target=webcam_inference_thread, daemon=True).start()

    # 4. Now start the cam0 worker — it will load its own YOLO
    #    This is done AFTER the main model is loaded to avoid GPU memory contention
    start_cam0_worker()

    # 5. Start Flask (blocking)
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    main()
