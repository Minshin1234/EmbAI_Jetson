"""
Configuration constants for the Jetson camera stream.
"""

import json
from pathlib import Path

import cv2
import numpy as np

# ── Camera ────────────────────────────────────────────────────────────
CAMERA_DEVICE = 2        # USB webcam /dev/video index (/dev/video0-1 are CSI, skip them)
FRAME_WIDTH   = 640
FRAME_HEIGHT  = 480
JPEG_QUALITY  = 70       # MJPEG encoding quality (0-100)

# Secondary Jetson CSI camera used only for forward-distance estimation.
DEPTH_CAMERA_ENABLED = True
# The Arducam is currently connected to CAM0 and should be addressed through
# Argus sensor-id 0 on this Jetson.
DEPTH_CAMERA_SENSOR_ID = 0
DEPTH_CAMERA_SENSOR_MODE = 0  # 4656x3496 @ 9fps full-sensor mode on the IMX519
DEPTH_FRAME_WIDTH = 4656
DEPTH_FRAME_HEIGHT = 3496
DEPTH_CAMERA_FPS = 9
DEPTH_PREVIEW_WIDTH = 640
DEPTH_PREVIEW_HEIGHT = 480
DEPTH_CAMERA_FLIP = True
DEPTH_CAMERA_FOV_H_DEG = 60.0   # Arducam IMX519 horizontal FOV
DEPTH_CAMERA_HEIGHT_M = 0.10
DEPTH_CAMERA_PITCH_DEG = 0.0
DEPTH_YOLO_EVERY_N_FRAMES = 5
DEPTH_FOCUS_ENABLED = True
DEPTH_FOCUS_BUS = 10
DEPTH_FOCUS_MIN = 0
DEPTH_FOCUS_MAX = 1000
DEPTH_FOCUS_DEFAULT = 268

# Height calibration factor: YOLO bboxes include padding beyond the real object.
# Set this to (known_cup_height / raw_reading).
# Re-tuned after MARKER_SIZE_M changed from 0.075 to 0.045.
DEPTH_HEIGHT_CALIBRATION = 0.65

# Approximate camera intrinsics using the configured horizontal FOV.
# Override these with your actual calibration for better accuracy.
CAMERA_FOV_H_DEG = 120.0  # Horizontal field of view in degrees
_fx = FRAME_WIDTH / (2.0 * np.tan(np.radians(CAMERA_FOV_H_DEG / 2.0)))
_fy = _fx  # Assume square pixels
_DEFAULT_CAMERA_MATRIX = np.array([
    [_fx,   0, FRAME_WIDTH / 2.0],
    [  0, _fy, FRAME_HEIGHT / 2.0],
    [  0,   0,                1.0],
], dtype=np.float64)
_DEFAULT_DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)  # Assume no distortion
_DEFAULT_CAMERA_SIZE = (FRAME_WIDTH, FRAME_HEIGHT)
CALIBRATION_PATH = Path(__file__).with_name("camera_calibration.json")


def _load_calibration():
    """Load a saved OpenCV calibration if one is available."""
    if not CALIBRATION_PATH.exists():
        return _DEFAULT_CAMERA_MATRIX, _DEFAULT_DIST_COEFFS, _DEFAULT_CAMERA_SIZE

    with CALIBRATION_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64)
    image_width = int(data.get("image_width", FRAME_WIDTH))
    image_height = int(data.get("image_height", FRAME_HEIGHT))
    return camera_matrix, dist_coeffs.reshape((-1, 1)), (image_width, image_height)


CAMERA_MATRIX, DIST_COEFFS, CAMERA_MATRIX_SIZE = _load_calibration()


def get_camera_model(frame_width, frame_height):
    """Scale the loaded/default intrinsics to a specific frame size."""
    base_width, base_height = CAMERA_MATRIX_SIZE
    sx = float(frame_width) / float(base_width)
    sy = float(frame_height) / float(base_height)

    camera_matrix = CAMERA_MATRIX.copy()
    camera_matrix[0, 0] *= sx
    camera_matrix[1, 1] *= sy
    camera_matrix[0, 2] *= sx
    camera_matrix[1, 2] *= sy
    return camera_matrix, DIST_COEFFS.copy()


def get_camera_model_for_fov(frame_width, frame_height, horizontal_fov_deg):
    """Build a simple camera model for a specific frame size and FOV."""
    fx = float(frame_width) / (2.0 * np.tan(np.radians(float(horizontal_fov_deg) / 2.0)))
    fy = fx
    camera_matrix = np.array([
        [fx, 0.0, float(frame_width) / 2.0],
        [0.0, fy, float(frame_height) / 2.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)
    return camera_matrix, dist_coeffs

# ── ArUco ─────────────────────────────────────────────────────────────
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# Workspace corner marker IDs (ordered: TL, TR, BR, BL from camera view)
WORKSPACE_IDS = [0, 1, 2, 3]

# Arm marker ID (base)
ARM_ID = 4

# Gripper marker ID (on end-effector, for calibration)
GRIPPER_ID = 5

# Set to True if the arm marker is physically mounted rotated 180°
# (i.e., the 3D axes show it pointing away from the workspace)
ARM_MARKER_FLIP = True

# Physical marker side length in meters (measure your printed markers)
MARKER_SIZE_M = 0.045  # 45mm — measured marker side length

# ── Colors (BGR) ──────────────────────────────────────────────────────
COLOR_WORKSPACE = (255, 180, 0)   # Cyan-ish
COLOR_ARM       = (0, 0, 255)     # Red
COLOR_YOLO      = (0, 255, 0)     # Green

# ── YOLO ──────────────────────────────────────────────────────────────
YOLO_ENABLED = True
YOLO_MODEL = "yolov8n.pt"
YOLO_CLASSES = [41]  # All objects enabled
YOLO_MIN_CONFIDENCE = 0.25

# ── Units ─────────────────────────────────────────────────────────────
M_TO_IN = 39.3701  # meters → inches
