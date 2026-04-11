"""Helpers for the forward-facing Arducam depth camera."""

from __future__ import annotations

import math

import cv2

from vision.config import M_TO_IN


def build_csi_gstreamer_pipeline(
    sensor_id: int,
    width: int,
    height: int,
    fps: int,
    sensor_mode: int | None = None,
) -> str:
    """Build a Jetson CSI pipeline string for OpenCV."""
    sensor_mode_arg = f" sensor-mode={sensor_mode}" if sensor_mode is not None else ""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id}{sensor_mode_arg} bufapi-version=true ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        "nvvidconv ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def build_v4l2_bayer_pipeline(device: str, width: int, height: int, fps: int) -> str:
    """Build a GStreamer pipeline for a raw-Bayer V4L2 camera node."""
    return (
        f"v4l2src device={device} ! "
        f"video/x-bayer,format=rggb,width=(int){width},height=(int){height},"
        f"framerate=(fraction){fps}/1 ! "
        "bayer2rgb ! "
        "videoconvert ! "
        "video/x-raw,format=(string)BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )


def estimate_forward_distance_m(
    bbox_bottom_y: int,
    frame_height: int,
    frame_width: int,
    camera_height_m: float,
    camera_fov_h_deg: float,
    pitch_deg: float = 0.0,
) -> float | None:
    """Estimate forward distance from a detected object's base point on the floor plane."""
    if camera_height_m <= 0 or frame_height <= 0 or frame_width <= 0:
        return None

    focal_length_px = frame_width / (2.0 * math.tan(math.radians(camera_fov_h_deg / 2.0)))
    cy = frame_height / 2.0
    pixel_angle = math.atan2(float(bbox_bottom_y) - cy, focal_length_px)
    total_downward_angle = math.radians(pitch_deg) + pixel_angle

    if total_downward_angle <= math.radians(0.5):
        return None

    return float(camera_height_m / math.tan(total_downward_angle))


def estimate_object_height_m(
    bbox_top_y: int,
    bbox_bottom_y: int,
    frame_height: int,
    frame_width: int,
    camera_height_m: float,
    camera_fov_h_deg: float,
    pitch_deg: float = 0.0,
) -> float | None:
    """Estimate the real-world height of a detected object.

    Approach:
    1. Compute the forward distance to the object from the bbox bottom (floor contact).
    2. Use the focal length to convert the bbox pixel height into a real-world height
       at that distance via similar triangles.

    Returns height in meters, or None if geometry can't be computed.
    """
    distance_m = estimate_forward_distance_m(
        bbox_bottom_y, frame_height, frame_width,
        camera_height_m, camera_fov_h_deg, pitch_deg,
    )
    if distance_m is None or distance_m <= 0:
        return None

    focal_length_px = frame_width / (2.0 * math.tan(math.radians(camera_fov_h_deg / 2.0)))
    bbox_height_px = abs(bbox_bottom_y - bbox_top_y)
    if bbox_height_px < 1:
        return 0.0

    # The actual distance from camera to object along the line of sight
    # (not just the forward component) gives better accuracy
    slant_distance = math.sqrt(distance_m ** 2 + camera_height_m ** 2)

    # Similar triangles: real_height / slant_distance = bbox_height_px / focal_length
    real_height = (bbox_height_px * slant_distance) / focal_length_px
    return float(real_height)


def meters_to_inches(distance_m: float | None) -> float | None:
    """Convert meters to inches."""
    if distance_m is None:
        return None
    return float(distance_m) * M_TO_IN


def detect_forward_target_bbox(frame) -> list[int] | None:
    """Detect the dominant forward-facing object as a simple contour bbox."""
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(edges, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_h, frame_w = frame.shape[:2]
    min_area = max(800.0, frame_w * frame_h * 0.002)
    best_bbox = None
    best_score = None
    frame_center_x = frame_w / 2.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        if w >= frame_w * 0.95 and h >= frame_h * 0.95:
            continue

        center_x = x + (w / 2.0)
        bottom_y = y + h
        if bottom_y < frame_h * 0.35:
            continue

        centrality_penalty = abs(center_x - frame_center_x)
        score = area - (centrality_penalty * 4.0) + (h * 20.0)
        if best_score is None or score > best_score:
            best_score = score
            best_bbox = [int(x), int(y), int(x + w), int(y + h)]

    return best_bbox
