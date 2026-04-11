"""
ArUco marker detection, workspace tracking, and 3D pose estimation.
"""

import cv2
import numpy as np
import threading
import vision.config as config
from vision.config import (
    ARUCO_DICT, WORKSPACE_IDS, ARM_ID,
    COLOR_WORKSPACE, COLOR_ARM,
    MARKER_SIZE_M, M_TO_IN,
)
ARM_MARKER_FLIP = getattr(config, "ARM_MARKER_FLIP", False)

# Create detector once
_aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_aruco_params = cv2.aruco.DetectorParameters()
detector      = cv2.aruco.ArucoDetector(_aruco_dict, _aruco_params)

# 3D model points for a single marker (in marker frame, Z=0)
_half = MARKER_SIZE_M / 2.0
_MARKER_OBJ_PTS = np.array([
    [-_half,  _half, 0],
    [ _half,  _half, 0],
    [ _half, -_half, 0],
    [-_half, -_half, 0],
], dtype=np.float32)

# Cached state
_perspective_matrix = None   # Pixel → normalised workspace (2D)
_workspace_plane_z = None    # Average Z of workspace markers in camera frame
_workspace_size_m = None     # (width, height) using corner marker centers
_workspace_rvec = None       # Rotation of workspace plane
_workspace_tvec = None       # Translation of workspace plane center
_workspace_origin = None     # Workspace origin in camera coordinates
_workspace_x_axis = None     # Unit +X direction in camera coordinates
_workspace_y_axis = None     # Unit +Y direction in camera coordinates
_workspace_normal = None     # Unit +Z direction (up from workspace plane)
_camera_matrix, _dist_coeffs = config.get_camera_model(*config.CAMERA_MATRIX_SIZE)
_camera_matrix_inv = np.linalg.inv(_camera_matrix)
_state_lock = threading.Lock()
_last_marker_centers = {}
_last_marker_poses = {}
_last_workspace_corners = {}
_last_arm_center = None
_last_arm_pose_3d = None
_last_gripper_ws = None  # Gripper marker workspace position (x, y, z) in meters
_WORKSPACE_CORNERS = {
    WORKSPACE_IDS[0]: (0.0, 0.0),
    WORKSPACE_IDS[1]: (0.0, 1.0),
    WORKSPACE_IDS[2]: (1.0, 1.0),
    WORKSPACE_IDS[3]: (1.0, 0.0),
}


def _to_native(val):
    """Convert numpy scalars to native Python types for JSON serialization."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    return val


def _estimate_pose(corners_2d):
    """Estimate 3D pose of a single marker using solvePnP.

    Returns (rvec, tvec) where tvec = [x, y, z] in meters from camera.
    """
    with _state_lock:
        camera_matrix = _camera_matrix.copy()
        dist_coeffs = _dist_coeffs.copy()

    success, rvec, tvec = cv2.solvePnP(
        _MARKER_OBJ_PTS,
        corners_2d.astype(np.float64),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        return None, None
    return rvec, tvec


def _marker_distance(marker_poses, marker_a, marker_b):
    """Return the 3D distance between two detected marker centers."""
    pose_a = marker_poses.get(marker_a)
    pose_b = marker_poses.get(marker_b)
    if pose_a is None or pose_b is None:
        return None

    tvec_a = np.array(pose_a["tvec"], dtype=np.float64)
    tvec_b = np.array(pose_b["tvec"], dtype=np.float64)
    return float(np.linalg.norm(tvec_a - tvec_b))


def _compute_workspace_size_m(marker_poses):
    """Estimate workspace width/height from the detected corner marker centers."""
    top_left, top_right, bottom_right, bottom_left = WORKSPACE_IDS

    widths = [
        d for d in (
            _marker_distance(marker_poses, top_left, top_right),
            _marker_distance(marker_poses, bottom_left, bottom_right),
        )
        if d is not None
    ]
    heights = [
        d for d in (
            _marker_distance(marker_poses, top_left, bottom_left),
            _marker_distance(marker_poses, top_right, bottom_right),
        )
        if d is not None
    ]

    if not widths or not heights:
        return None

    return (
        sum(widths) / len(widths),
        sum(heights) / len(heights),
    )


def _get_workspace_points(marker_poses):
    """Return workspace corner centers in camera coordinates."""
    points = []
    for marker_id in WORKSPACE_IDS:
        pose = marker_poses.get(marker_id)
        if pose is None:
            return None
        points.append(np.array(pose["tvec"], dtype=np.float64))
    return points


def _compute_workspace_frame(marker_poses):
    """Build a workspace frame from the 4 corner markers."""
    points = _get_workspace_points(marker_poses)
    if points is None:
        return None

    top_left, top_right, bottom_right, bottom_left = points

    # Swap axes: X = TL→BL (long side), Y = TL→TR (short side)
    x_vec = ((bottom_left - top_left) + (bottom_right - top_right)) / 2.0
    y_vec = ((top_right - top_left) + (bottom_right - bottom_left)) / 2.0
    width = float(np.linalg.norm(x_vec))
    height = float(np.linalg.norm(y_vec))

    if width < 1e-6 or height < 1e-6:
        return None

    x_axis = x_vec / width
    y_axis = y_vec / height
    normal = np.cross(x_axis, y_axis)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm < 1e-6:
        return None
    normal = normal / normal_norm

    # Make +Z point upward from the workspace plane toward the camera.
    if float(np.dot(normal, top_left)) > 0:
        normal = -normal

    return {
        "origin": top_left,
        "x_axis": x_axis,
        "y_axis": y_axis,
        "normal": normal,
        "width": width,
        "height": height,
    }


def _pixel_to_ray(px, py):
    """Return a camera-space ray for a pixel."""
    with _state_lock:
        camera_matrix_inv = _camera_matrix_inv.copy()
    pixel = np.array([float(px), float(py), 1.0], dtype=np.float64)
    ray = camera_matrix_inv @ pixel
    return ray / np.linalg.norm(ray)


def configure_camera_model(frame_width, frame_height, horizontal_fov_deg=None):
    """Update the active intrinsics for the current frame size."""
    global _camera_matrix, _dist_coeffs, _camera_matrix_inv
    if horizontal_fov_deg is None:
        camera_matrix, dist_coeffs = config.get_camera_model(frame_width, frame_height)
    else:
        camera_matrix, dist_coeffs = config.get_camera_model_for_fov(
            frame_width,
            frame_height,
            horizontal_fov_deg,
        )
    with _state_lock:
        _camera_matrix = camera_matrix
        _dist_coeffs = dist_coeffs
        _camera_matrix_inv = np.linalg.inv(_camera_matrix)


def camera_tilt_degrees():
    """Return the camera tilt away from the workspace normal in degrees."""
    with _state_lock:
        workspace_normal = None if _workspace_normal is None else _workspace_normal.copy()

    if workspace_normal is None:
        return None

    normal = workspace_normal / np.linalg.norm(workspace_normal)
    cos_angle = float(np.clip(abs(np.dot(normal, np.array([0.0, 0.0, 1.0]))), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _intersect_ray_with_workspace(px, py):
    """Intersect a camera ray with the workspace plane."""
    with _state_lock:
        workspace_origin = None if _workspace_origin is None else _workspace_origin.copy()
        workspace_normal = None if _workspace_normal is None else _workspace_normal.copy()

    if workspace_origin is None or workspace_normal is None:
        return None

    ray = _pixel_to_ray(px, py)
    denom = float(np.dot(workspace_normal, ray))
    if abs(denom) < 1e-6:
        return None

    distance = float(np.dot(workspace_normal, workspace_origin) / denom)
    if distance <= 0:
        return None

    return ray * distance


def pixel_to_workspace(px, py):
    """Convert a pixel coordinate to normalised workspace coordinates.

    Returns (wx, wy) or None if workspace is not calibrated.
    """
    with _state_lock:
        workspace_origin = None if _workspace_origin is None else _workspace_origin.copy()
        workspace_x_axis = None if _workspace_x_axis is None else _workspace_x_axis.copy()
        workspace_y_axis = None if _workspace_y_axis is None else _workspace_y_axis.copy()
        workspace_size_m = _workspace_size_m

    if workspace_origin is None or workspace_x_axis is None or workspace_y_axis is None or workspace_size_m is None:
        return None

    point = _intersect_ray_with_workspace(px, py)
    if point is None:
        return None

    relative = point - workspace_origin
    wx = float(np.dot(relative, workspace_x_axis) / workspace_size_m[0])
    wy = float(np.dot(relative, workspace_y_axis) / workspace_size_m[1])
    return (wx, wy)


def workspace_to_inches(position):
    """Convert normalised workspace coordinates to inches."""
    with _state_lock:
        workspace_size_m = _workspace_size_m

    if position is None or workspace_size_m is None:
        return None

    coords = [
        float(position[0]) * workspace_size_m[0] * M_TO_IN,
        float(position[1]) * workspace_size_m[1] * M_TO_IN,
    ]

    if len(position) >= 3:
        coords.append(float(position[2]) * M_TO_IN)

    return tuple(coords)


def get_workspace_plane_z():
    """Return the ArUco-measured distance from the camera to the workspace plane (meters)."""
    with _state_lock:
        return _workspace_plane_z


def get_distance_to_point(px, py):
    """Compute the 3D distance from the camera to a point on the workspace plane.

    Uses the webcam's ArUco-calibrated workspace to precisely determine
    how far the camera is from the given pixel's intersection with the workspace.

    Returns distance in meters, or None if workspace is not calibrated.
    """
    point = _intersect_ray_with_workspace(px, py)
    if point is None:
        return None
    return float(np.linalg.norm(point))


def pose_to_inches(position):
    """Convert a camera-space position from meters to inches."""
    if position is None:
        return None
    return tuple(float(v) * M_TO_IN for v in position)


def camera_point_to_workspace(point):
    """Convert a camera-space 3D point into workspace coordinates."""
    with _state_lock:
        workspace_origin = None if _workspace_origin is None else _workspace_origin.copy()
        workspace_x_axis = None if _workspace_x_axis is None else _workspace_x_axis.copy()
        workspace_y_axis = None if _workspace_y_axis is None else _workspace_y_axis.copy()
        workspace_normal = None if _workspace_normal is None else _workspace_normal.copy()
        workspace_size_m = _workspace_size_m

    if point is None or workspace_origin is None or workspace_x_axis is None or workspace_y_axis is None or workspace_normal is None or workspace_size_m is None:
        return None

    point = np.array(point, dtype=np.float64)
    relative = point - workspace_origin
    wx = float(np.dot(relative, workspace_x_axis) / workspace_size_m[0])
    wy = float(np.dot(relative, workspace_y_axis) / workspace_size_m[1])
    wz = float(np.dot(relative, workspace_normal))
    return (wx, wy, wz)


def marker_to_workspace(marker_id, center, marker_poses):
    """Convert a detected marker center into workspace-relative coordinates."""
    if center is None:
        return None

    if marker_id in _WORKSPACE_CORNERS:
        wx, wy = _WORKSPACE_CORNERS[marker_id]
        return (wx, wy, 0.0)

    pose = marker_poses.get(marker_id)
    with _state_lock:
        workspace_origin = None if _workspace_origin is None else _workspace_origin.copy()
        workspace_x_axis = None if _workspace_x_axis is None else _workspace_x_axis.copy()
        workspace_y_axis = None if _workspace_y_axis is None else _workspace_y_axis.copy()
        workspace_normal = None if _workspace_normal is None else _workspace_normal.copy()
        workspace_size_m = _workspace_size_m

    if pose is None or workspace_origin is None or workspace_x_axis is None or workspace_y_axis is None or workspace_normal is None or workspace_size_m is None:
        return None

    point = np.array(pose["tvec"], dtype=np.float64)
    relative = point - workspace_origin
    wx = float(np.dot(relative, workspace_x_axis) / workspace_size_m[0])
    wy = float(np.dot(relative, workspace_y_axis) / workspace_size_m[1])
    wz = max(0.0, float(np.dot(relative, workspace_normal)))

    return (wx, wy, round(wz, 4))


def estimate_object_z(px, bbox_top_y, bbox_bottom_y):
    """Estimate the real-world height of an object above the workspace plane.

    Uses similar triangles: real_height = (pixel_height × Z_distance) / focal_length
    The bottom of the bbox is assumed to sit on the workspace plane.

    Args:
        px: x-pixel used for the object's vertical ray
        bbox_top_y: top y-pixel of the bounding box
        bbox_bottom_y: bottom y-pixel of the bounding box (on workspace plane)

    Returns: estimated height in meters, or None if workspace is not calibrated.
    """
    with _state_lock:
        workspace_normal = None if _workspace_normal is None else _workspace_normal.copy()

    if workspace_normal is None:
        return None

    base_point = _intersect_ray_with_workspace(px, bbox_bottom_y)
    top_ray = _pixel_to_ray(px, bbox_top_y)
    if base_point is None:
        return None

    # Solve: s * top_ray = base_point + h * workspace_normal
    system = np.column_stack((top_ray, -workspace_normal))
    solution, _, _, _ = np.linalg.lstsq(system, base_point, rcond=None)
    height = float(solution[1])
    return max(0.0, height)


def _apply_marker_memory(result, visible_ids):
    """Backfill missing marker data from the last known observations."""
    global _last_arm_center, _last_arm_pose_3d

    with _state_lock:
        cached_centers = dict(_last_marker_centers)
        cached_poses = {
            marker_id: {
                "rvec": list(pose["rvec"]),
                "tvec": list(pose["tvec"]),
            }
            for marker_id, pose in _last_marker_poses.items()
        }
        cached_workspace_corners = dict(_last_workspace_corners)
        cached_arm_center = None if _last_arm_center is None else tuple(_last_arm_center)
        cached_arm_pose_3d = None if _last_arm_pose_3d is None else list(_last_arm_pose_3d)

    visible_set = set(visible_ids)
    memory_ids = set()

    for marker_id in WORKSPACE_IDS:
        if marker_id not in result["workspace_corners"] and marker_id in cached_workspace_corners:
            result["workspace_corners"][marker_id] = cached_workspace_corners[marker_id]
            if marker_id in cached_centers:
                result["marker_centers"][marker_id] = cached_centers[marker_id]
            memory_ids.add(marker_id)
        if marker_id not in result["marker_poses"] and marker_id in cached_poses:
            result["marker_poses"][marker_id] = cached_poses[marker_id]
            if marker_id not in visible_set:
                memory_ids.add(marker_id)

    if result["arm_center"] is None and cached_arm_center is not None:
        result["arm_center"] = cached_arm_center
        result["marker_centers"][ARM_ID] = cached_arm_center
        memory_ids.add(ARM_ID)
    if result["arm_pose_3d"] is None and cached_arm_pose_3d is not None:
        result["arm_pose_3d"] = list(cached_arm_pose_3d)
        if ARM_ID not in visible_set:
            memory_ids.add(ARM_ID)
    if ARM_ID not in result["marker_poses"] and ARM_ID in cached_poses:
        result["marker_poses"][ARM_ID] = cached_poses[ARM_ID]
        if ARM_ID not in visible_set:
            memory_ids.add(ARM_ID)

    result["memory_ids"] = sorted(int(marker_id) for marker_id in memory_ids if marker_id not in visible_set)
    result["workspace_from_memory"] = any(marker_id in memory_ids for marker_id in WORKSPACE_IDS)
    result["arm_from_memory"] = ARM_ID in memory_ids
    return result


def _update_marker_memory(result, visible_ids):
    """Persist the latest visible marker observations for future fallback."""
    global _last_arm_center, _last_arm_pose_3d

    visible_set = set(visible_ids)
    with _state_lock:
        for marker_id in visible_set:
            if marker_id in result["marker_centers"]:
                _last_marker_centers[marker_id] = tuple(int(v) for v in result["marker_centers"][marker_id])
            if marker_id in result["marker_poses"]:
                pose = result["marker_poses"][marker_id]
                _last_marker_poses[marker_id] = {
                    "rvec": [float(v) for v in pose["rvec"]],
                    "tvec": [float(v) for v in pose["tvec"]],
                }

        for marker_id in WORKSPACE_IDS:
            if marker_id in result["workspace_corners"]:
                _last_workspace_corners[marker_id] = tuple(int(v) for v in result["workspace_corners"][marker_id])

        if ARM_ID in visible_set and result["arm_center"] is not None:
            _last_arm_center = tuple(int(v) for v in result["arm_center"])
            if result["arm_pose_3d"] is not None:
                _last_arm_pose_3d = [float(v) for v in result["arm_pose_3d"]]


def detect(frame):
    """Detect ArUco markers and compute workspace / arm positions.

    Returns a dict with all detection results including 3D poses.
    """
    global _perspective_matrix, _workspace_plane_z, _workspace_size_m, _workspace_rvec, _workspace_tvec
    global _workspace_origin, _workspace_x_axis, _workspace_y_axis, _workspace_normal

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    result = {
        "corners": corners,
        "ids": ids,
        "memory_ids": [],
        "workspace_from_memory": False,
        "arm_from_memory": False,
        "marker_centers": {},
        "workspace_corners": {},
        "workspace_poly": None,
        "marker_workspace": {},
        "arm_center": None,
        "arm_in_workspace": None,
        "arm_pose_3d": None,      # (x, y, z) in meters from camera
        "marker_poses": {},       # {id: {rvec, tvec}} for all detected markers
    }
    flat_ids = [] if ids is None or len(ids) == 0 else [int(marker_id) for marker_id in ids.flatten()]

    # Estimate 3D pose for every detected marker
    for i, marker_id in enumerate(flat_ids):
        c = corners[i][0]
        center = tuple(int(v) for v in c.mean(axis=0))
        result["marker_centers"][marker_id] = center

        # Flip arm marker orientation 180° if configured
        pose_corners = c
        if marker_id == ARM_ID and ARM_MARKER_FLIP:
            pose_corners = np.roll(c, 2, axis=0)

        rvec, tvec = _estimate_pose(pose_corners)

        if rvec is not None:
            result["marker_poses"][marker_id] = {
                "rvec": rvec.flatten().tolist(),
                "tvec": tvec.flatten().tolist(),  # [x, y, z] meters from camera
            }

        if marker_id in WORKSPACE_IDS:
            result["workspace_corners"][marker_id] = center
        elif marker_id == ARM_ID:
            result["arm_center"] = center
            if tvec is not None:
                result["arm_pose_3d"] = [float(v) for v in tvec.flatten()]

    result = _apply_marker_memory(result, flat_ids)

    # Build workspace polygon if all 4 corners detected
    if len(result["workspace_corners"]) == 4:
        poly = np.array(
            [result["workspace_corners"][wid] for wid in WORKSPACE_IDS],
            dtype=np.float32,
        )
        result["workspace_poly"] = poly

        # Cache 2D perspective transform
        dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        _perspective_matrix = cv2.getPerspectiveTransform(poly, dst)

        # Compute average Z of workspace plane from marker poses
        ws_z_values = []
        for wid in WORKSPACE_IDS:
            if wid in result["marker_poses"]:
                ws_z_values.append(result["marker_poses"][wid]["tvec"][2])
        frame_data = _compute_workspace_frame(result["marker_poses"])
        with _state_lock:
            if ws_z_values:
                _workspace_plane_z = sum(ws_z_values) / len(ws_z_values)
            if frame_data is not None:
                _workspace_origin = frame_data["origin"]
                _workspace_x_axis = frame_data["x_axis"]
                _workspace_y_axis = frame_data["y_axis"]
                _workspace_normal = frame_data["normal"]
                _workspace_size_m = (frame_data["width"], frame_data["height"])
            else:
                _workspace_origin = None
                _workspace_x_axis = None
                _workspace_y_axis = None
                _workspace_normal = None
                _workspace_size_m = None

        for marker_id, center in result["marker_centers"].items():
            marker_ws = marker_to_workspace(marker_id, center, result["marker_poses"])
            if marker_ws is not None:
                result["marker_workspace"][marker_id] = marker_ws

        # Compute arm's workspace position
        result["arm_in_workspace"] = result["marker_workspace"].get(ARM_ID)

        # Compute gripper's workspace position
        from vision.config import GRIPPER_ID
        gripper_ws = result["marker_workspace"].get(GRIPPER_ID)
        result["gripper_in_workspace"] = gripper_ws
        if gripper_ws is not None:
            global _last_gripper_ws
            _last_gripper_ws = list(gripper_ws)
    else:
        with _state_lock:
            if _last_workspace_corners:
                result["workspace_corners"] = dict(_last_workspace_corners)

    _update_marker_memory(result, flat_ids)
    return result


def draw(frame, aruco_data):
    """Draw ArUco overlays onto the frame."""
    corners = aruco_data["corners"]
    ids = aruco_data["ids"]

    if ids is not None and len(ids) > 0:
        flat_ids = ids.flatten()

        for i, marker_id in enumerate(flat_ids):
            marker_id = int(marker_id)
            c = corners[i][0].astype(int)

            if marker_id in WORKSPACE_IDS:
                color = COLOR_WORKSPACE
                label = f"WS-{marker_id} z=0.0in"
            elif marker_id == ARM_ID:
                color = COLOR_ARM
                label = "ARM"
                marker_ws = aruco_data["marker_workspace"].get(marker_id)
                marker_ws_in = workspace_to_inches(marker_ws)
                if marker_ws_in is not None:
                    label += f" ({marker_ws_in[0]:.1f},{marker_ws_in[1]:.1f},z={marker_ws_in[2]:.1f})in"
            else:
                color = (128, 128, 128)
                label = f"ID:{marker_id}"

            cv2.polylines(frame, [c], True, color, 2)
            center = tuple(c.mean(axis=0).astype(int))
            cv2.circle(frame, center, 5, color, -1)
            cv2.putText(frame, label, (c[0][0], c[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            # Draw 3D axis on markers
            if marker_id in aruco_data["marker_poses"]:
                pose = aruco_data["marker_poses"][marker_id]
                rvec = np.array(pose["rvec"], dtype=np.float64)
                tvec = np.array(pose["tvec"], dtype=np.float64)
                with _state_lock:
                    camera_matrix = _camera_matrix.copy()
                    dist_coeffs = _dist_coeffs.copy()
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs,
                                  rvec, tvec, MARKER_SIZE_M * 0.7)

    # Workspace boundary polygon
    poly = aruco_data["workspace_poly"]
    if poly is not None:
        pts = poly.astype(int).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, COLOR_WORKSPACE, 2, cv2.LINE_AA)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], COLOR_WORKSPACE)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Arm info
    arm_ws = aruco_data["arm_in_workspace"]
    if arm_ws is not None:
        arm_ws_in = workspace_to_inches(arm_ws)
        if arm_ws_in is None:
            return

        if len(arm_ws_in) == 3:
            info = f"Arm WS: ({arm_ws_in[0]:.1f}in, {arm_ws_in[1]:.1f}in, z={arm_ws_in[2]:.1f}in)"
        else:
            info = f"Arm WS: ({arm_ws_in[0]:.1f}in, {arm_ws_in[1]:.1f}in)"
        cv2.putText(frame, info, (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ARM, 2)
