"""
Inverse Kinematics solver for the VEX V5 5-DOF arm using ikpy.

The end-effector is constrained to a strictly vertical (downward)
orientation so the arm always approaches from above.

Chain (7 links, 5 active joints):
  [0] base         — fixed origin offset (ARUCO_TO_BASE_XYZ)
  [1] turntable    — J0: yaw   (Z-axis)
  [2] shoulder     — J1: pitch (Y-axis)
  [3] elbow        — J2: pitch (Y-axis), offset = BICEP_LEN
  [4] wrist_yaw    — J3: yaw   (Z-axis), offset = FOREARM_LEN
  [5] wrist_pitch  — J4: pitch (Y-axis), offset = WRIST_YAW_LEN
  [6] tip          — fixed end-effector,  offset = WRIST_TO_TIP_LEN
"""

import math
import numpy as np

from ikpy.chain import Chain
from ikpy.link import URDFLink

from vex.setup import (
    ARUCO_TO_BASE_XYZ,
    BASE_TO_SHOULDER_XYZ,
    BICEP_LEN,
    FOREARM_LEN,
    JOINT_LIMITS_DEG,
    WRIST_TO_TIP_LEN,
    WRIST_YAW_LEN,
)


def _deg2rad(deg):
    return math.radians(deg)


def _build_chain() -> Chain:
    """Build the 5-joint arm chain with joint limits."""

    # Convert degree limits to radians for ikpy
    lim = {}
    for name, (lo, hi) in JOINT_LIMITS_DEG.items():
        lim[name] = (_deg2rad(lo), _deg2rad(hi))

    return Chain(
        name="vex_arm",
        links=[
            # ── Link 0: fixed base offset (ArUco → turntable axis) ───
            URDFLink(
                name="base",
                origin_translation=ARUCO_TO_BASE_XYZ,
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
                bounds=(0, 0),
            ),
            # ── Link 1 (J0): Turntable — yaw around Z ───────────────
            URDFLink(
                name="turntable",
                origin_translation=BASE_TO_SHOULDER_XYZ,
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=lim["turntable"],
            ),
            # ── Link 2 (J1): Shoulder — pitch around Y ──────────────
            URDFLink(
                name="shoulder",
                origin_translation=[0, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                bounds=lim["shoulder"],
            ),
            # ── Link 3 (J2): Elbow — pitch around Y ─────────────────
            URDFLink(
                name="elbow",
                origin_translation=[BICEP_LEN, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                bounds=lim["elbow"],
            ),
            # ── Link 4 (J3): Wrist yaw — yaw around Z ───────────────
            URDFLink(
                name="wrist_yaw",
                origin_translation=[FOREARM_LEN, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 1],
                bounds=lim["wrist_yaw"],
            ),
            # ── Link 5 (J4): Wrist pitch — pitch around Y ───────────
            URDFLink(
                name="wrist_pitch",
                origin_translation=[WRIST_YAW_LEN, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 1, 0],
                bounds=lim["wrist_pitch"],
            ),
            # ── Link 6: end-effector tip (fixed) ─────────────────────
            URDFLink(
                name="tip",
                origin_translation=[WRIST_TO_TIP_LEN, 0, 0],
                origin_orientation=[0, 0, 0],
                rotation=[0, 0, 0],
                bounds=(0, 0),
            ),
        ],
        active_links_mask=[
            False,  # 0: base         (fixed)
            True,   # 1: turntable    (J0 - yaw)
            True,   # 2: shoulder     (J1 - pitch)
            True,   # 3: elbow        (J2 - pitch)
            True,   # 4: wrist_yaw    (J3 - yaw)
            True,   # 5: wrist_pitch  (J4 - pitch)
            False,  # 6: tip          (fixed)
        ],
    )


# ── Module-level chain instance (built once) ─────────────────────────
_chain = _build_chain()

# Target orientation: end-effector pointing straight down (-Z)
_TARGET_ORIENTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
])

# Active joint indices in the chain
_ACTIVE_INDICES = [i for i, a in enumerate(_chain.active_links_mask) if a]


def _default_initial_seed() -> np.ndarray:
    """Return the resting-position seed for every active joint."""
    # Default VEX arm calibration position (degrees)
    _DEFAULT_ANGLES_DEG = [0, -34, 90, 0, 90]
    initial = np.zeros(len(_chain.links))
    for idx, ai in enumerate(_ACTIVE_INDICES):
        lo, hi = _chain.links[ai].bounds
        seed_rad = _deg2rad(_DEFAULT_ANGLES_DEG[idx])
        initial[ai] = min(max(seed_rad, lo), hi)
    return initial


def _sanitize_seed_angle_deg(angle_deg: float, lo_rad: float, hi_rad: float) -> float:
    """Normalize/clamp a seed angle so SciPy always receives an in-bounds x0."""
    lo_deg = math.degrees(lo_rad)
    hi_deg = math.degrees(hi_rad)
    span_deg = hi_deg - lo_deg
    seed_deg = float(angle_deg)

    # Wrap full-rotation joints into their configured range instead of pinning.
    if span_deg >= 359.0:
        seed_deg = ((seed_deg - lo_deg) % span_deg) + lo_deg

    seed_deg = min(max(seed_deg, lo_deg), hi_deg)
    return _deg2rad(seed_deg)


def _build_initial_seed(current_angles_deg: list[float] | None = None) -> np.ndarray:
    """Build a valid seed for ikpy from live joint angles or neutral midpoints."""
    initial = _default_initial_seed()
    if current_angles_deg is None or len(current_angles_deg) < len(_ACTIVE_INDICES):
        return initial

    for idx, ai in enumerate(_ACTIVE_INDICES):
        lo, hi = _chain.links[ai].bounds
        initial[ai] = _sanitize_seed_angle_deg(current_angles_deg[idx], lo, hi)
    return initial


# Identify which active joints are yaw (Z-axis rotation)
_YAW_JOINT_FLAGS = [
    list(_chain.links[ai].rotation) == [0, 0, 1]
    for ai in _ACTIVE_INDICES
]


def _normalize_yaw_angles(angles_deg: list[float]) -> list[float]:
    """Wrap yaw joint angles to [-180, 180] so motors take the shortest path.

    For example, 326° becomes -34° — same position but only 34° of rotation
    instead of 326°.
    """
    result = list(angles_deg)
    for i, is_yaw in enumerate(_YAW_JOINT_FLAGS):
        if is_yaw:
            a = result[i] % 360.0        # force into [0, 360)
            if a > 180.0:
                a -= 360.0                # map to (-180, 180]
            result[i] = a
    return result


def get_ik_angles(
    target_x: float,
    target_y: float,
    target_z: float,
    current_angles_deg: list[float] | None = None,
) -> list[float]:
    """Compute joint angles (degrees) for the 5 active joints.

    Parameters
    ----------
    target_x, target_y, target_z : float
        Desired end-effector position in the ArUco workspace frame (meters).
    current_angles_deg : list[float] | None
        Current joint angles [J0, J1, J2, J3, J4] in degrees (optional).
        Used as the initial seed for the optimizer.

    Returns
    -------
    list[float]
        [J0_deg, J1_deg, J2_deg, J3_deg, J4_deg] — turntable, shoulder,
        elbow, wrist_yaw, wrist_pitch.
    """
    target_pose = np.eye(4)
    target_pose[:3, :3] = _TARGET_ORIENTATION
    target_pose[:3, 3] = [target_x, target_y, target_z]

    initial = _build_initial_seed(current_angles_deg)

    # Pre-compute turntable angle geometrically so the optimizer doesn't
    # get stuck at the midpoint seed (180°).  The turntable (J0, link 1)
    # rotates around Z; its 0° direction extends along +X from the base.
    base_offset = np.array(ARUCO_TO_BASE_XYZ)
    dx = target_x - base_offset[0]
    dy = target_y - base_offset[1]
    turntable_rad = math.atan2(dy, dx)
    # Ensure it's within bounds
    lo, hi = _chain.links[_ACTIVE_INDICES[0]].bounds
    if turntable_rad < lo:
        turntable_rad += 2 * math.pi
    elif turntable_rad > hi:
        turntable_rad -= 2 * math.pi
    initial[_ACTIVE_INDICES[0]] = turntable_rad

    kwargs = {"initial_position": initial}

    target_position = [target_x, target_y, target_z]

    try:
        full_angles = _chain.inverse_kinematics(
            target_position=target_position,
            **kwargs,
        )
    except ValueError as exc:
        if "`x0` is infeasible" not in str(exc):
            raise

        # Fall back to a neutral seed if a live STATUS sample is still rejected.
        full_angles = _chain.inverse_kinematics(
            target_position=target_position,
            initial_position=_default_initial_seed(),
        )

    # Extract and convert active joint angles
    joint_angles_rad = [full_angles[i] for i in _ACTIVE_INDICES]
    joint_angles_deg = [math.degrees(a) for a in joint_angles_rad]

    # Normalize yaw joints to shortest-path angles.
    joint_angles_deg = _normalize_yaw_angles(joint_angles_deg)

    # Apply per-joint sign map (ikpy → VEX motor convention)
    from vex.setup import JOINT_SIGN_MAP
    joint_angles_deg = [s * a for s, a in zip(JOINT_SIGN_MAP, joint_angles_deg)]

    print(f"[IK] Raw IK (deg): {[round(a,1) for a in joint_angles_deg]}", flush=True)

    return joint_angles_deg


def get_ik_angles_raw(
    target_x: float,
    target_y: float,
    target_z: float,
) -> list[float]:
    """Same as get_ik_angles but returns the full radians array."""
    target_pose = np.eye(4)
    target_pose[:3, :3] = _TARGET_ORIENTATION
    target_pose[:3, 3] = [target_x, target_y, target_z]

    return _chain.inverse_kinematics_frame(
        target=target_pose,
        orientation_mode="all",
        initial_position=_default_initial_seed(),
    )


def forward_kinematics(angles_rad: list[float]) -> np.ndarray:
    """Compute the end-effector 4×4 pose from a full angle array."""
    return _chain.forward_kinematics(angles_rad)


def compute_fk_position(angles_deg: list[float]) -> list[float]:
    """Compute expected end-effector [x, y, z] (meters) from 5 joint angles (degrees).

    Returns position in the workspace coordinate frame (same as IK targets).
    """
    full_angles = np.zeros(len(_chain.links))
    for idx, ai in enumerate(_ACTIVE_INDICES):
        full_angles[ai] = _deg2rad(angles_deg[idx])
    pose = _chain.forward_kinematics(full_angles)
    pos = pose[:3, 3]
    return [float(pos[0]), float(pos[1]), float(pos[2])]


# ── Quick self-test ──────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chain links:")
    for i, link in enumerate(_chain.links):
        active = "JOINT" if _chain.active_links_mask[i] else "fixed"
        lo, hi = link.bounds
        axis = "Z" if list(link.rotation) == [0, 0, 1] else ("Y" if list(link.rotation) == [0, 1, 0] else "-")
        print(f"  [{i}] {link.name:20s}  ({active})  axis={axis}  bounds=[{math.degrees(lo):.0f}°, {math.degrees(hi):.0f}°]")

    test_pos = [0.20, 0.0, 0.10]
    print(f"\nTarget position: {test_pos}")

    angles_deg = get_ik_angles(*test_pos)
    print(f"IK solution (deg): {[f'{a:.1f}' for a in angles_deg]}")

    full_rad = get_ik_angles_raw(*test_pos)
    fk_pose = forward_kinematics(full_rad)
    reached = fk_pose[:3, 3]
    print(f"FK verification:   [{reached[0]:.4f}, {reached[1]:.4f}, {reached[2]:.4f}]")
    err = np.linalg.norm(reached - test_pos)
    print(f"Position error:    {err*1000:.2f} mm")
