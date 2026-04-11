"""
Kinematic parameters for the VEX V5 5-DOF robotic arm.

All lengths are in METERS. Replace placeholder values with your
actual measured dimensions.
"""

# ── Offset from ArUco workspace origin to the turntable rotation axis ──
# [x, y, z] in IK frame (IK_x = -WS_y, IK_y = -WS_x, IK_z = -WS_z)
# Arm true base is at workspace X=0. Marker is offset 2.1in to the side. -> IK (-0.376, 0.0)
ARUCO_TO_BASE_XYZ = [-0.376, 0.0, 0.042]

# ── Eccentric offset from turntable Z-axis to shoulder joint ───────────
# In IK frame: X=forward, Y=left, Z=up
# Physically: J1 pivot is forward (+X) and above (+Z) the turntable center
BASE_TO_SHOULDER_XYZ = [0.052, 0.0, 0.04]

# ── Link lengths (meters) ─────────────────────────────────────────────
BICEP_LEN = 0.123          # TODO: measure — shoulder to elbow
FOREARM_LEN = 0.114        # TODO: measure — elbow to wrist-yaw joint
WRIST_YAW_LEN = 0.097       # TODO: measure — wrist-yaw to wrist-pitch (0 if co-located)
WRIST_TO_TIP_LEN = 0.20   # TODO: measure — wrist-pitch (J4 pivot) to gripper fingertips

# ── Joint limits (degrees) ────────────────────────────────────────────
# [min_deg, max_deg] for each active joint
#
# VEX motor conventions (for mapping IK output → VEX commands):
#   J1 shoulder: +100° = fully UP (vertical), -52° = fully DOWN
JOINT_LIMITS_DEG = {
    "turntable":   [-90, 90],     # J0 — base yaw (Z)
    "shoulder":    [-52, 100],    # J1 — shoulder pitch (Y); +100°=up, -52°=down
    "elbow":       [-90, 90],     # J2 — elbow pitch (Y)
    "wrist_yaw":   [-5, 5],       # J3 — wrist yaw (Z)
    "wrist_pitch": [-90, 90],     # J4 — wrist pitch (Y)
}

# ── Joint sign map ────────────────────────────────────────────────────
# Multiply each IK output by this sign before sending to VEX.
# Set to -1 if ikpy's positive direction is opposite to VEX's.
# [J0, J1, J2, J3, J4]
JOINT_SIGN_MAP = [1, 1, 1, 1, 1]
