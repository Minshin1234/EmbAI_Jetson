#ifndef ARM_CONFIG_HPP
#define ARM_CONFIG_HPP

#include <cstdint>

// ════════════════════════════════════════════════════════════════════════════
//  HARDWARE PORT MAP  —  Update these to match your wiring
// ════════════════════════════════════════════════════════════════════════════

// Joint 0 — Base swivel (single motor, 9:17 external gear)
constexpr std::int8_t BASE_PORT = -15;

// Joint 1 — 1st stage / shoulder (TWO motors as a MotorGroup)
// Calibration / home direction: clockwise
constexpr std::int8_t STAGE1_PORT_A = -12; // Motor A
constexpr std::int8_t STAGE1_PORT_B = 14;  // Motor B (reversed)

// Joint 2 — 2nd stage
constexpr std::int8_t STAGE2_PORT =
    -16; // Calibration / home direction: counterclockwise

// Joint 3 — 3rd stage (wrist L/R)
constexpr std::int8_t STAGE3_PORT =
    -17; // Calibration / home direction: counterclockwise

// Joint 4 — Gripper up/down
constexpr std::int8_t STAGE4_PORT =
    13; // Calibration / home direction: counterclockwise

// Sensors
constexpr std::int8_t ROTATION_PORT =
    18; // External rotation sensor for stage 1

// Gripper servo (SG90 on ADI 3-wire port)
constexpr std::uint8_t GRIPPER_ADI_PORT = 'H';

// Comms
constexpr std::uint8_t SERIAL_PORT = 10; // RS-485 smart-port
constexpr std::int32_t BAUDRATE = 115200;

// ════════════════════════════════════════════════════════════════════════════
//  TIMING CONSTANTS
// ════════════════════════════════════════════════════════════════════════════
constexpr std::uint32_t COMMAND_TIMEOUT_MS = 5000; // watchdog: 5 s
constexpr double PID_DT_SEC = 0.010;               // 10 ms PID period
constexpr std::uint32_t PID_PERIOD_MS = 10;
constexpr std::uint32_t SERIAL_PERIOD_MS = 50; // serial loop ~20 Hz
constexpr std::uint32_t JOINT_STAGGER_MS =
    20; // delay between joints in A/AR/HOME commands

// ── Homing / taring constants ─────────────────────────────────────────
constexpr std::int32_t HOMING_VOLTAGE_MV = 6000; // mV to drive into hard stop
constexpr std::int32_t HOMING_CURRENT_THRESHOLD_MA =
    800; // mA spike = stall detected
constexpr std::uint32_t HOMING_STALL_TIME_MS =
    100; // current must stay above threshold this long
constexpr std::uint32_t HOMING_SETTLE_MS =
    50; // wait after stopping before taring
constexpr std::uint32_t HOMING_POLL_MS = 10; // polling interval during homing
constexpr std::uint32_t HOMING_TIMEOUT_MS =
    1000; // max time per joint — tare wherever it is

// ── Motor limits ──────────────────────────────────────────────────────
constexpr std::int32_t PID_MAX_VOLTAGE_MV = 6000; // max motor speed (millivolts)
constexpr std::int32_t MOTOR_CURRENT_LIMIT_MA =
    3400; // if exceeded, snap target to current pos

// ════════════════════════════════════════════════════════════════════════════
//  ARM GEOMETRY
// ════════════════════════════════════════════════════════════════════════════
constexpr int NUM_JOINTS = 5;
constexpr double BASE_GEAR_RATIO =
    9.0 / 17.0; // motor driver / turntable driven

// Joint 1 — shoulder (rotation sensor)
constexpr double STAGE1_MIN_DEG = -52.0; // hard stop (homing end)
constexpr double STAGE1_MAX_DEG = 100.0;
constexpr double STAGE1_TARE_DEG = -52.0; // value at homing hard stop

// Joints 2–4 — elbow / wrist / gripper (motor encoders)
constexpr double STAGE234_MIN_DEG = -90.0;
constexpr double STAGE234_MAX_DEG = 90.0;
constexpr double STAGE234_TARE_DEG = 90.0; // value at homing hard stop

// ════════════════════════════════════════════════════════════════════════════
//  PID GAINS  —  Starting-point values; tune on the real robot
// ════════════════════════════════════════════════════════════════════════════
//                                kP     kI    kD
constexpr double BASE_KP = 250.0, BASE_KI = 0, BASE_KD = 2.0;
constexpr double STAGE1_KP = 200.0, STAGE1_KI = 10, STAGE1_KD = 5.0;
constexpr double STAGE2_KP = 200.0, STAGE2_KI = 10, STAGE2_KD = 2.0;
constexpr double STAGE3_KP = 100.0, STAGE3_KI = 10, STAGE3_KD = 2.0;
constexpr double STAGE4_KP = 100.0, STAGE4_KI = 10, STAGE4_KD = 2.0;

#endif // ARM_CONFIG_HPP
