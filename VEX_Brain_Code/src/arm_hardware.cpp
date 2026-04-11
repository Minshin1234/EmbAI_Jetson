#include "arm_hardware.hpp"
#include "main.h"
#include "pros/rtos.hpp"
#include "pros/screen.hpp"
#include "serial_handler.hpp" // For status_text, last_command, etc.
#include <cstdio>

// ════════════════════════════════════════════════════════════════════════════
//  HARDWARE OBJECTS
// ════════════════════════════════════════════════════════════════════════════
pros::Motor motor_base(BASE_PORT);
pros::MotorGroup motor_stage1({STAGE1_PORT_A, STAGE1_PORT_B});
pros::Motor motor_stage2(STAGE2_PORT);
pros::Motor motor_stage3(STAGE3_PORT);
pros::Motor motor_stage4(STAGE4_PORT);

pros::Rotation rotation_stage1(ROTATION_PORT);

ArmPID pid_base(motor_base, BASE_KP, BASE_KI, BASE_KD, BASE_GEAR_RATIO);
ArmPID pid_stage1(motor_stage1, STAGE1_KP, STAGE1_KI, STAGE1_KD, 1.0);
ArmPID pid_stage2(motor_stage2, STAGE2_KP, STAGE2_KI, STAGE2_KD, 1.0);
ArmPID pid_stage3(motor_stage3, STAGE3_KP, STAGE3_KI, STAGE3_KD, 1.0);
ArmPID pid_stage4(motor_stage4, STAGE4_KP, STAGE4_KI, STAGE4_KD, 1.0);

ArmPID *joints[NUM_JOINTS] = {&pid_base, &pid_stage1, &pid_stage2, &pid_stage3,
                              &pid_stage4};

// SG90 gripper servo on ADI 3-wire port (legacy servo mode)
pros::adi::Port gripper_servo(GRIPPER_ADI_PORT, pros::E_ADI_LEGACY_SERVO);

// ════════════════════════════════════════════════════════════════════════════
//  UTILITY FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════
void update_screen() {
  pros::screen::erase();
  pros::screen::print(TEXT_MEDIUM, 0, "5-DOF Arm | RS-485 Controller");
  pros::screen::print(TEXT_SMALL, 1, "CMD : %s", last_command.c_str());
  pros::screen::print(TEXT_SMALL, 2, "STAT: %s | AGE: %lu ms", 
                      status_text.c_str(), 
                      static_cast<unsigned long>(pros::millis() - last_cmd_time));

  // Show target / actual / output voltage for each joint (output-space degrees)
  for (int i = 0; i < NUM_JOINTS; ++i) {
    pros::screen::print(TEXT_SMALL, 4 + i, "J%d | tgt: %5.1f | act: %5.1f | %5d mV", i,
                        joints[i]->getTargetOutputDeg(),
                        joints[i]->getPositionOutputDeg(),
                        joints[i]->getOutputVoltage());
  }

  // Motor temperatures — VEX V5 throttles at ~55°C, shuts off at 75°C
  double temps[6] = { 
      motor_base.get_temperature(), 
      motor_stage1.get_temperature(0), motor_stage1.get_temperature(1),
      motor_stage2.get_temperature(), 
      motor_stage3.get_temperature(), 
      motor_stage4.get_temperature() 
  };
  
  bool hot = false;
  for(double t : temps) { if(t > 55.0) hot = true; }

  pros::screen::print(TEXT_SMALL, 10, "TMP | B:%.0f 1:%.0f,%.0f 2:%.0f 3:%.0f 4:%.0f %s", 
                      temps[0], temps[1], temps[2], temps[3], temps[4], temps[5], 
                      hot ? "[WARN: HOT]" : "");
}

void set_gripper_deg(double angle_deg) {
  // Clamp to [0, 180]
  if (angle_deg < 0.0)   angle_deg = 0.0;
  if (angle_deg > 180.0) angle_deg = 180.0;
  // Map 0–180° to ADI servo range 0–127
  // (Negative values drive the SG90 past its physical stop)
  int value = static_cast<int>((angle_deg / 180.0) * 127.0);
  gripper_servo.set_value(value);
}

void freeze_all_joints() {
  for (int i = 0; i < NUM_JOINTS; ++i) {
    joints[i]->setTargetMotorDeg(joints[i]->getPositionDeg());
    joints[i]->resetIntegral();
  }
}

void pid_task_fn(void *param) {
  (void)param;
  while (true) {
    for (int i = 0; i < NUM_JOINTS; ++i) {
      joints[i]->calculate(PID_DT_SEC);
    }
    pros::delay(PID_PERIOD_MS);
  }
}

// ════════════════════════════════════════════════════════════════════════════
//  HOMING — drive into physical hard stops and tare encoders
// ════════════════════════════════════════════════════════════════════════════

template <typename CurrentFunc>
static void wait_for_stall(std::uint32_t timeout_ms, CurrentFunc get_current_ma) {
  std::uint32_t start_time = pros::millis();
  std::uint32_t stall_start = 0;
  bool stalling = false;

  while (pros::millis() - start_time < timeout_ms) {
    if (get_current_ma() >= HOMING_CURRENT_THRESHOLD_MA) {
      if (!stalling) {
        stalling = true;
        stall_start = pros::millis();
      } else if (pros::millis() - stall_start >= HOMING_STALL_TIME_MS) {
        return; // Confirmed stall
      }
    } else {
      stalling = false;
    }
    pros::delay(HOMING_POLL_MS);
  }
}

/**
 * @brief Drive a single motor into its hard stop, detect the stall via current
 *        spike, then stop and tare the encoder.
 */
static void home_motor(pros::AbstractMotor &motor, std::int32_t voltage_mv,
                       double tare_position, const char *label) {
  pros::screen::print(TEXT_MEDIUM, 4, "Homing %s...", label);
  motor.move_voltage(voltage_mv);
  wait_for_stall(HOMING_TIMEOUT_MS, [&]() { return motor.get_current_draw(); });
  
  motor.move_voltage(0);
  motor.brake();
  pros::delay(HOMING_SETTLE_MS);
  motor.tare_position();

  pros::screen::print(TEXT_MEDIUM, 4, "Homed %s at %.0f deg", label, tare_position);
}

/**
 * @brief Home joint 1 (MotorGroup) using the rotation sensor for position reference.
 */
static void home_stage1(pros::MotorGroup &motor, pros::Rotation &rot_sensor,
                        ArmPID &pid) {
  pros::screen::print(TEXT_MEDIUM, 4, "Homing J1 (shoulder)...");
  motor.move_voltage(HOMING_VOLTAGE_MV);
  wait_for_stall(HOMING_TIMEOUT_MS, [&]() {
    return std::max(motor.get_current_draw(0), motor.get_current_draw(1));
  });

  motor.move_voltage(0);
  motor.brake();
  pros::delay(HOMING_SETTLE_MS);

  double raw_deg = static_cast<double>(rot_sensor.get_position()) / 100.0;
  pid.setRotationOffset(raw_deg + STAGE1_TARE_DEG);

  pros::screen::print(TEXT_MEDIUM, 4, "Homed J1 at %.0f deg", STAGE1_TARE_DEG);
}

// ════════════════════════════════════════════════════════════════════════════
//  INITIALIZATION
// ════════════════════════════════════════════════════════════════════════════
void init_hardware() {
  // ── Brake mode: HOLD keeps joints from falling under gravity ───────
  motor_base.set_brake_mode(pros::E_MOTOR_BRAKE_HOLD);
  motor_stage1.set_brake_mode(pros::E_MOTOR_BRAKE_HOLD);
  motor_stage2.set_brake_mode(pros::E_MOTOR_BRAKE_HOLD);
  motor_stage3.set_brake_mode(pros::E_MOTOR_BRAKE_HOLD);
  motor_stage4.set_brake_mode(pros::E_MOTOR_BRAKE_HOLD);

  // ── Cartridge gearing: all motors use green (200 RPM, E_MOTOR_GEARSET_18)
  motor_base.set_gearing(pros::E_MOTOR_GEARSET_18);
  motor_stage1.set_gearing(pros::E_MOTOR_GEARSET_18);
  motor_stage2.set_gearing(pros::E_MOTOR_GEARSET_18);
  motor_stage3.set_gearing(pros::E_MOTOR_GEARSET_18);
  motor_stage4.set_gearing(pros::E_MOTOR_GEARSET_18);

  // ── Encoder units: degrees ────────────────────────────────────────
  motor_base.set_encoder_units(pros::E_MOTOR_ENCODER_DEGREES);
  motor_stage1.set_encoder_units(pros::E_MOTOR_ENCODER_DEGREES);
  motor_stage2.set_encoder_units(pros::E_MOTOR_ENCODER_DEGREES);
  motor_stage3.set_encoder_units(pros::E_MOTOR_ENCODER_DEGREES);
  motor_stage4.set_encoder_units(pros::E_MOTOR_ENCODER_DEGREES);

  // ── Base encoder: tare at current position (no hard stop to home) ──
  motor_base.tare_position();

  // ── Joint-specific setup ──────────────────────────────────────────
  pid_stage1.setExternalRotation(&rotation_stage1);

  // Wait for the rotation sensor to settle
  pros::delay(200);

  // ── Screen setup for homing progress ──────────────────────────────
  pros::screen::set_eraser(0x000000);
  pros::screen::erase();
  pros::screen::set_pen(0x00FFFFFF);
  pros::screen::print(TEXT_MEDIUM, 0, "5-DOF Arm  HOMING...");

  // ════════════════════════════════════════════════════════════════════
  //  HOMING SEQUENCE (joints 1–4)
  //
  //  Joint 1: drives toward -52° hard stop, tare as -52°.
  //  Joints 2–4: drive toward +90° hard stop, tare as +90°.
  //
  //  Joint 0 (base) has no hard stop — uses relative encoding only.
  // ════════════════════════════════════════════════════════════════════

  // Joint 1 — shoulder: uses rotation sensor, drives toward -52° hard stop
  home_stage1(motor_stage1, rotation_stage1, pid_stage1);
  pid_stage1.setInvertMotor(true);  // positive voltage drives into stop; invert for PID
  pid_stage1.setTargetMotorDeg(pid_stage1.getPositionDeg()); // hold here

  // Joint 2 — elbow: motor tares to 0, inverted coords (+90° at stop)
  home_motor(motor_stage2, -HOMING_VOLTAGE_MV, STAGE234_TARE_DEG, "J2 (elbow)");
  pid_stage2.setInvertMotor(true);
  pid_stage2.setMotorOffset(STAGE234_TARE_DEG);
  pid_stage2.setTargetMotorDeg(pid_stage2.getPositionDeg()); // hold here

  // Joint 3 — wrist: motor tares to 0, inverted coords (+90° at stop)
  home_motor(motor_stage3, -HOMING_VOLTAGE_MV, STAGE234_TARE_DEG, "J3 (wrist)");
  pid_stage3.setInvertMotor(true);
  pid_stage3.setMotorOffset(STAGE234_TARE_DEG);
  pid_stage3.setTargetMotorDeg(pid_stage3.getPositionDeg()); // hold here

  // Joint 4 — gripper: motor tares to 0, inverted coords (+90° at stop)
  home_motor(motor_stage4, -HOMING_VOLTAGE_MV, STAGE234_TARE_DEG, "J4 (gripper)");
  pid_stage4.setInvertMotor(true);
  pid_stage4.setMotorOffset(STAGE234_TARE_DEG);
  pid_stage4.setTargetMotorDeg(pid_stage4.getPositionDeg()); // hold here

  // ── Software Hard-Stop Limits ──────────────────────────────────────
  pid_stage1.setLimitsMinMax(STAGE1_MIN_DEG, STAGE1_MAX_DEG);
  pid_stage2.setLimitsMinMax(STAGE234_MIN_DEG, STAGE234_MAX_DEG);
  pid_stage3.setLimitsMinMax(STAGE234_MIN_DEG, STAGE234_MAX_DEG);
  pid_stage4.setLimitsMinMax(STAGE234_MIN_DEG, STAGE234_MAX_DEG);

  // ── All joints use standard PID (no bang-bang) ──────────────────────

  // ── Runtime current protection ─────────────────────────────────────
  // If any motor exceeds this threshold, snap its target to the current
  // position so it stops straining and just holds.
  for (int i = 0; i < NUM_JOINTS; ++i) {
    joints[i]->setCurrentLimit(MOTOR_CURRENT_LIMIT_MA);
  }

  // Set all PID targets to current positions so the arm holds in place
  freeze_all_joints();

  pros::screen::erase();
  pros::screen::print(TEXT_MEDIUM, 0, "5-DOF Arm  HOMING COMPLETE");
  printf("5-DOF Arm Hardware initialized (homed)\n");
}
