#ifndef ARM_HARDWARE_HPP
#define ARM_HARDWARE_HPP

#include "ArmPID.hpp"
#include "arm_config.hpp"
#include "pros/motors.hpp"
#include "pros/motor_group.hpp"
#include "pros/rotation.hpp"
#include "pros/adi.hpp"

// ════════════════════════════════════════════════════════════════════════════
//  HARDWARE OBJECTS (Externs)
// ════════════════════════════════════════════════════════════════════════════
extern pros::Motor      motor_base;
extern pros::MotorGroup motor_stage1;
extern pros::Motor      motor_stage2;
extern pros::Motor      motor_stage3;
extern pros::Motor      motor_stage4;

extern pros::Rotation   rotation_stage1;

// PID Controllers
extern ArmPID pid_base;
extern ArmPID pid_stage1;
extern ArmPID pid_stage2;
extern ArmPID pid_stage3;
extern ArmPID pid_stage4;

extern ArmPID* joints[NUM_JOINTS];

// ════════════════════════════════════════════════════════════════════════════
//  UTILITY PROTOTYPES
// ════════════════════════════════════════════════════════════════════════════
void update_screen();
void freeze_all_joints();
void pid_task_fn(void* param);
void init_hardware();

/// Set the SG90 gripper servo to a given angle (0–180°).
void set_gripper_deg(double angle_deg);

#endif // ARM_HARDWARE_HPP
