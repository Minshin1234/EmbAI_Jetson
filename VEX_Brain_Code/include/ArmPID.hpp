#ifndef ARM_PID_HPP
#define ARM_PID_HPP

#include "pros/abstract_motor.hpp"
#include "pros/rotation.hpp"
#include "arm_config.hpp"
#include <algorithm>
#include <cmath>

/**
 * @brief Custom PID controller for a single robotic arm joint.
 *
 * Outputs raw millivolt commands via move_voltage() to bypass the VEX
 * internal motor PID.  Supports:
 *   - Any AbstractMotor (single Motor or MotorGroup)
 *   - External gear ratios (motor-side degrees ≠ output-side degrees)
 *   - Optional external rotation sensor for position feedback
 *   - Software hard-stop limits (min/max motor degrees)
 *   - Integral-windup protection (clamped integral term)
 *   - Radians input from an IK solver (ikpy / Jetson)
 *   - Bang-bang (full-power) mode for joints too heavy for proportional control
 */
class ArmPID {
public:
    // ── Construction ────────────────────────────────────────────────────
    ArmPID(pros::AbstractMotor& motor,
           double kP, double kI, double kD,
           double external_gear_ratio = 1.0);

    // ── External rotation sensor (used by 1st-stage shoulder) ───────────
    void setExternalRotation(pros::Rotation* rot_sensor);
    void resetRotation();
    void setRotationOffset(double offset);
    void setMotorOffset(double offset);
    void setInvertMotor(bool invert);

    // ── Software hard-stop limits ──────────────────────────────────────
    void setLimits(double range_deg, double center_deg = 0.0);
    void setLimitsMinMax(double min_output_deg, double max_output_deg);

    // ── Target setters ─────────────────────────────────────────────────
    void setTargetRadians(double target_rads);
    void setTargetOutputDeg(double output_deg);
    void setTargetMotorDeg(double deg);

    // ── Bang-bang (full-power) mode ────────────────────────────────────
    void setFullPower(bool enable,
                      double       deadband_deg  = 2.0,
                      double       approach_deg  = 0.0,
                      std::int32_t approach_mv   = 4000,
                      bool         pid_hold      = false);

    // ── Runtime current protection ─────────────────────────────────────
    void setCurrentLimit(std::int32_t limit_ma);

    // ── PID core ───────────────────────────────────────────────────────
    void calculate(double dt_sec);

    // ── Accessors ──────────────────────────────────────────────────────
    double getPositionDeg() const;
    double getTargetMotorDeg() const { return target_motor_deg_; }
    double getTargetOutputDeg() const { return target_motor_deg_ * gear_ratio_; }
    double getPositionOutputDeg() const { return getPositionDeg() * gear_ratio_; }
    std::int32_t getOutputVoltage() const { return output_voltage_; }

    void resetIntegral();
    void setGains(double kP, double kI, double kD);

private:
    // ── Helpers ─────────────────────────────────────────────────────────
    static double clamp(double v, double lo, double hi) {
        return std::max(lo, std::min(v, hi));
    }

    // ── Members ─────────────────────────────────────────────────────────
    pros::AbstractMotor& motor_;          // Motor *or* MotorGroup
    double           kP_, kI_, kD_;
    double           gear_ratio_;         // motor_teeth / output_teeth
    pros::Rotation*  ext_rotation_;       // nullptr → use motor encoder
    double           rotation_offset_;    // subtracted from get_position() to zero the sensor

    double           target_motor_deg_;
    double           integral_;
    double           prev_error_;
    double           min_deg_, max_deg_;
    double           integral_max_;
    std::int32_t     output_voltage_;
    bool             active_;
    bool             full_power_;           // bang-bang mode
    double           full_power_deadband_;  // ± degrees: brake zone
    double           full_power_approach_;  // ± degrees: slow-down zone (0 = disabled)
    std::int32_t     full_power_slow_mv_;   // voltage used in the slow-down zone
    bool             full_power_pid_hold_;  // true: PID runs inside deadband for holding
    bool             was_in_fast_zone_;     // true once arm left the approach zone toward target
    std::int32_t     current_limit_ma_;     // runtime current protection (0 = disabled)
    double           motor_offset_;          // software offset added to motor encoder (degrees)
    bool             invert_motor_;           // negate motor position and voltage for inverted coords
};

#endif // ARM_PID_HPP
