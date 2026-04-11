#include "ArmPID.hpp"

ArmPID::ArmPID(pros::AbstractMotor& motor,
               double kP, double kI, double kD,
               double external_gear_ratio)
    : motor_(motor),
      kP_(kP), kI_(kI), kD_(kD),
      gear_ratio_(external_gear_ratio),
      ext_rotation_(nullptr),
      rotation_offset_(0.0),
      target_motor_deg_(0.0),
      integral_(0.0),
      prev_error_(0.0),
      min_deg_(-1e9),
      max_deg_(1e9),
      integral_max_(6000.0),
      output_voltage_(0),
      active_(false),
      full_power_(false),
      full_power_deadband_(2.0),
      full_power_approach_(0.0),
      full_power_slow_mv_(4000),
      full_power_pid_hold_(false),
      was_in_fast_zone_(false),
      current_limit_ma_(0),
      motor_offset_(0.0),
      invert_motor_(false) {}

void ArmPID::setExternalRotation(pros::Rotation* rot_sensor) {
    ext_rotation_     = rot_sensor;
    rotation_offset_  = 0.0;
}

void ArmPID::resetRotation() {
    if (ext_rotation_) {
        rotation_offset_ = static_cast<double>(ext_rotation_->get_position()) / 100.0;
    }
}

void ArmPID::setRotationOffset(double offset) {
    rotation_offset_ = offset;
}

void ArmPID::setMotorOffset(double offset) {
    motor_offset_ = offset;
}

void ArmPID::setInvertMotor(bool invert) {
    invert_motor_ = invert;
}

void ArmPID::setLimits(double range_deg, double center_deg) {
    double half = range_deg / 2.0;
    min_deg_ = (center_deg - half) / gear_ratio_;
    max_deg_ = (center_deg + half) / gear_ratio_;
}

void ArmPID::setLimitsMinMax(double min_output_deg, double max_output_deg) {
    min_deg_ = min_output_deg / gear_ratio_;
    max_deg_ = max_output_deg / gear_ratio_;
}

void ArmPID::setTargetRadians(double target_rads) {
    double output_deg = target_rads * (180.0 / M_PI);
    double motor_deg  = output_deg / gear_ratio_;
    target_motor_deg_ = clamp(motor_deg, min_deg_, max_deg_);
    was_in_fast_zone_ = false;
    active_ = true;
}

void ArmPID::setTargetOutputDeg(double output_deg) {
    double motor_deg  = output_deg / gear_ratio_;
    target_motor_deg_ = clamp(motor_deg, min_deg_, max_deg_);
    was_in_fast_zone_ = false;
    active_ = true;
}

void ArmPID::setTargetMotorDeg(double deg) {
    target_motor_deg_ = clamp(deg, min_deg_, max_deg_);
    resetIntegral();
    was_in_fast_zone_ = false;
    active_ = true;
}

void ArmPID::setFullPower(bool enable, double deadband_deg, double approach_deg,
                          std::int32_t approach_mv, bool pid_hold) {
    full_power_          = enable;
    full_power_deadband_ = deadband_deg;
    full_power_approach_ = approach_deg;
    full_power_slow_mv_  = approach_mv;
    full_power_pid_hold_ = pid_hold;
    resetIntegral();
}

void ArmPID::setCurrentLimit(std::int32_t limit_ma) {
    current_limit_ma_ = limit_ma;
}

void ArmPID::resetIntegral() {
    integral_   = 0.0;
    prev_error_ = 0.0;
}

void ArmPID::setGains(double kP, double kI, double kD) {
    kP_ = kP;
    kI_ = kI;
    kD_ = kD;
}

double ArmPID::getPositionDeg() const {
    if (ext_rotation_) {
        double raw = static_cast<double>(ext_rotation_->get_position()) / 100.0;
        if (invert_motor_) {
            return rotation_offset_ - raw;
        }
        return raw - rotation_offset_;
    }
    if (invert_motor_) {
        return motor_offset_ - motor_.get_position();
    }
    return motor_.get_position() + motor_offset_;
}

void ArmPID::calculate(double dt_sec) {
    if (!active_) {
        motor_.brake();
        output_voltage_ = 0;
        return;
    }

    if (current_limit_ma_ > 0) {
        std::int32_t current_ma = motor_.get_current_draw();
        if (current_ma >= current_limit_ma_) {
            double pos = getPositionDeg();
            target_motor_deg_ = pos;
            resetIntegral();
            motor_.brake();
            output_voltage_ = 0;
            return;
        }
    }

    double actual_deg = getPositionDeg();
    double error      = target_motor_deg_ - actual_deg;

    if (full_power_) {
        double abs_err = std::abs(error);
        if (abs_err > full_power_deadband_) {
            if (full_power_approach_ > 0.0
                && was_in_fast_zone_
                && abs_err <= full_power_approach_) {
                output_voltage_ = (error > 0) ? full_power_slow_mv_
                                              : -full_power_slow_mv_;
                motor_.move_voltage(output_voltage_);
            } else {
                if (abs_err > full_power_approach_ && full_power_approach_ > 0.0) {
                    was_in_fast_zone_ = true;
                }
                output_voltage_ = (error > 0) ? PID_MAX_VOLTAGE_MV : -PID_MAX_VOLTAGE_MV;
                if (invert_motor_) output_voltage_ = -output_voltage_;
                motor_.move_voltage(output_voltage_);
            }
            return;
        }

        if (!full_power_pid_hold_) {
            motor_.brake();
            output_voltage_ = 0;
            return;
        }
        if (was_in_fast_zone_) {
            resetIntegral();
            was_in_fast_zone_ = false;
        }
    }

    integral_ += error * dt_sec;
    double i_contribution = kI_ * integral_;
    if (i_contribution > integral_max_) {
        integral_ = integral_max_ / kI_;
    } else if (i_contribution < -integral_max_) {
        integral_ = -integral_max_ / kI_;
    }

    double derivative = (dt_sec > 0.0) ? (error - prev_error_) / dt_sec : 0.0;
    prev_error_ = error;

    double output = (kP_ * error)
                  + (kI_ * integral_)
                  + (kD_ * derivative);

    output_voltage_ = static_cast<std::int32_t>(
        clamp(output, -static_cast<double>(PID_MAX_VOLTAGE_MV), static_cast<double>(PID_MAX_VOLTAGE_MV)));

    if (invert_motor_) output_voltage_ = -output_voltage_;
    motor_.move_voltage(output_voltage_);
}
