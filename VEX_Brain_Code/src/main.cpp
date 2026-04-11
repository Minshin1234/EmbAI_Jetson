#include "main.h"
#include "arm_config.hpp"
#include "arm_hardware.hpp"
#include "serial_handler.hpp"
#include "pros/rtos.hpp"

// ════════════════════════════════════════════════════════════════════════════
//  PROS LIFECYCLE CALLBACKS
// ════════════════════════════════════════════════════════════════════════════

void initialize() {
    init_hardware();
    init_serial();
}

void disabled() {
    // Kill all motor output immediately
    motor_base.move_voltage(0);
    motor_stage1.move_voltage(0);
    motor_stage2.move_voltage(0);
    motor_stage3.move_voltage(0);
    motor_stage4.move_voltage(0);
    status_text = "DISABLED";
    update_screen();
}

void competition_initialize() {}

void autonomous() {}

// ════════════════════════════════════════════════════════════════════════════
//  OPERATOR CONTROL  —  serial listener + watchdog
// ════════════════════════════════════════════════════════════════════════════

void opcontrol() {
    // Re-initialize serial states for active opcontrol
    init_serial();

    // Spawn the PID control loop as a separate RTOS task
    pros::Task pid_task(pid_task_fn, nullptr, "pid_task");

    send_line("V5 READY");
    status_text = "LISTENING";
    update_screen();

    while (true) {
        // ── Advance staggered joint sequence ───────────────────────────
        tick_stagger();

        // ── Read serial data byte-by-byte ──────────────────────────────
        while (serial_link.get_read_avail() > 0) {
            std::int32_t byte_read = serial_link.read_byte();
            if (byte_read < 0) {
                break;
            }

            char c = static_cast<char>(byte_read);

            if (c == '\n') {
                if (!rx_buffer.empty()) {
                    handle_command(rx_buffer);
                    rx_buffer.clear();
                }
            } else if (c != '\r') {
                rx_buffer.push_back(c);

                // Guard against buffer overrun from garbage data
                if (rx_buffer.size() > 64) {
                    rx_buffer.clear();
                    last_command = "OVERFLOW";
                    status_text  = "ERR OVERFLOW";
                    send_line("ERR OVERFLOW");
                    update_screen();
                }
            }
        }

        // ── Watchdog disabled (re-enable for production) ──────────────

        // Refresh display at ~20 Hz
        update_screen();

        pros::delay(SERIAL_PERIOD_MS);
    }
}