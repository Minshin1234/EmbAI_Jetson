#ifndef SERIAL_HANDLER_HPP
#define SERIAL_HANDLER_HPP

#include "pros/serial.hpp"
#include <string>
#include <cstdint>

// ════════════════════════════════════════════════════════════════════════════
//  SERIAL STATE (Externs)
// ════════════════════════════════════════════════════════════════════════════
extern pros::Serial  serial_link;
extern std::string   rx_buffer;
extern std::string   last_command;
extern std::string   status_text;
extern std::uint32_t last_cmd_time;
extern bool          timed_out;

// ════════════════════════════════════════════════════════════════════════════
//  PROTOTYPES
// ════════════════════════════════════════════════════════════════════════════
void send_line(const std::string& msg);
void handle_command(const std::string& line);
void init_serial();
void tick_stagger(); // advance staggered joint sequence; call each serial loop iteration

#endif // SERIAL_HANDLER_HPP
