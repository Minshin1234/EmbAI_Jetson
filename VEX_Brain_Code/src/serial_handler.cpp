#include "serial_handler.hpp"
#include "arm_config.hpp"
#include "arm_hardware.hpp"
#include "pros/rtos.hpp"
#include <cstdio>

// ════════════════════════════════════════════════════════════════════════════
//  SERIAL STATE
// ════════════════════════════════════════════════════════════════════════════
pros::Serial serial_link(SERIAL_PORT, BAUDRATE);

std::string rx_buffer;
std::string last_command = "NONE";
std::string status_text = "INIT";
std::uint32_t last_cmd_time = 0;
bool timed_out = false;

struct StaggeredCommander {
    double targets[NUM_JOINTS];
    bool is_rad = false;
    int next_idx = NUM_JOINTS; // NUM_JOINTS = idle
    std::uint32_t next_time = 0;

    void launch(const double *new_targets, bool rad) {
        for (int i = 0; i < NUM_JOINTS; ++i) {
            targets[i] = new_targets[i];
        }
        is_rad = rad;
        next_idx = 0;
        next_time = pros::millis();
    }

    void abort() {
        next_idx = NUM_JOINTS;
    }

    void tick() {
        if (next_idx >= NUM_JOINTS) return;
        if (pros::millis() < next_time) return;

        if (is_rad) {
            joints[next_idx]->setTargetRadians(targets[next_idx]);
        } else {
            joints[next_idx]->setTargetOutputDeg(targets[next_idx]);
        }

        ++next_idx;
        next_time = pros::millis() + JOINT_STAGGER_MS;
    }
} stagger_cmd;

void tick_stagger() {
    stagger_cmd.tick();
}

// ════════════════════════════════════════════════════════════════════════════
//  FUNCTIONS
// ════════════════════════════════════════════════════════════════════════════
void send_line(const std::string &msg) {
  std::string out = msg + "\n";
  serial_link.write(reinterpret_cast<std::uint8_t *>(out.data()),
                    static_cast<std::int32_t>(out.size()));
}

void init_serial() {
  serial_link.flush();
  rx_buffer.clear();
  last_cmd_time = pros::millis();
  status_text = "READY";
}

void handle_command(const std::string &line) {
  last_command = line;
  timed_out = false;
  last_cmd_time = pros::millis();

  int joint_idx = 0;
  double v0 = 0.0, v1 = 0.0, v2 = 0.0, v3 = 0.0, v4 = 0.0;
  char reply[128];

  // ── Parsers ────────────────────────────────────────────────────────
  if (std::sscanf(line.c_str(), "T %d %lf", &joint_idx, &v0) == 2) {
    if (joint_idx < 0 || joint_idx >= NUM_JOINTS) {
      status_text = "ERR JOINT";
      send_line("ERR JOINT_RANGE");
    } else {
      joints[joint_idx]->setTargetOutputDeg(v0);
      status_text = "RUNNING";
      std::snprintf(reply, sizeof(reply), "OK T %d %.2f", joint_idx, v0);
      send_line(reply);
    }
  } 
  else if (std::sscanf(line.c_str(), "TR %d %lf", &joint_idx, &v0) == 2) {
    if (joint_idx < 0 || joint_idx >= NUM_JOINTS) {
      status_text = "ERR JOINT";
      send_line("ERR JOINT_RANGE");
    } else {
      joints[joint_idx]->setTargetRadians(v0);
      status_text = "RUNNING";
      std::snprintf(reply, sizeof(reply), "OK TR %d %.4f", joint_idx, v0);
      send_line(reply);
    }
  } 
  else if (std::sscanf(line.c_str(), "A %lf %lf %lf %lf %lf", &v0, &v1, &v2, &v3, &v4) == NUM_JOINTS) {
    double degs[] = {v0, v1, v2, v3, v4};
    stagger_cmd.launch(degs, false);
    status_text = "RUNNING";
    std::snprintf(reply, sizeof(reply), "OK A %.2f %.2f %.2f %.2f %.2f", v0, v1, v2, v3, v4);
    send_line(reply);
  } 
  else if (std::sscanf(line.c_str(), "AR %lf %lf %lf %lf %lf", &v0, &v1, &v2, &v3, &v4) == NUM_JOINTS) {
    double rads[] = {v0, v1, v2, v3, v4};
    stagger_cmd.launch(rads, true);
    status_text = "RUNNING";
    std::snprintf(reply, sizeof(reply), "OK AR %.4f %.4f %.4f %.4f %.4f", v0, v1, v2, v3, v4);
    send_line(reply);
  } 
  else if (std::sscanf(line.c_str(), "G %lf", &v0) == 1) {
    set_gripper_deg(v0);
    status_text = "GRIP";
    std::snprintf(reply, sizeof(reply), "OK G %.1f", v0);
    send_line(reply);
  } 
  // ── Text Commands ──────────────────────────────────────────────────
  else if (line == "STOP") {
    stagger_cmd.abort();
    freeze_all_joints();
    status_text = "STOPPED";
    send_line("OK STOP");
  } 
  else if (line == "PING") {
    status_text = "PING";
    send_line("PONG");
  } 
  else if (line == "HOME") {
    double zeros[] = {0.0, 0.0, 0.0, 0.0, 0.0};
    stagger_cmd.launch(zeros, false);
    status_text = "HOMING";
    send_line("OK HOME");
  } 
  else if (line == "STATUS") {
    std::snprintf(reply, sizeof(reply), "POS %.2f %.2f %.2f %.2f %.2f",
                  joints[0]->getPositionOutputDeg(), joints[1]->getPositionOutputDeg(),
                  joints[2]->getPositionOutputDeg(), joints[3]->getPositionOutputDeg(),
                  joints[4]->getPositionOutputDeg());
    send_line(reply);
  } 
  else {
    status_text = "BAD CMD";
    send_line("ERR UNKNOWN_CMD");
  }

  // Update screen once per command
  update_screen();
}
