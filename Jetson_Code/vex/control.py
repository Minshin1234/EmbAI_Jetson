"""
VEX V5 serial bridge – importable module.

Provides a VexSerial class that can be used by the Flask app to send
joint angle commands over serial. Also works standalone via __main__.
"""

import glob
import serial
import threading
import time


BAUD = 115200


def find_vex_port():
    """Auto-detect the VEX V5 serial port."""
    candidates = sorted(glob.glob("/dev/ttyACM*") + glob.glob("/dev/ttyUSB*"))
    if candidates:
        return candidates[0]
    return None


class VexSerial:
    """Thread-safe serial connection to the VEX V5 Brain."""

    def __init__(self, port=None, baud=BAUD):
        self.port = port or find_vex_port()
        self.baud = baud
        self._ser = None
        self._lock = threading.Lock()
        self._reader_thread = None
        self._running = False
        self._last_response = ""

    @property
    def connected(self):
        return self._ser is not None and self._ser.is_open

    def connect(self):
        """Open the serial connection. Returns True on success."""
        if self.port is None:
            print("[VEX] No serial port found.")
            return False
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.1)
            time.sleep(2)  # wait for V5 brain to settle
            self._running = True
            self._reader_thread = threading.Thread(target=self._reader, daemon=True)
            self._reader_thread.start()
            print(f"[VEX] Connected to {self.port} @ {self.baud} baud")
            return True
        except serial.SerialException as e:
            print(f"[VEX] Could not open {self.port}: {e}")
            self._ser = None
            return False

    def disconnect(self):
        self._running = False
        if self._ser and self._ser.is_open:
            try:
                self._ser.write(b"STOP\n")
            except Exception:
                pass
            self._ser.close()
        self._ser = None

    def send(self, command: str) -> bool:
        """Send a raw command string (newline appended automatically)."""
        if not self.connected:
            return False
        with self._lock:
            try:
                self._ser.write((command.strip() + "\n").encode())
                return True
            except Exception as e:
                print(f"[VEX] Write error: {e}")
                return False

    def send_all_joints_deg(self, angles_deg: list[float]) -> bool:
        """Send all joint angles in degrees: A d0 d1 d2 d3 d4."""
        if len(angles_deg) < 4:
            return False
        # Pad to 5 joints if only 4 provided (5th = 0 for gripper, etc.)
        while len(angles_deg) < 5:
            angles_deg.append(0.0)
        parts = " ".join(f"{a:.1f}" for a in angles_deg[:5])
        cmd = f"A {parts}"
        print(f"[VEX] Sending: {cmd}")
        return self.send(cmd)

    def send_single_joint_deg(self, joint_id: int, angle_deg: float) -> bool:
        """Send a single joint target: T <joint> <degrees>."""
        cmd = f"T {joint_id} {angle_deg:.1f}"
        print(f"[VEX] Sending: {cmd}")
        return self.send(cmd)

    def home(self) -> bool:
        return self.send("HOME")

    def stop(self) -> bool:
        return self.send("STOP")

    def status(self) -> bool:
        return self.send("STATUS")

    def request_status(self, timeout: float = 1.0) -> list[float] | None:
        """Send STATUS and wait for the V5 to reply with current angles.

        Expected V5 response format:  POS <d0> <d1> <d2> <d3> <d4>
        Returns [d0, d1, d2, d3, d4] in degrees, or None on timeout.
        """
        self._last_status_angles = None
        if not self.send("STATUS"):
            return None

        import time
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._last_status_angles is not None:
                return self._last_status_angles
            time.sleep(0.02)
        print("[VEX] STATUS response timed out")
        return None

    def _reader(self):
        """Background reader thread — prints V5 responses and parses STATUS."""
        while self._running:
            try:
                if self._ser and self._ser.is_open:
                    line = self._ser.readline()
                    if line:
                        decoded = line.decode(errors="ignore").strip()
                        self._last_response = decoded
                        print(f"[VEX] V5 > {decoded}")

                        # Parse STATUS response:  POS <d0> <d1> <d2> <d3> <d4>
                        if decoded.startswith("POS "):
                            try:
                                parts = decoded.split()
                                angles = [float(x) for x in parts[1:]]
                                self._last_status_angles = angles
                            except (ValueError, IndexError):
                                pass
            except Exception:
                break


# ── Standalone interactive mode ──────────────────────────────────────
if __name__ == "__main__":
    import sys

    vex = VexSerial()
    if not vex.connect():
        print("Failed to connect. Exiting.")
        sys.exit(1)

    print("Type commands and press Enter.")
    print("Commands: PING, T <j> <deg>, A <d0> <d1> <d2> <d3> <d4>, HOME, STOP, STATUS, q")

    try:
        while True:
            cmd = input(">> ").strip()
            if not cmd:
                continue
            if cmd.lower() == "q":
                vex.stop()
                break
            vex.send(cmd)
    except KeyboardInterrupt:
        vex.stop()
    finally:
        vex.disconnect()
