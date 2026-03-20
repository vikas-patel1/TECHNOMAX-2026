# =============================================================================
# robot_serial.py
# Serial communication layer between Python and the Arduino firmware.
#
# HOW TO USE:
#   Real hardware:  arm = RobotArm();  arm.connect()
#   No hardware:    arm = RobotArm(mock=True);  arm.connect()
#
# The mock Arduino runs in a background thread and speaks the exact same
# packet protocol as the real firmware — your code cannot tell the difference.
# Swap mock=True -> mock=False (or set SERIAL_PORT in config.py) when hardware
# arrives. Every other line of your code stays identical.
#
# Dependencies:
#   pip install pyserial
# =============================================================================

from __future__ import annotations

import os
import struct
import threading
import time
import importlib.util
from pathlib import Path
from typing import Optional


# ── Load config (same path-search as kinematics.py) ──────────────────────────

def _load_config():
    here    = Path(__file__).resolve().parent
    project = here.parent
    for path in [here/'config.py', here/'config'/'config.py',
                 project/'config.py', project/'config'/'config.py']:
        if path.exists():
            spec   = importlib.util.spec_from_file_location('_cfg', path)
            module = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
            spec.loader.exec_module(module)                     # type: ignore[union-attr]
            return module
    return None

_cfg = _load_config()

if _cfg is not None:
    SERIAL_PORT    = _cfg.SERIAL_PORT
    SERIAL_BAUD    = _cfg.SERIAL_BAUD
    SERIAL_TIMEOUT = _cfg.SERIAL_TIMEOUT
    SERIAL_RETRIES = _cfg.SERIAL_RETRIES
    HOME_ANGLES    = _cfg.HOME_ANGLES
    GRIPPER_OPEN   = _cfg.GRIPPER_OPEN_DEG
    GRIPPER_CLOSE  = _cfg.GRIPPER_CLOSE_DEG
    GRIPPER_WAIT   = _cfg.GRIPPER_WAIT_S
    DEBUG          = _cfg.DEBUG
    print("[robot_serial] loaded config")
else:
    print("[robot_serial] config.py not found — using built-in defaults")
    SERIAL_PORT    = None
    SERIAL_BAUD    = 115200
    SERIAL_TIMEOUT = 2.0
    SERIAL_RETRIES = 3
    HOME_ANGLES    = [90.0, 20.0, 160.0, -80.0]
    GRIPPER_OPEN   = 10.0
    GRIPPER_CLOSE  = 75.0
    GRIPPER_WAIT   = 0.4
    DEBUG          = True


# ── Packet protocol constants (must match Arduino firmware exactly) ───────────

HEADER_CMD   = 0xFF   # Python  → Arduino: command packet
HEADER_REPLY = 0xFE   # Arduino → Python:  ACK (success)
HEADER_NAK   = 0xFD   # Arduino → Python:  NAK (bad checksum, resend)
HEADER_READY = 0xAA   # Arduino → Python:  sent once on boot

# Packet layout (13 bytes total):
#   [0]     header  0xFF
#   [1-2]   J1 angle × 10  int16 big-endian
#   [3-4]   J2 angle × 10  int16 big-endian
#   [5-6]   J3 angle × 10  int16 big-endian
#   [7-8]   J4 angle × 10  int16 big-endian
#   [9-10]  Gripper × 10   int16 big-endian
#   [11]    checksum  (sum of bytes 0-10) mod 256
CMD_PACKET_SIZE   = 11   # bytes after the header (10 data + 1 checksum)
REPLY_PACKET_SIZE = 12   # header(1) + 5×int16(10) + checksum(1)


# =============================================================================
# MOCK ARDUINO
# Runs in a background thread. Speaks the exact same packet protocol as the
# real Arduino firmware. Uses Python's pty module (Linux/macOS) to create a
# virtual serial port pair, or a simple pipe-based fallback on Windows.
# =============================================================================

class _MockArduino(threading.Thread):
    """
    Software Arduino that replaces real hardware during development.

    Works on Windows, Linux, and macOS — uses Python queues internally
    instead of OS-level virtual serial ports. The RobotArm class talks
    to it through a lightweight _MockSerial adapter that looks exactly
    like a pyserial Serial object.

    Simulates servo movement delay proportional to angle change so
    timing behaviour matches a real arm.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._current: list[int] = [int(a * 10) for a in HOME_ANGLES] + [100]
        self._running = True
        self._lock    = threading.Lock()

        # Two queues: one for each direction
        # host_to_mock  — RobotArm writes, MockArduino reads
        # mock_to_host  — MockArduino writes, RobotArm reads
        import queue
        self._host_to_mock: queue.Queue[bytes] = queue.Queue()
        self._mock_to_host: queue.Queue[bytes] = queue.Queue()

        # port is not a real port string — _connect_mock uses the queues directly
        self.port = "MOCK"

    # ── Internal I/O (called from background thread) ──────────────────────────

    def _read_from_host(self, timeout: float = 0.1) -> bytes:
        """
        Read all available bytes from host queue.
        Does NOT demand a fixed byte count — takes whatever has arrived.
        Blocks for up to timeout seconds waiting for at least one byte.
        """
        import queue
        data = b''
        deadline = time.time() + timeout
        # Wait for at least one chunk
        while time.time() < deadline:
            try:
                chunk = self._host_to_mock.get(
                    timeout=min(0.02, deadline - time.time()))
                data += chunk
                break
            except queue.Empty:
                continue
        # Drain any additional chunks that already arrived
        while True:
            try:
                data += self._host_to_mock.get_nowait()
            except queue.Empty:
                break
        return data

    def _write_to_host(self, data: bytes) -> None:
        """Send bytes back to RobotArm."""
        # Push one byte at a time so the host can read(1) for the ready signal
        for b in data:
            self._mock_to_host.put(bytes([b]))

    # ── Thread body ───────────────────────────────────────────────────────────

    def run(self) -> None:
        # Send ready signal on boot (matches real firmware behaviour)
        time.sleep(0.05)
        self._write_to_host(bytes([HEADER_READY]))

        buf = b''
        while self._running:
            chunk = self._read_from_host(timeout=0.1)
            if not chunk:
                continue
            buf += chunk

            while len(buf) >= CMD_PACKET_SIZE + 1:
                if buf[0] != HEADER_CMD:
                    buf = buf[1:]
                    continue

                packet = buf[:CMD_PACKET_SIZE + 1]
                buf    = buf[CMD_PACKET_SIZE + 1:]

                # Validate checksum
                expected = sum(packet[:-1]) % 256
                if packet[-1] != expected:
                    if DEBUG:
                        print(f"[MockArduino] checksum error — "
                              f"got {packet[-1]}, expected {expected}")
                    self._write_to_host(self._build_reply(ok=False))
                    continue

                # Unpack 5 × int16 (tenths of degrees)
                raw = list(struct.unpack('>5h', packet[1:11]))

                # Clamp to safe ranges (mirrors real firmware limits)
                limits = [(0,1800),(0,1700),(0,1700),(-900,900),(100,800)]
                clamped = [max(lo, min(hi, v)) for v, (lo, hi) in zip(raw, limits)]

                # Simulate movement delay proportional to angle change
                with self._lock:
                    max_delta = max(abs(n - c) for n, c in
                                    zip(clamped, self._current))
                    move_time = min(max_delta / 1000.0 * 1.5, 1.5)
                    self._current = clamped

                time.sleep(move_time)

                if DEBUG:
                    angles = [v / 10.0 for v in clamped]
                    print(f"[MockArduino] moved to "
                          f"{[round(a, 1) for a in angles[:4]]}  "
                          f"gripper={angles[4]:.0f}°")

                self._write_to_host(self._build_reply(ok=True))

    def _build_reply(self, ok: bool) -> bytes:
        with self._lock:
            current = list(self._current)
        header  = HEADER_REPLY if ok else HEADER_NAK
        payload = struct.pack('>B5h', header, *current)
        cksum   = sum(payload) % 256
        return payload + struct.pack('B', cksum)

    def get_angles(self) -> list[float]:
        with self._lock:
            return [v / 10.0 for v in self._current]

    def stop(self) -> None:
        self._running = False


class _MockSerial:
    """
    Drop-in replacement for serial.Serial when using the mock Arduino.

    Implements exactly the methods RobotArm uses (read, write, close,
    reset_input_buffer, is_open, in_waiting) using the mock's queues.
    Works identically on Windows, Linux, and macOS.
    """

    def __init__(self, mock: _MockArduino) -> None:
        self._mock   = mock
        self.is_open = True

    def read(self, n: int) -> bytes:
        import queue
        data = b''
        deadline = time.time() + SERIAL_TIMEOUT
        while len(data) < n:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                chunk = self._mock._mock_to_host.get(
                    timeout=min(remaining, 0.05)
                )
                data += chunk
            except queue.Empty:
                continue
        return data

    def write(self, data: bytes) -> int:
        self._mock._host_to_mock.put(data)
        return len(data)

    def close(self) -> None:
        self.is_open = False

    def reset_input_buffer(self) -> None:
        import queue
        while not self._mock._mock_to_host.empty():
            try:
                self._mock._mock_to_host.get_nowait()
            except queue.Empty:
                break

    @property
    def in_waiting(self) -> int:
        return self._mock._mock_to_host.qsize()


# =============================================================================
# ROBOT ARM  —  the public API your state machine calls
# =============================================================================

class RobotArm:
    """
    High-level interface to the physical arm (or mock Arduino).

    Quick start
    -----------
    # With mock (no hardware needed):
    arm = RobotArm(mock=True)
    arm.connect()
    arm.home()
    arm.set_joints([90, 45, 90, -45], gripper=10)
    arm.open_gripper()
    arm.close_gripper()
    arm.disconnect()

    # With real hardware (change SERIAL_PORT in config.py first):
    arm = RobotArm()
    arm.connect()
    ...
    """

    def __init__(self, port: Optional[str] = None, mock: bool = False) -> None:
        self._port         = port or SERIAL_PORT
        self._mock_mode    = mock
        self._serial       = None   # serial.Serial instance (real hardware)
        self._mock         = None   # _MockArduino instance
        self._current      = list(HOME_ANGLES) + [GRIPPER_OPEN]
        self._connected    = False

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self, auto_detect: bool = True) -> None:
        """Open connection to the arm (mock or real)."""
        if self._mock_mode:
            self._connect_mock()
        else:
            self._connect_real(auto_detect)

    def _connect_mock(self) -> None:
        self._mock   = _MockArduino()
        self._mock.start()
        time.sleep(0.1)   # let mock thread start

        # Use _MockSerial — works on Windows, Linux, macOS without pyserial
        self._serial = _MockSerial(self._mock)

        self._wait_for_ready(timeout=3.0)
        self._connected = True
        print("[RobotArm] connected to mock Arduino  (Windows-compatible queue mode)")

    def _connect_real(self, auto_detect: bool) -> None:
        import serial as _serial
        import serial.tools.list_ports as _lp

        if self._port is None and auto_detect:
            self._port = self._find_arduino()
        if self._port is None:
            raise ConnectionError(
                "No Arduino found. Check USB cable and drivers, or set "
                "SERIAL_PORT in config.py."
            )

        print(f"[RobotArm] connecting to {self._port} @ {SERIAL_BAUD} baud...")
        self._serial = _serial.Serial(
            port      = self._port,
            baudrate  = SERIAL_BAUD,
            timeout   = SERIAL_TIMEOUT,
            bytesize  = _serial.EIGHTBITS,
            parity    = _serial.PARITY_NONE,
            stopbits  = _serial.STOPBITS_ONE,
        )
        # Arduino resets when serial opens — wait for it to boot
        time.sleep(2.0)
        self._serial.reset_input_buffer()

        if self._wait_for_ready(timeout=5.0):
            print("[RobotArm] Arduino ready")
        else:
            print("[RobotArm] warning: no ready signal — Arduino may already "
                  "be running")

        self._connected = True

    def _find_arduino(self) -> Optional[str]:
        """Scan serial ports for an Arduino by vendor string."""
        import serial.tools.list_ports as _lp
        for p in _lp.comports():
            desc = (p.description or '') + (p.manufacturer or '')
            if any(k in desc.lower() for k in
                   ['arduino', 'ch340', 'ch341', 'ftdi', 'atmega']):
                print(f"[RobotArm] found Arduino: {p.device} — {p.description}")
                return p.device
        print("[RobotArm] could not auto-detect Arduino. Available ports:")
        for p in _lp.comports():
            print(f"  {p.device}: {p.description}")
        return None

    def _wait_for_ready(self, timeout: float = 5.0) -> bool:
        """Wait for the 0xAA ready byte from firmware."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._serial.in_waiting > 0:  # type: ignore[union-attr]
                b = self._serial.read(1)      # type: ignore[union-attr]
                if b and b[0] == HEADER_READY:
                    return True
            time.sleep(0.01)
        return False

    def disconnect(self) -> None:
        """Close the connection and shut down mock if running."""
        if self._serial and self._serial.is_open:
            self._serial.close()
        if self._mock:
            self._mock.stop()
        self._connected = False
        print("[RobotArm] disconnected")

    # ── Core command ──────────────────────────────────────────────────────────

    def set_joints(self, joint_angles_deg: list[float],
                   gripper: Optional[float] = None) -> list[float]:
        """
        Send 4 joint angles + gripper angle to the arm.

        Parameters
        ----------
        joint_angles_deg : [J1, J2, J3, J4] in degrees
        gripper          : gripper angle in degrees (None = keep current)

        Returns
        -------
        Confirmed positions [J1, J2, J3, J4, gripper] as reported by firmware.

        Raises
        ------
        RuntimeError  if no valid reply after SERIAL_RETRIES attempts.
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")
        if len(joint_angles_deg) != 4:
            raise ValueError(f"Need 4 joint angles, got {len(joint_angles_deg)}")

        gripper_angle = gripper if gripper is not None else self._current[4]
        all_angles    = list(joint_angles_deg) + [gripper_angle]

        # Clear any stale data once before the first send.
        # Do NOT reset inside the retry loop — that drains the NAK reply
        # the firmware just sent, causing every retry to time out.
        self._serial.reset_input_buffer()       # type: ignore[union-attr]

        for attempt in range(1, SERIAL_RETRIES + 1):
            packet = self._build_packet(all_angles)
            self._serial.write(packet)           # type: ignore[union-attr]
            reply = self._read_reply()

            if reply is not None and reply['status'] == 'ACK':
                self._current = reply['angles']
                return reply['angles']
            if reply is not None and reply['status'] == 'NAK':
                print(f"[RobotArm] NAK on attempt {attempt}/{SERIAL_RETRIES} "
                      f"— resending")
            else:
                print(f"[RobotArm] timeout on attempt {attempt}/{SERIAL_RETRIES}")

        raise RuntimeError(
            f"Arm did not respond after {SERIAL_RETRIES} attempts."
        )

    # ── Convenience wrappers ──────────────────────────────────────────────────

    def home(self) -> list[float]:
        """Move to the safe home position defined in config.py."""
        return self.set_joints(HOME_ANGLES[:4], gripper=GRIPPER_OPEN)

    def open_gripper(self) -> list[float]:
        """Open the gripper fully."""
        angles = self._current[:4]
        result = self.set_joints(angles, gripper=GRIPPER_OPEN)
        time.sleep(GRIPPER_WAIT)
        return result

    def close_gripper(self) -> list[float]:
        """Close the gripper to grasp a ball."""
        angles = self._current[:4]
        result = self.set_joints(angles, gripper=GRIPPER_CLOSE)
        time.sleep(GRIPPER_WAIT)
        return result

    def get_current_angles(self) -> list[float]:
        """Return the last confirmed joint angles [J1, J2, J3, J4, gripper]."""
        return list(self._current)

    # ── Packet building ───────────────────────────────────────────────────────

    @staticmethod
    def _build_packet(angles_deg: list[float]) -> bytes:
        """
        Pack 5 angles into a 12-byte command packet.

        Format:
          byte 0     : 0xFF header
          bytes 1-10 : 5 × int16 big-endian  (angle × 10, tenths of degrees)
          byte 11    : checksum = sum(bytes 0-10) mod 256
        """
        raw    = [int(round(a * 10)) for a in angles_deg]
        data   = struct.pack('>B5h', HEADER_CMD, *raw)
        cksum  = sum(data) % 256
        return data + struct.pack('B', cksum)

    def _read_reply(self) -> Optional[dict]:
        """
        Read and parse the 11-byte reply from the Arduino.

        Returns {'status': 'ACK'|'NAK', 'angles': [j1..j5]}
        or None on timeout / bad data.
        """
        raw = self._serial.read(REPLY_PACKET_SIZE)  # type: ignore[union-attr]
        if len(raw) < REPLY_PACKET_SIZE:
            return None

        header = raw[0]
        if header not in (HEADER_REPLY, HEADER_NAK):
            return None

        # Verify checksum
        expected = sum(raw[:-1]) % 256
        if raw[-1] != expected:
            return None

        joint_raw = struct.unpack('>5h', raw[1:11])
        angles    = [v / 10.0 for v in joint_raw]

        return {
            'status': 'ACK' if header == HEADER_REPLY else 'NAK',
            'angles': angles,
        }