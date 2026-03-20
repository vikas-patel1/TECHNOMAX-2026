# =============================================================================
# test_serial.py
# Standalone test suite for robot_serial.py
#
# HOW TO RUN  (from your project root):
#   No hardware:    python tests/test_serial.py
#   Real hardware:  python tests/test_serial.py --real
#
# Works from any working directory — uses Path(__file__) to locate src/.
#
# EXIT CODES:
#   0 — all tests passed
#   1 — one or more tests failed
# =============================================================================

import sys
import time
import struct
import importlib.util
from pathlib import Path


# =============================================================================
# PATH FIX — add src/ to sys.path so robot_serial.py can always be found
# regardless of which directory you run this script from.
#
# Searches:  tests/../src/   ->  project_root/src/
#            tests/          ->  same folder as this test (fallback)
#            project root    ->  project_root/ (fallback)
# =============================================================================

_HERE    = Path(__file__).resolve().parent          # .../tests/
_PROJECT = _HERE.parent                              # .../TechnoMax2026/
_SRC     = _PROJECT / 'src'

for _p in [_SRC, _HERE, _PROJECT]:
    s = str(_p)
    if _p.exists() and s not in sys.path:
        sys.path.insert(0, s)


# =============================================================================
# LOAD robot_serial — works even if the module is not pip-installed
# =============================================================================

def _find_and_load(filename: str):
    """
    Find filename by searching sys.path directories in order.
    Returns the loaded module, or raises ImportError with a helpful message.
    """
    for directory in sys.path:
        path = Path(directory) / filename
        if path.exists():
            spec   = importlib.util.spec_from_file_location(
                         filename.replace('.py', ''), path)
            module = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
            spec.loader.exec_module(module)                  # type: ignore[union-attr]
            return module
    raise ImportError(
        f"Cannot find {filename}.\n"
        f"Searched: {[str(Path(d) / filename) for d in sys.path[:6]]}\n"
        f"Make sure robot_serial.py is inside your src/ folder."
    )


rs           = _find_and_load('robot_serial.py')
RobotArm     = rs.RobotArm
_MockArduino = rs._MockArduino

# Pull constants directly from the loaded module — never use bare 'from X import'
HOME_ANGLES   = rs.HOME_ANGLES
GRIPPER_OPEN  = rs.GRIPPER_OPEN
GRIPPER_CLOSE = rs.GRIPPER_CLOSE

USE_REAL = '--real' in sys.argv


# =============================================================================
# TEST HELPERS
# =============================================================================

_PASS = 0
_FAIL = 0

def _result(name: str, ok: bool, detail: str = '') -> None:
    global _PASS, _FAIL
    status = 'PASS' if ok else 'FAIL'
    line   = f"  [{status}]  {name}"
    if detail:
        line += f"  -  {detail}"
    print(line)
    if ok:
        _PASS += 1
    else:
        _FAIL += 1

def _close(a: float, b: float, tol: float = 1.0) -> bool:
    """Return True if a and b are within tol of each other."""
    return abs(a - b) <= tol


# =============================================================================
# INDIVIDUAL TESTS
# =============================================================================

def test_connection(arm) -> None:
    print("\n[T1] Connection")
    _result("arm.connect() completes without exception",
            arm._connected)
    _result("serial port is open",
            arm._serial is not None and arm._serial.is_open)


def test_packet_structure() -> None:
    """Verify _build_packet produces correctly-structured bytes (no connection needed)."""
    print("\n[T2] Packet structure")
    angles = [90.0, 20.0, 160.0, -80.0, 10.0]
    packet = RobotArm._build_packet(angles)

    _result("packet length is 12 bytes",
            len(packet) == 12,
            f"got {len(packet)}")

    _result("first byte is 0xFF header",
            packet[0] == 0xFF,
            f"got 0x{packet[0]:02X}")

    raw     = struct.unpack('>5h', packet[1:11])
    decoded = [v / 10.0 for v in raw]
    all_ok  = all(_close(d, e, 0.1) for d, e in zip(decoded, angles))
    _result("all 5 angles encoded correctly",
            all_ok,
            f"decoded={[round(v,1) for v in decoded]}  expected={angles}")

    expected_cksum = sum(packet[:-1]) % 256
    _result("checksum byte is correct",
            packet[-1] == expected_cksum,
            f"got {packet[-1]}, expected {expected_cksum}")


def test_home(arm) -> None:
    print("\n[T3] Home position")
    try:
        result = arm.home()
        _result("home() returns without exception", True)
        _result("reply contains 5 angle values",
                len(result) == 5,
                f"got {len(result)}")
        joints_ok = all(_close(r, h, 2.0) for r, h in
                        zip(result[:4], HOME_ANGLES[:4]))
        _result("confirmed angles match home angles",
                joints_ok,
                f"got={[round(v,1) for v in result[:4]]}  "
                f"expected={HOME_ANGLES[:4]}")
    except Exception as e:
        _result("home() call",                        False, str(e))
        _result("reply contains 5 angle values",      False)
        _result("confirmed angles match home angles",  False)


def test_set_joints(arm) -> None:
    print("\n[T4] set_joints - various positions")
    test_cases = [
        ([  0,  25,  50, -45], "forward reach"),
        ([ 45,  30,  90, -30], "diagonal right"),
        ([135,  30,  90, -30], "diagonal left"),
        ([ 90,  25,  75,   0], "side, wrist flat"),
        ([  0,  25, 100, -45], "deep forward"),
    ]
    for angles, desc in test_cases:
        try:
            result    = arm.set_joints(angles)
            joints_ok = all(_close(r, a, 2.0) for r, a in
                            zip(result[:4], angles))
            _result(f"set_joints {angles}  ({desc})",
                    joints_ok,
                    f"confirmed={[round(v,1) for v in result[:4]]}")
        except Exception as e:
            _result(f"set_joints {angles}  ({desc})", False, str(e))


def test_gripper(arm) -> None:
    print("\n[T5] Gripper open / close")
    try:
        result = arm.open_gripper()
        _result("open_gripper() completes", True)
        _result("gripper angle near OPEN",
                _close(result[4], GRIPPER_OPEN, 3.0),
                f"got={result[4]:.1f}  expected~{GRIPPER_OPEN}")
    except Exception as e:
        _result("open_gripper() completes", False, str(e))
        _result("gripper angle near OPEN",  False)

    try:
        result = arm.close_gripper()
        _result("close_gripper() completes", True)
        _result("gripper angle near CLOSE",
                _close(result[4], GRIPPER_CLOSE, 3.0),
                f"got={result[4]:.1f}  expected~{GRIPPER_CLOSE}")
    except Exception as e:
        _result("close_gripper() completes", False, str(e))
        _result("gripper angle near CLOSE",  False)


def test_retries(arm) -> None:
    """
    Deliberately corrupt the first packet checksum so the mock sends NAK.
    Verify RobotArm retries and succeeds on the second attempt.
    """
    print("\n[T6] Retry / error recovery")

    call_count = [0]
    original_build = RobotArm._build_packet

    def corrupt_first_only(angles):
        call_count[0] += 1
        pkt = bytearray(original_build(angles))
        if call_count[0] == 1:
            pkt[-1] = (pkt[-1] + 1) % 256   # corrupt checksum on first attempt
        return bytes(pkt)

    RobotArm._build_packet = staticmethod(corrupt_first_only)   # type: ignore[method-assign]
    try:
        result = arm.set_joints([90, 25, 75, 0])
        _result("retry fires on bad checksum",
                call_count[0] >= 2,
                f"took {call_count[0]} attempt(s)")
        _result("arm recovers and returns valid result",
                len(result) == 5,
                f"confirmed={[round(v,1) for v in result[:4]]}")
    except Exception as e:
        _result("retry fires on bad checksum",
                call_count[0] >= 2,
                f"took {call_count[0]} attempt(s)")
        _result("arm recovers and returns valid result", False, str(e))
    finally:
        RobotArm._build_packet = staticmethod(original_build)   # type: ignore[method-assign]


def test_out_of_range(arm) -> None:
    print("\n[T7] Out-of-range input handling")

    # Wrong number of joints
    try:
        arm.set_joints([0, 0, 0])
        _result("wrong joint count raises ValueError", False,
                "no exception raised")
    except ValueError:
        _result("wrong joint count raises ValueError", True)
    except Exception as e:
        _result("wrong joint count raises ValueError", False, str(e))

    # Extreme angles — firmware clamps, should not crash
    try:
        result = arm.set_joints([999, -999, 999, -999])
        _result("extreme angles clamped by firmware - no crash",
                len(result) == 5,
                f"confirmed={[round(v,1) for v in result[:4]]}")
    except Exception as e:
        _result("extreme angles clamped by firmware - no crash", False, str(e))


def test_get_current_angles(arm) -> None:
    print("\n[T8] get_current_angles() state tracking")
    arm.set_joints([0, 25, 75, -45])
    current = arm.get_current_angles()
    _result("returns list of 5 values",
            len(current) == 5,
            f"got {len(current)}")
    _result("J1 matches last commanded value",
            _close(current[0], 0.0, 2.0),
            f"got {current[0]:.1f}")


def test_simulated_pick_cycle(arm) -> None:
    """
    Complete pick-and-place sequence.
    This is the exact order the state machine will call these methods.
    If this passes, the communication layer is ready for hardware integration.
    """
    print("\n[T9] Simulated pick-and-place cycle")

    steps = [
        ("home",                   lambda: arm.home()),
        ("open gripper",           lambda: arm.open_gripper()),
        ("approach above ball",    lambda: arm.set_joints([0,  45, 100, -45])),
        ("descend to ball",        lambda: arm.set_joints([0,  50, 110, -55])),
        ("close gripper (pick)",   lambda: arm.close_gripper()),
        ("lift up",                lambda: arm.set_joints([0,  35,  80, -35])),
        ("rotate to bin",          lambda: arm.set_joints([45, 35,  80, -35])),
        ("descend to bin",         lambda: arm.set_joints([45, 50, 100, -45])),
        ("open gripper (release)", lambda: arm.open_gripper()),
        ("return home",            lambda: arm.home()),
    ]

    all_ok = True
    for name, action in steps:
        try:
            result = action()
            ok     = len(result) == 5
            _result(f"  {name}", ok)
            if not ok:
                all_ok = False
        except Exception as e:
            _result(f"  {name}", False, str(e))
            all_ok = False

    _result("full pick cycle completed", all_ok)


def test_disconnect(arm) -> None:
    print("\n[T10] Disconnect")
    arm.disconnect()
    _result("disconnect() completes cleanly", not arm._connected)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    SEP  = "=" * 60
    mode = "REAL HARDWARE" if USE_REAL else "MOCK ARDUINO"

    print(SEP)
    print(f"ROBOT_SERIAL TEST SUITE  -  {mode}")
    print(SEP)

    # T2 runs first — no connection needed
    test_packet_structure()

    # Connect
    print("\n[setup] connecting...")
    arm = RobotArm(mock=not USE_REAL)
    try:
        arm.connect()
    except Exception as e:
        print(f"\nFATAL: could not connect - {e}")
        if USE_REAL:
            print("Check USB cable and SERIAL_PORT in config/config.py")
        sys.exit(1)

    test_connection(arm)
    test_home(arm)
    test_set_joints(arm)
    test_gripper(arm)
    test_retries(arm)
    test_out_of_range(arm)
    test_get_current_angles(arm)
    test_simulated_pick_cycle(arm)
    test_disconnect(arm)

    # Summary
    total = _PASS + _FAIL
    print(f"\n{SEP}")
    print(f"RESULT:  {_PASS}/{total} tests passed")
    if _FAIL == 0:
        print("ALL TESTS PASSED")
        if not USE_REAL:
            print()
            print("Next steps when hardware arrives:")
            print("  1. Set SERIAL_PORT = 'COM4' in config/config.py")
            print("  2. Upload firmware/robot_arm_firmware.ino to Arduino")
            print("  3. Run:  python tests/test_serial.py --real")
            print("  4. Expect identical output.")
    else:
        print(f"{_FAIL} FAILURE(S) - fix robot_serial.py before connecting hardware")
    print(SEP)

    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == '__main__':
    main()