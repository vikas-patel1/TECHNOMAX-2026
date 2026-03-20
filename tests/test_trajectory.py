# =============================================================================
# test_trajectory.py
# Standalone test suite for trajectory.py
#
# HOW TO RUN  (from your project root):
#   python tests/test_trajectory.py
#
# Runs entirely without hardware — uses the mock Arduino from robot_serial.py.
# Tests are split into two groups:
#   Group A — pure math tests (no arm, no serial, instant)
#   Group B — execution tests (uses mock arm, verifies timing and motion)
#
# EXIT CODES:
#   0 — all tests passed
#   1 — one or more tests failed
# =============================================================================

import sys
import time
import importlib.util
from pathlib import Path

import numpy as np


# =============================================================================
# PATH SETUP — same pattern as test_serial.py
# =============================================================================

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC     = _PROJECT / 'src'

for _p in [_SRC, _HERE, _PROJECT]:
    s = str(_p)
    if _p.exists() and s not in sys.path:
        sys.path.insert(0, s)


def _load(filename: str):
    for directory in sys.path:
        path = Path(directory) / filename
        if path.exists():
            spec   = importlib.util.spec_from_file_location(
                         filename.replace('.py', ''), path)
            module = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
            spec.loader.exec_module(module)                  # type: ignore[union-attr]
            return module
    raise ImportError(
        f"Cannot find {filename}. "
        f"Make sure it is inside src/ and that src/ is in the project."
    )


traj = _load('trajectory.py')
rs   = _load('robot_serial.py')

RobotArm     = rs.RobotArm
interpolate  = traj.interpolate
check_vel    = traj.check_velocity
estimate_dur = traj.estimate_duration
clamp_wps    = traj.clamp_waypoints
move_to      = traj.move_to
move_via     = traj.move_via
pick_approach= traj.pick_approach
home_fn      = traj.home

JOINT_LIMITS = traj.JOINT_LIMITS
MAX_VEL      = traj.MAX_VEL
HOME_ANGLES  = traj.HOME_ANGLES


# =============================================================================
# TEST HELPERS
# =============================================================================

_PASS = 0
_FAIL = 0

def check(name: str, ok: bool, detail: str = '') -> None:
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

def close(a: float, b: float, tol: float = 0.5) -> bool:
    return abs(a - b) <= tol

def all_close(a: list, b: list, tol: float = 0.5) -> bool:
    return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))

def angles_in_limits(angles: list) -> bool:
    return all(lo <= a <= hi for a, (lo, hi) in zip(angles, JOINT_LIMITS))


# =============================================================================
# GROUP A — PURE MATH TESTS (no arm, no serial)
# These run instantly and test the spline logic in complete isolation.
# =============================================================================

def test_interpolate_basic() -> None:
    print("\n[A1] interpolate() — basic shape and endpoints")

    wps = interpolate([0,0,0,0], [90,90,90,90], steps=50, duration=1.0)

    check("returns 50 waypoints",
          len(wps) == 50, f"got {len(wps)}")

    check("each waypoint has 4 joints",
          all(len(w) == 4 for w in wps),
          f"sizes={[len(w) for w in wps[:3]]}")

    check("first waypoint near start [0,0,0,0]",
          all_close(wps[0], [0,0,0,0], tol=1.0),
          f"got {[round(v,1) for v in wps[0]]}")

    check("last waypoint near end [90,90,90,90]",
          all_close(wps[-1], [90,90,90,90], tol=1.0),
          f"got {[round(v,1) for v in wps[-1]]}")


def test_interpolate_monotonic() -> None:
    print("\n[A2] interpolate() — monotonic motion (all joints moving same direction)")

    wps = interpolate([10,10,10,-45], [90,80,100,0], steps=40, duration=1.5)

    # J1: should increase monotonically from 10 to 90
    j1_vals = [w[0] for w in wps]
    mono_ok = all(j1_vals[i] <= j1_vals[i+1] + 0.1 for i in range(len(j1_vals)-1))
    check("J1 increases monotonically",
          mono_ok,
          f"range {j1_vals[0]:.1f} -> {j1_vals[-1]:.1f}")

    # J4: should increase monotonically from -45 to 0
    j4_vals = [w[3] for w in wps]
    mono_j4 = all(j4_vals[i] <= j4_vals[i+1] + 0.1 for i in range(len(j4_vals)-1))
    check("J4 increases monotonically",
          mono_j4,
          f"range {j4_vals[0]:.1f} -> {j4_vals[-1]:.1f}")


def test_interpolate_no_teleport() -> None:
    print("\n[A3] interpolate() — no large jumps between consecutive steps")

    start = [90, 20, 160, -80]
    end   = [0,  45, 100, -45]
    steps = 30
    dur   = 1.0
    wps   = interpolate(start, end, steps=steps, duration=dur)

    dt           = dur / steps
    max_step_deg = MAX_VEL * dt * 1.5   # 1.5× to allow for spline peak

    max_jump = 0.0
    worst    = (0, 0)
    for i in range(len(wps) - 1):
        for j in range(4):
            jump = abs(wps[i+1][j] - wps[i][j])
            if jump > max_jump:
                max_jump = jump
                worst    = (i, j)

    check(f"no step jumps more than {max_step_deg:.1f} deg",
          max_jump <= max_step_deg,
          f"max jump={max_jump:.2f} deg at step {worst[0]} joint J{worst[1]+1}")


def test_interpolate_same_position() -> None:
    print("\n[A4] interpolate() — start equals end (no movement needed)")

    angles = [90, 25, 75, -30]
    wps    = interpolate(angles, angles, steps=20, duration=0.5)

    all_same = all(all_close(w, angles, tol=0.5) for w in wps)
    check("all waypoints equal start when start==end",
          all_same,
          f"first={[round(v,1) for v in wps[0]]}  "
          f"last={[round(v,1) for v in wps[-1]]}")


def test_interpolate_negative_angles() -> None:
    print("\n[A5] interpolate() — handles negative angles correctly (J4 wrist)")

    wps = interpolate([90, 30, 90, -80], [45, 50, 110, -20], steps=30, duration=1.0)

    check("first J4 near -80",
          close(wps[0][3], -80, tol=1.0),
          f"got {wps[0][3]:.1f}")

    check("last J4 near -20",
          close(wps[-1][3], -20, tol=1.0),
          f"got {wps[-1][3]:.1f}")

    check("J4 stays negative throughout",
          all(w[3] < 5 for w in wps),
          f"max J4={max(w[3] for w in wps):.1f}")


def test_check_velocity_safe() -> None:
    print("\n[A6] check_velocity() — does not stretch safe trajectories")

    start = [90, 25, 75, -30]
    end   = [90, 30, 80, -25]   # tiny move — well under MAX_VEL
    dur   = 1.0
    wps   = interpolate(start, end, steps=30, duration=dur)

    new_dur, new_wps = check_vel(wps, dur)

    check("safe trajectory: duration unchanged",
          abs(new_dur - dur) < 0.01,
          f"original={dur:.2f}s  new={new_dur:.2f}s")

    check("safe trajectory: waypoints unchanged",
          len(new_wps) == len(wps),
          f"original={len(wps)}  new={len(new_wps)}")


def test_check_velocity_stretch() -> None:
    print("\n[A7] check_velocity() — stretches duration for fast moves")

    # 170 degrees in 0.1 seconds would require 1700 deg/s — way over MAX_VEL=90
    start = [0,  0,   0,  -90]
    end   = [180, 170, 170, 90]
    dur   = 0.1
    wps   = interpolate(start, end, steps=30, duration=dur)

    new_dur, new_wps = check_vel(wps, dur)

    check("duration stretched for fast move",
          new_dur > dur,
          f"original={dur:.2f}s  stretched={new_dur:.2f}s")

    # Verify velocity is now safe
    dt = new_dur / (len(new_wps) - 1)
    max_v = max(
        abs(new_wps[i+1][j] - new_wps[i][j]) / dt
        for i in range(len(new_wps)-1)
        for j in range(4)
    )
    check("velocity now within MAX_VEL after stretch",
          max_v <= MAX_VEL * 1.05,   # 5% tolerance for float rounding
          f"max_vel={max_v:.1f} deg/s  limit={MAX_VEL:.1f} deg/s")


def test_estimate_duration() -> None:
    print("\n[A8] estimate_duration() — reasonable values")

    # Zero move → minimum duration
    d = estimate_dur([90,25,75,-30], [90,25,75,-30])
    check("zero move → minimum duration (0.4s)",
          close(d, 0.4, tol=0.05), f"got {d:.2f}s")

    # 90 deg move → reasonable duration
    d = estimate_dur([0,0,0,0], [90,0,0,0])
    check("90 deg move → 1.0-2.5s range",
          1.0 <= d <= 2.5, f"got {d:.2f}s")

    # Very large move → capped at 3.0s
    d = estimate_dur([0,0,0,-90], [180,170,170,90])
    check("max move → capped at 3.0s",
          close(d, 3.0, tol=0.05), f"got {d:.2f}s")


def test_clamp_waypoints() -> None:
    print("\n[A9] clamp_waypoints() — clips to joint limits")

    # Deliberately over-range waypoints
    wps = [
        [200, 200, 200, 200],    # all too high
        [-50, -50, -50, -150],   # all too low
        [90,  25,  75, -30],     # in range
    ]
    clamped = clamp_wps(wps)

    check("over-limit angles clamped to upper bound",
          all(clamped[0][j] <= JOINT_LIMITS[j][1] for j in range(4)),
          f"got {[round(v,1) for v in clamped[0]]}")

    check("under-limit angles clamped to lower bound",
          all(clamped[1][j] >= JOINT_LIMITS[j][0] for j in range(4)),
          f"got {[round(v,1) for v in clamped[1]]}")

    check("in-range waypoint unchanged",
          all_close(clamped[2], [90,25,75,-30], tol=0.1),
          f"got {[round(v,1) for v in clamped[2]]}")


# =============================================================================
# GROUP B — EXECUTION TESTS (uses mock arm via robot_serial mock)
# These verify timing accuracy and integration with set_joints().
# =============================================================================

def _make_arm() -> RobotArm:
    """Create and connect a mock arm for execution tests."""
    arm = RobotArm(mock=True)
    arm.connect()
    return arm


def test_move_to_reaches_target(arm) -> None:
    print("\n[B1] move_to() — arm reaches target within tolerance")

    test_cases = [
        ([90, 20, 160, -80], [0,  25,  50, -45], "home to forward"),
        ([0,  25,  50, -45], [45, 30,  90, -30], "forward to diagonal"),
        ([45, 30,  90, -30], [90, 20, 160, -80], "diagonal back to home"),
    ]

    for start_conf, target, desc in test_cases:
        # Force arm to start position
        arm.set_joints(start_conf)
        result = move_to(arm, target)

        check(f"move_to reaches target — {desc}",
              all_close(result[:4], target, tol=3.0),
              f"target={target}  confirmed={[round(v,1) for v in result[:4]]}")


def test_move_to_timing(arm) -> None:
    print("\n[B2] move_to() — duration accuracy")

    # Move with explicit duration.
    # In mock mode: set_joints() includes simulated servo delay so actual
    # wall-clock time is always >= requested. We test the lower bound only.
    # On real hardware: actual will be within 20% both ways.
    arm.set_joints([90, 20, 160, -80])
    target   = [0, 45, 100, -45]
    duration = 1.0

    t0     = time.perf_counter()
    move_to(arm, target, duration=duration, steps=20)
    actual = time.perf_counter() - t0

    check(f"wall-clock >= 80% of requested {duration:.1f}s (lower bound)",
          actual >= duration * 0.8,
          f"requested={duration:.2f}s  actual={actual:.2f}s")

    check(f"move completed (upper bound: actual < 60s)",
          actual < 60.0,
          f"actual={actual:.2f}s")


def test_move_to_auto_duration(arm) -> None:
    print("\n[B3] move_to() — auto duration scales with move size")

    arm.set_joints([90, 20, 160, -80])

    # Small move
    t0 = time.perf_counter()
    move_to(arm, [90, 25, 155, -75], steps=15)
    small_dur = time.perf_counter() - t0

    arm.set_joints([0, 0, 0, 0])

    # Large move
    t0 = time.perf_counter()
    move_to(arm, [170, 160, 160, 80], steps=15)
    large_dur = time.perf_counter() - t0

    check("large move takes longer than small move",
          large_dur > small_dur,
          f"small={small_dur:.2f}s  large={large_dur:.2f}s")


def test_move_to_all_joints_in_limits(arm) -> None:
    print("\n[B4] move_to() — all waypoints stay within joint limits")

    # Track every set_joints call
    called_with = []
    original_sj = arm.set_joints

    def tracking_set_joints(angles, gripper=None):
        called_with.append(list(angles))
        return original_sj(angles, gripper)

    arm.set_joints = tracking_set_joints

    arm.set_joints.__wrapped__ = True
    arm._current = list(HOME_ANGLES) + [10.0]

    # Do a large move
    move_to(arm, [0, 45, 100, -45], steps=30)

    arm.set_joints = original_sj

    violations = [
        (i, j, v)
        for i, wp in enumerate(called_with)
        for j, v in enumerate(wp)
        if not (JOINT_LIMITS[j][0] - 0.5 <= v <= JOINT_LIMITS[j][1] + 0.5)
    ]
    check("no joint limit violations across all waypoints",
          len(violations) == 0,
          f"{len(violations)} violations found: {violations[:3]}")


def test_move_via(arm) -> None:
    print("\n[B5] move_via() — passes through via-point then reaches target")

    arm.set_joints(list(HOME_ANGLES[:4]))
    via    = [45, 25, 80, -30]
    target = [0,  45, 100, -45]

    result = move_via(arm, via, target)

    check("move_via reaches final target",
          all_close(result[:4], target, tol=3.0),
          f"target={target}  confirmed={[round(v,1) for v in result[:4]]}")


def test_pick_approach(arm) -> None:
    print("\n[B6] pick_approach() — two-phase motion, ends at pick position")

    arm.set_joints(list(HOME_ANGLES[:4]))

    approach = [0, 45, 100, -45]   # 40mm above ball
    pick     = [0, 50, 110, -55]   # ball contact

    result = pick_approach(arm, approach, pick)

    check("pick_approach ends at pick position",
          all_close(result[:4], pick, tol=3.0),
          f"target={pick}  confirmed={[round(v,1) for v in result[:4]]}")


def test_home_fn(arm) -> None:
    print("\n[B7] home() — returns to HOME_ANGLES")

    # Start from somewhere else
    arm.set_joints([0, 45, 100, -45])
    result = home_fn(arm)

    check("home() reaches HOME_ANGLES",
          all_close(result[:4], HOME_ANGLES[:4], tol=3.0),
          f"target={HOME_ANGLES[:4]}  confirmed={[round(v,1) for v in result[:4]]}")


def test_full_pick_and_place_cycle(arm) -> None:
    print("\n[B8] Full pick-and-place trajectory cycle")

    # Simulate: home -> approach ball -> pick -> lift -> rotate -> bin -> home
    steps = [
        ("home",                     lambda: home_fn(arm)),
        ("open gripper",             lambda: arm.open_gripper()),
        ("approach above ball",      lambda: move_to(arm, [0,  45, 100, -45], steps=20)),
        ("descend to ball",          lambda: move_to(arm, [0,  50, 110, -55],
                                                     duration=0.5, steps=10)),
        ("close gripper",            lambda: arm.close_gripper()),
        ("lift up",                  lambda: move_to(arm, [0,  35,  80, -35], steps=20)),
        ("rotate to bin",            lambda: move_to(arm, [45, 35,  80, -35], steps=20)),
        ("descend to bin",           lambda: move_to(arm, [45, 50, 100, -45],
                                                     duration=0.5, steps=10)),
        ("open gripper",             lambda: arm.open_gripper()),
        ("return home",              lambda: home_fn(arm)),
    ]

    all_ok = True
    for name, fn in steps:
        try:
            result = fn()
            ok     = result is not None and len(result) >= 4
            check(f"  {name}", ok)
            if not ok:
                all_ok = False
        except Exception as e:
            check(f"  {name}", False, str(e))
            all_ok = False

    check("full trajectory cycle completed", all_ok)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    SEP = "=" * 62

    print(SEP)
    print("TRAJECTORY TEST SUITE")
    print(SEP)

    # ── Group A: pure math (no arm needed) ───────────────────────────────────
    print("\nGROUP A — Pure math tests (no hardware, no serial)")
    print("-" * 62)

    test_interpolate_basic()
    test_interpolate_monotonic()
    test_interpolate_no_teleport()
    test_interpolate_same_position()
    test_interpolate_negative_angles()
    test_check_velocity_safe()
    test_check_velocity_stretch()
    test_estimate_duration()
    test_clamp_waypoints()

    # ── Group B: execution (mock arm) ────────────────────────────────────────
    print("\nGROUP B — Execution tests (mock Arduino)")
    print("-" * 62)
    print("[setup] connecting mock arm...")

    arm = _make_arm()

    test_move_to_reaches_target(arm)
    test_move_to_timing(arm)
    test_move_to_auto_duration(arm)
    test_move_to_all_joints_in_limits(arm)
    test_move_via(arm)
    test_pick_approach(arm)
    test_home_fn(arm)
    test_full_pick_and_place_cycle(arm)

    arm.disconnect()

    # ── Summary ──────────────────────────────────────────────────────────────
    total = _PASS + _FAIL
    print(f"\n{SEP}")
    print(f"RESULT:  {_PASS}/{total} tests passed")
    if _FAIL == 0:
        print("ALL TESTS PASSED")
        print()
        print("trajectory.py is verified and ready.")
        print("Next steps:")
        print("  1. Build vision.py")
        print("  2. Build state_machine.py")
        print("  3. When hardware arrives — run end-to-end with real arm")
    else:
        print(f"{_FAIL} FAILURE(S) — review trajectory.py before hardware integration")
    print(SEP)

    sys.exit(0 if _FAIL == 0 else 1)


if __name__ == '__main__':
    main()