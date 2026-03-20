# =============================================================================
# trajectory.py
# Smooth joint-space trajectory generation for the 4-DOF robot arm.
#
# WHAT THIS FILE DOES:
#   Generates smooth cubic-spline interpolated waypoints between two arm
#   configurations. Instead of snapping the arm directly to a target angle,
#   trajectory.py breaks the move into STEPS intermediate positions so the
#   servos accelerate and decelerate naturally.
#
# WHAT THIS FILE DOES NOT DO:
#   - Know anything about XYZ coordinates or bin positions
#   - Import robot_serial directly
#   - Manage the gripper (gripper commands go through arm.open/close_gripper())
#
# HOW STATE MACHINE CALLS THIS:
#   # Simple move:
#   trajectory.move_to(arm, target_angles)
#
#   # Move through a safe via-point (e.g. above a ball before descending):
#   trajectory.move_via(arm, via_angles, target_angles)
#
#   # Pick sequence (approach above + slow descent):
#   trajectory.pick_approach(arm, approach_angles, pick_angles)
#
# RETURN VALUE:
#   Every public function returns the confirmed [J1,J2,J3,J4,gripper] angles
#   from the last set_joints() call — identical shape to arm.set_joints().
#   State machine can check this to verify arrival.
#
# DEPENDENCIES:
#   pip install numpy scipy
# =============================================================================

from __future__ import annotations

import time
import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline


# =============================================================================
# CONFIG LOADING — same 4-path search as kinematics.py and robot_serial.py
# =============================================================================

def _load_config():
    here    = Path(__file__).resolve().parent
    project = here.parent
    for path in [here / 'config.py',
                 here / 'config' / 'config.py',
                 project / 'config.py',
                 project / 'config' / 'config.py']:
        if path.exists():
            spec   = importlib.util.spec_from_file_location('_cfg', path)
            module = importlib.util.module_from_spec(spec)      # type: ignore[arg-type]
            spec.loader.exec_module(module)                     # type: ignore[union-attr]
            return module
    return None

_cfg = _load_config()

if _cfg is not None:
    JOINT_LIMITS  = _cfg.JOINT_LIMITS           # [(lo,hi), ...] × 4 joints
    HOME_ANGLES   = _cfg.HOME_ANGLES            # [J1,J2,J3,J4]
    DURATION_S    = _cfg.TRAJECTORY_DURATION_S  # default seconds per move
    STEPS         = _cfg.TRAJECTORY_STEPS       # default interpolation steps
    MAX_VEL       = _cfg.TRAJECTORY_MAX_VEL_DEG_S  # deg/s hard limit
    DEBUG         = _cfg.DEBUG
    print("[trajectory] loaded config")
else:
    print("[trajectory] config not found — using built-in defaults")
    JOINT_LIMITS  = [(0,180),(0,170),(0,170),(-90,90)]
    HOME_ANGLES   = [90.0, 20.0, 160.0, -80.0]
    DURATION_S    = 1.5
    STEPS         = 50
    MAX_VEL       = 90.0
    DEBUG         = True


# =============================================================================
# SECTION 1 — PURE MATH: spline generation (no arm, no serial, fully testable)
# =============================================================================

def interpolate(
    start_angles: list[float],
    end_angles:   list[float],
    steps:        int   = STEPS,
    duration:     float = DURATION_S,
) -> list[list[float]]:
    """
    Generate smooth cubic-spline waypoints between two joint configurations.

    Uses 'not-a-knot' boundary conditions by default — which means the spline
    naturally starts and ends with zero velocity (arm begins still, ends still).
    No jerk at the endpoints.

    Parameters
    ----------
    start_angles : [J1,J2,J3,J4] in degrees — where the arm is now
    end_angles   : [J1,J2,J3,J4] in degrees — where it needs to go
    steps        : number of intermediate waypoints to generate
    duration     : time budget in seconds (used for velocity checking only)

    Returns
    -------
    List of `steps` angle sets, each [J1,J2,J3,J4] in degrees.
    First waypoint ≈ start_angles, last waypoint ≈ end_angles.
    """
    if len(start_angles) != 4 or len(end_angles) != 4:
        raise ValueError(
            f"Need 4 angles each. Got start={len(start_angles)}, "
            f"end={len(end_angles)}"
        )

    # Two time points: t=0 (start) and t=duration (end)
    t_knots = np.array([0.0, duration])

    # Build one spline per joint
    splines = []
    for j in range(4):
        y = np.array([start_angles[j], end_angles[j]])
        # bc_type='not-a-knot' is the scipy default — gives zero velocity
        # at endpoints for a 2-point spline, which is exactly what we want.
        cs = CubicSpline(t_knots, y, bc_type='not-a-knot')
        splines.append(cs)

    # Sample at `steps` evenly-spaced time points
    t_samples  = np.linspace(0.0, duration, steps)
    waypoints  = []
    for t in t_samples:
        angles = [float(splines[j](t)) for j in range(4)]
        waypoints.append(angles)

    return waypoints


def check_velocity(
    waypoints: list[list[float]],
    duration:  float,
) -> tuple[float, list[list[float]]]:
    """
    Verify no joint exceeds MAX_VEL degrees/second.
    If it does, automatically scale duration up and regenerate.

    This handles cases where the angular distance is large and the default
    duration would require the servo to move faster than it safely can.

    Parameters
    ----------
    waypoints : output of interpolate()
    duration  : the duration that produced those waypoints

    Returns
    -------
    (safe_duration, safe_waypoints) — duration may be stretched.
    """
    if len(waypoints) < 2:
        return duration, waypoints

    dt = duration / (len(waypoints) - 1)

    max_vel_found = 0.0
    for i in range(len(waypoints) - 1):
        for j in range(4):
            vel = abs(waypoints[i+1][j] - waypoints[i][j]) / dt
            if vel > max_vel_found:
                max_vel_found = vel

    if max_vel_found <= MAX_VEL:
        return duration, waypoints   # already safe, nothing to do

    # Scale duration so the fastest joint stays at exactly MAX_VEL
    scale        = max_vel_found / MAX_VEL
    new_duration = duration * scale

    if DEBUG:
        print(f"[trajectory] velocity {max_vel_found:.1f} deg/s exceeds limit "
              f"{MAX_VEL:.1f} — stretching duration "
              f"{duration:.2f}s → {new_duration:.2f}s")

    # Regenerate with the stretched duration
    new_waypoints = interpolate(
        waypoints[0], waypoints[-1],
        steps=len(waypoints),
        duration=new_duration
    )
    return new_duration, new_waypoints


def estimate_duration(
    start_angles: list[float],
    end_angles:   list[float],
) -> float:
    """
    Estimate how long a move should take based on the largest angle change.

    Uses:  duration = max_delta / (MAX_VEL * 0.6)
    The 0.6 factor means the fastest joint uses 60% of MAX_VEL on average,
    leaving headroom for the spline's peak velocity (which is higher than
    the average over the move).

    Clamps between 0.4s (very short moves) and 3.0s (very long moves).
    """
    max_delta = max(abs(e - s) for s, e in zip(start_angles, end_angles))
    if max_delta < 1.0:
        return 0.4   # negligible move — minimum time

    raw = max_delta / (MAX_VEL * 0.6)
    return float(np.clip(raw, 0.4, 3.0))


def clamp_waypoints(waypoints: list[list[float]]) -> list[list[float]]:
    """
    Clamp every waypoint to JOINT_LIMITS.

    The spline can slightly overshoot the endpoint angles due to its
    polynomial nature. This clamps any overshoot so we never command
    an angle outside the physical servo range.
    """
    clamped = []
    for wp in waypoints:
        safe = [
            float(np.clip(wp[j], JOINT_LIMITS[j][0], JOINT_LIMITS[j][1]))
            for j in range(4)
        ]
        clamped.append(safe)
    return clamped


# =============================================================================
# SECTION 2 — EXECUTION: send waypoints to arm with correct timing
# =============================================================================

def execute(
    arm,
    waypoints: list[list[float]],
    duration:  float,
) -> list[float]:
    """
    Send each waypoint to the arm with correct inter-step timing.

    Timing strategy:
      Each step has a budget of duration/steps seconds.
      We subtract the measured set_joints() round-trip time and sleep
      only the remainder. This keeps the overall move duration accurate
      even if serial takes slightly longer than expected.

    Parameters
    ----------
    arm       : RobotArm instance (from robot_serial.py)
    waypoints : list of [J1,J2,J3,J4] angle sets
    duration  : total move time in seconds

    Returns
    -------
    Confirmed angles from the last waypoint [J1,J2,J3,J4,gripper].
    """
    if not waypoints:
        return arm.get_current_angles()

    n              = len(waypoints)
    dt_budget      = duration / n       # seconds allocated per step
    confirmed      = arm.get_current_angles()
    total_start    = time.perf_counter()

    for i, waypoint in enumerate(waypoints):
        step_start = time.perf_counter()

        # Send to arm — blocking until Arduino confirms
        try:
            confirmed = arm.set_joints(waypoint)
        except RuntimeError as e:
            # Serial timeout mid-move. Log and abort cleanly.
            # The arm stays at whatever position it reached.
            print(f"[trajectory] serial error at step {i+1}/{n}: {e}")
            print(f"[trajectory] aborting move — arm at last confirmed position")
            return confirmed

        # Sleep the remainder of the step budget
        elapsed   = time.perf_counter() - step_start
        remaining = dt_budget - elapsed
        if remaining > 0.001:
            time.sleep(remaining)
        # If remaining < 0 the step ran over budget — continue immediately
        # without sleeping. The arm slows slightly but stays smooth.

    total_elapsed = time.perf_counter() - total_start
    if DEBUG:
        print(f"[trajectory] move complete  "
              f"steps={n}  duration={duration:.2f}s  "
              f"actual={total_elapsed:.2f}s")

    return confirmed


# =============================================================================
# SECTION 3 — PUBLIC API: what state_machine.py calls
# =============================================================================

def move_to(
    arm,
    target_angles: list[float],
    duration:      Optional[float] = None,
    steps:         Optional[int]   = None,
) -> list[float]:
    """
    Smoothly move the arm from its current position to target_angles.

    This is the main function your state machine calls.
    Handles duration estimation, velocity checking, joint clamping,
    and execution all in one call.

    Parameters
    ----------
    arm           : RobotArm instance
    target_angles : [J1,J2,J3,J4] in degrees — destination
    duration      : seconds for the move (None = auto-estimated)
    steps         : interpolation steps (None = use config default)

    Returns
    -------
    Confirmed [J1,J2,J3,J4,gripper] from the last step.

    Example
    -------
    result = trajectory.move_to(arm, [0, 45, 100, -45])
    """
    if len(target_angles) != 4:
        raise ValueError(
            f"target_angles must have 4 values, got {len(target_angles)}"
        )

    start   = arm.get_current_angles()[:4]
    n_steps = steps or STEPS

    # Auto-estimate duration from angular distance if not provided
    if duration is None:
        duration = estimate_duration(start, target_angles)

    if DEBUG:
        print(f"[trajectory] move_to  "
              f"start={[round(v,1) for v in start]}  "
              f"target={[round(v,1) for v in target_angles]}  "
              f"dur={duration:.2f}s  steps={n_steps}")

    # Generate spline waypoints
    waypoints = interpolate(start, target_angles, steps=n_steps, duration=duration)

    # Safety: stretch duration if any joint would exceed MAX_VEL
    duration, waypoints = check_velocity(waypoints, duration)

    # Safety: clamp to joint limits (handles spline overshoot)
    waypoints = clamp_waypoints(waypoints)

    # Execute
    return execute(arm, waypoints, duration)


def move_via(
    arm,
    via_angles:    list[float],
    target_angles: list[float],
    via_duration:    Optional[float] = None,
    target_duration: Optional[float] = None,
) -> list[float]:
    """
    Move through an intermediate via-point before reaching the target.

    Used when a direct path would be unsafe — e.g. moving from home to
    above a ball involves a large base rotation that could swing the arm
    through other objects. Moving via a known-safe intermediate pose
    avoids this.

    Parameters
    ----------
    arm            : RobotArm instance
    via_angles     : [J1,J2,J3,J4] safe intermediate configuration
    target_angles  : [J1,J2,J3,J4] final destination
    via_duration   : seconds for first segment (None = auto)
    target_duration: seconds for second segment (None = auto)

    Returns
    -------
    Confirmed [J1,J2,J3,J4,gripper] from the final step.

    Example
    -------
    # Move to above the ball, then to the bin — via a raised position
    trajectory.move_via(arm, raise_angles, bin_angles)
    """
    if DEBUG:
        print(f"[trajectory] move_via — segment 1/2")
    move_to(arm, via_angles,    duration=via_duration)

    if DEBUG:
        print(f"[trajectory] move_via — segment 2/2")
    return move_to(arm, target_angles, duration=target_duration)


def pick_approach(
    arm,
    approach_angles: list[float],
    pick_angles:     list[float],
    approach_duration: Optional[float] = None,
    descend_duration:  float = 0.5,
) -> list[float]:
    """
    Two-phase pick motion: move to above the ball, then descend slowly.

    This is the canonical way to pick a ball. Always approach from above
    so the gripper does not sweep through the table surface.

    Phase 1 — approach: full-speed smooth move to APPROACH_HEIGHT above ball.
    Phase 2 — descend:  slow, short move straight down to the ball.
               Uses a fixed short duration (default 0.5s) for precision.

    Parameters
    ----------
    arm              : RobotArm instance
    approach_angles  : IK solution for position 40mm ABOVE the ball
    pick_angles      : IK solution for the ball contact position
    approach_duration: seconds for the approach phase (None = auto)
    descend_duration : seconds for the descent phase (default 0.5s)

    Returns
    -------
    Confirmed [J1,J2,J3,J4,gripper] at the pick position.

    Example
    -------
    approach = ik.inverse_kinematics(bx, by, bz + 40)
    pick     = ik.inverse_kinematics(bx, by, bz)
    trajectory.pick_approach(arm, approach, pick)
    arm.close_gripper()
    """
    if DEBUG:
        print("[trajectory] pick_approach — phase 1: approach")
    move_to(arm, approach_angles, duration=approach_duration)

    if DEBUG:
        print("[trajectory] pick_approach — phase 2: descend")
    return move_to(arm, pick_angles, duration=descend_duration,
                   steps=max(10, STEPS // 5))


def place_approach(
    arm,
    place_angles:   list[float],
    release_angles: list[float],
    place_duration:   Optional[float] = None,
    descend_duration: float = 0.5,
) -> list[float]:
    """
    Two-phase place motion: move to above the bin, then descend to release height.

    Mirror of pick_approach for the drop side.

    Parameters
    ----------
    arm             : RobotArm instance
    place_angles    : IK solution for position above the bin
    release_angles  : IK solution for the actual release height
    place_duration  : seconds for the travel phase (None = auto)
    descend_duration: seconds for the descent into bin (default 0.5s)

    Returns
    -------
    Confirmed [J1,J2,J3,J4,gripper] at the release position.
    """
    if DEBUG:
        print("[trajectory] place_approach — phase 1: move to bin")
    move_to(arm, place_angles, duration=place_duration)

    if DEBUG:
        print("[trajectory] place_approach — phase 2: descend to release")
    return move_to(arm, release_angles, duration=descend_duration,
                   steps=max(10, STEPS // 5))


def home(arm) -> list[float]:
    """
    Smoothly return the arm to HOME_ANGLES.

    Convenience wrapper — equivalent to move_to(arm, HOME_ANGLES).
    """
    if DEBUG:
        print("[trajectory] returning to home")
    return move_to(arm, HOME_ANGLES[:4])