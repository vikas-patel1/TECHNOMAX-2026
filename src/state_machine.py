# =============================================================================
# state_machine.py
# Orchestrates the full pick-and-place cycle for TechnoMax 2026.
#
# WHAT THIS FILE DOES:
#   Connects vision → kinematics → trajectory → serial into one loop.
#   Detects coloured balls on the workspace, plans a pick, executes it,
#   then drops the ball into the correct colour bin. Repeats until stopped.
#
# STATE FLOW:
#   IDLE → SCAN → DETECT → PLAN → PICK → CARRY → PLACE → HOME → SCAN → ...
#
#   IDLE   : arm at home, waiting for start signal
#   SCAN   : capture a fresh camera frame, call detect_balls()
#   DETECT : select which ball to pick next (priority order from config)
#   PLAN   : run IK for approach, pick, bin approach, bin drop positions
#   PICK   : move to approach → descend → close gripper
#   CARRY  : lift → rotate to bin side
#   PLACE  : move to bin approach → descend → open gripper
#   HOME   : return to home position, loop back to SCAN
#   ERROR  : something failed — log, recover, return home
#
# HOW TO RUN:
#   Real hardware:   python src/state_machine.py
#   Mock (no arm):   python src/state_machine.py --mock
#   Single cycle:    python src/state_machine.py --once
#   Specific color:  python src/state_machine.py --color green
#
# DEPENDENCIES:
#   kinematics.py, robot_serial.py, trajectory.py, vision.py, config.py
# =============================================================================

from __future__ import annotations

import sys
import time
import argparse
import importlib.util
from pathlib import Path
from enum import Enum, auto
from typing import Optional

# ── Path setup (same pattern as all other modules) ────────────────────────────
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent

for _p in [_HERE, _PROJECT, _HERE / 'src', _PROJECT / 'src']:
    s = str(_p)
    if _p.exists() and s not in sys.path:
        sys.path.insert(0, s)


def _load(filename: str):
    for d in sys.path:
        p = Path(d) / filename
        if p.exists():
            spec = importlib.util.spec_from_file_location(
                filename.replace('.py', ''), p)
            m = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
            spec.loader.exec_module(m)                  # type: ignore[union-attr]
            return m
    raise ImportError(
        f"Cannot find {filename}. "
        f"Make sure all modules are in src/ and sys.path includes it."
    )


# ── Import all modules ────────────────────────────────────────────────────────
kin  = _load('kinematics.py')
rs   = _load('robot_serial.py')
traj = _load('trajectory.py')
vis  = _load('vision.py')

# ── Import config ─────────────────────────────────────────────────────────────
_cfg = None
for _path in [_HERE / 'config.py',
              _HERE / 'config' / 'config.py',
              _PROJECT / 'config.py',
              _PROJECT / 'config' / 'config.py']:
    if _path.exists():
        spec = importlib.util.spec_from_file_location('_cfg', _path)
        _cfg = importlib.util.module_from_spec(spec)    # type: ignore[arg-type]
        spec.loader.exec_module(_cfg)                   # type: ignore[union-attr]
        break

if _cfg is None:
    raise RuntimeError("config.py not found. Cannot start state machine.")

# Pull all config values we need
HOME_ANGLES        = _cfg.HOME_ANGLES
GRIPPER_OPEN       = _cfg.GRIPPER_OPEN_DEG
GRIPPER_CLOSE      = _cfg.GRIPPER_CLOSE_DEG
GRIPPER_WAIT       = _cfg.GRIPPER_WAIT_S
APPROACH_HEIGHT    = _cfg.APPROACH_HEIGHT_MM
TABLE_HEIGHT       = _cfg.TABLE_HEIGHT_MM
BIN_POSITIONS      = _cfg.BIN_POSITIONS
PICK_ZONE          = _cfg.PICK_ZONE
IK_PITCH           = _cfg.IK_DEFAULT_PITCH
DEBUG              = _cfg.DEBUG

# Pick priority — which colour to pick first if multiple balls are detected
# Adjust this order to match your demo preference
PICK_PRIORITY: list[str] = ['green', 'yellow', 'orange', 'red', 'blue']

# How many times to retry detection before giving up and going to ERROR
MAX_SCAN_RETRIES: int = 5

# How many times to retry IK before skipping that ball
MAX_IK_RETRIES: int = 3

# Lift height above the ball before moving to the bin (mm above table)
LIFT_HEIGHT_MM: float = 80.0

# Height above the bin when approaching before descending to drop
BIN_APPROACH_HEIGHT_MM: float = 80.0


# =============================================================================
# STATE ENUM
# =============================================================================

class State(Enum):
    IDLE   = auto()
    SCAN   = auto()
    DETECT = auto()
    PLAN   = auto()
    PICK   = auto()
    CARRY  = auto()
    PLACE  = auto()
    HOME   = auto()
    ERROR  = auto()
    DONE   = auto()


# =============================================================================
# PLAN DATACLASS
# Stores all IK solutions for one pick-and-place cycle.
# Computed once in PLAN state, consumed across PICK/CARRY/PLACE states.
# =============================================================================

class PickPlan:
    """IK solutions and metadata for one complete pick-and-place cycle."""
    __slots__ = [
        'ball',
        'approach_angles',   # above ball (APPROACH_HEIGHT mm above)
        'pick_angles',       # at ball contact height
        'lift_angles',       # directly above pick, lifted up
        'bin_approach_angles', # above bin
        'bin_drop_angles',   # at bin drop height
    ]

    def __init__(self, ball: dict,
                 approach_angles:     list[float],
                 pick_angles:         list[float],
                 lift_angles:         list[float],
                 bin_approach_angles: list[float],
                 bin_drop_angles:     list[float]) -> None:
        self.ball                = ball
        self.approach_angles     = approach_angles
        self.pick_angles         = pick_angles
        self.lift_angles         = lift_angles
        self.bin_approach_angles = bin_approach_angles
        self.bin_drop_angles     = bin_drop_angles


# =============================================================================
# STATE MACHINE CLASS
# =============================================================================

class StateMachine:
    """
    Main orchestrator for the pick-and-place robot arm.

    Usage
    -----
    sm = StateMachine(mock=False)
    sm.setup()
    sm.run()          # runs indefinitely until Ctrl-C or all balls placed
    sm.teardown()
    """

    def __init__(self,
                 mock:         bool = False,
                 once:         bool = False,
                 target_color: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        mock         : if True uses mock Arduino (no hardware needed)
        once         : if True stops after one successful pick-and-place
        target_color : if set only picks balls of this colour
        """
        self._mock         = mock
        self._once         = once
        self._target_color = target_color

        self._arm:   Optional[rs.RobotArm]    = None
        self._cam:   Optional[vis.Camera]      = None
        self._state: State                     = State.IDLE
        self._plan:  Optional[PickPlan]        = None

        self._scan_retries:  int = 0
        self._cycles_done:   int = 0
        self._errors:        int = 0

        self._last_error_msg: str = ''

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Open hardware connections. Call once before run()."""
        print()
        print("=" * 56)
        print("  TechnoMax 2026 — Pick and Place State Machine")
        print("=" * 56)
        print(f"  Mode:   {'MOCK (no hardware)' if self._mock else 'REAL HARDWARE'}")
        print(f"  Filter: {self._target_color or 'all colours'}")
        print(f"  Cycles: {'one then stop' if self._once else 'continuous'}")
        print("=" * 56)
        print()

        # Open arm
        print("[setup] Connecting to arm...")
        self._arm = rs.RobotArm(mock=self._mock)
        self._arm.connect()

        # Open camera
        print("[setup] Opening camera...")
        self._cam = vis.Camera()
        self._cam.open()

        # Move to home
        print("[setup] Moving to home position...")
        traj.home(self._arm)
        self._arm.open_gripper()

        print("[setup] Ready.")
        print()

    def teardown(self) -> None:
        """Close hardware connections. Call after run() exits."""
        print()
        print("[teardown] Returning to home...")
        if self._arm and self._arm.is_connected:
            try:
                traj.home(self._arm)
                self._arm.open_gripper()
            except Exception as e:
                print(f"[teardown] Warning during home: {e}")
            self._arm.disconnect()

        if self._cam:
            self._cam.close()

        print(f"[teardown] Session complete.")
        print(f"  Cycles completed : {self._cycles_done}")
        print(f"  Errors recovered : {self._errors}")

    def run(self) -> None:
        """
        Main loop. Runs until:
          - Ctrl-C (KeyboardInterrupt)
          - State reaches DONE
          - Unrecoverable error
        """
        self._state = State.IDLE

        try:
            while self._state != State.DONE:
                self._step()
        except KeyboardInterrupt:
            print("\n[run] Interrupted by user.")
        finally:
            self.teardown()

    # ── Single step — called every loop iteration ─────────────────────────────

    def _step(self) -> None:
        """Execute one state transition."""

        if self._state == State.IDLE:
            self._state_idle()

        elif self._state == State.SCAN:
            self._state_scan()

        elif self._state == State.DETECT:
            self._state_detect()

        elif self._state == State.PLAN:
            self._state_plan()

        elif self._state == State.PICK:
            self._state_pick()

        elif self._state == State.CARRY:
            self._state_carry()

        elif self._state == State.PLACE:
            self._state_place()

        elif self._state == State.HOME:
            self._state_home()

        elif self._state == State.ERROR:
            self._state_error()

    # =========================================================================
    # STATE HANDLERS
    # =========================================================================

    def _state_idle(self) -> None:
        """
        IDLE — arm is at home, gripper open, waiting.
        Immediately transitions to SCAN on first call.
        """
        self._log("IDLE", "Starting scan cycle")
        self._scan_retries = 0
        self._transition(State.SCAN)

    # ── SCAN ──────────────────────────────────────────────────────────────────

    def _state_scan(self) -> None:
        """
        SCAN — capture a fresh camera frame and detect all balls.
        Retries up to MAX_SCAN_RETRIES times if detection returns nothing.
        """
        self._log("SCAN", f"Detecting balls (attempt {self._scan_retries + 1}/{MAX_SCAN_RETRIES})")

        try:
            detections = self._cam.detect_balls()
        except Exception as e:
            self._fail(f"Camera error during detect_balls(): {e}")
            return

        if not detections:
            self._scan_retries += 1
            if self._scan_retries >= MAX_SCAN_RETRIES:
                self._log("SCAN", "No balls detected after maximum retries.")
                if self._once:
                    self._transition(State.DONE)
                else:
                    # Wait and try again indefinitely
                    self._log("SCAN", "Waiting 2s before next scan...")
                    time.sleep(2.0)
                    self._scan_retries = 0
            else:
                time.sleep(0.3)
            return

        self._scan_retries = 0

        # Filter to pick-zone only and apply colour filter
        in_zone = [
            d for d in detections
            if d['in_pick_zone']
            and (self._target_color is None or d['color'] == self._target_color)
        ]

        if not in_zone:
            self._log("SCAN",
                      f"Detected {len(detections)} ball(s) but none in pick zone "
                      f"or matching colour filter.")
            time.sleep(0.5)
            return

        self._log("SCAN", f"Found {len(in_zone)} ball(s) in pick zone:")
        for d in in_zone:
            self._log("SCAN",
                      f"  {d['color']:8s}  "
                      f"X={d['x']:+6.1f}mm  Y={d['y']:+6.1f}mm  "
                      f"circ={d['circularity']:.2f}")

        # Store for DETECT state
        self._detections_in_zone = in_zone
        self._transition(State.DETECT)

    # ── DETECT ────────────────────────────────────────────────────────────────

    def _state_detect(self) -> None:
        """
        DETECT — choose which ball to pick from the candidates.
        Uses PICK_PRIORITY order. Falls back to largest area (most visible).
        """
        detections = self._detections_in_zone

        # Try priority order first
        selected = None
        for color in PICK_PRIORITY:
            for d in detections:
                if d['color'] == color:
                    selected = d
                    break
            if selected:
                break

        # Fallback: pick the largest (most confidently detected)
        if selected is None:
            selected = max(detections, key=lambda d: d['area'])

        self._log("DETECT",
                  f"Selected: {selected['color']} at "
                  f"({selected['x']:+.1f}, {selected['y']:+.1f})mm")

        self._selected_ball = selected
        self._transition(State.PLAN)

    # ── PLAN ──────────────────────────────────────────────────────────────────

    def _state_plan(self) -> None:
        """
        PLAN — compute all IK solutions needed for this pick-and-place cycle.

        Computes:
          1. approach_angles  — APPROACH_HEIGHT mm above the ball
          2. pick_angles      — at ball contact height (TABLE_HEIGHT)
          3. lift_angles      — LIFT_HEIGHT mm above table, same X/Y as ball
          4. bin_approach_angles — BIN_APPROACH_HEIGHT mm above the bin
          5. bin_drop_angles  — at bin drop height
        """
        ball = self._selected_ball
        color = ball['color']

        bx = ball['x']
        by = ball['y']
        bz = ball['z']   # = TABLE_HEIGHT_MM (ball sits on table)

        # Get bin position
        if color not in BIN_POSITIONS:
            self._fail(f"No bin position defined for color '{color}' in config.py")
            return

        bin_pos = BIN_POSITIONS[color]
        bin_x, bin_y, bin_z = bin_pos['x'], bin_pos['y'], bin_pos['z']

        self._log("PLAN", f"Planning for {color} ball:")
        self._log("PLAN", f"  Ball position: ({bx:+.1f}, {by:+.1f}, {bz:.1f})mm")
        self._log("PLAN", f"  Bin  position: ({bin_x:+.1f}, {bin_y:+.1f}, {bin_z:.1f})mm")

        # ── IK 1: Approach (above ball) ───────────────────────────────────────
        approach_angles = self._ik(
            bx, by, bz + APPROACH_HEIGHT,
            label="approach above ball"
        )
        if approach_angles is None:
            return

        # ── IK 2: Pick (ball contact) ─────────────────────────────────────────
        pick_angles = self._ik(
            bx, by, bz,
            label="ball contact"
        )
        if pick_angles is None:
            return

        # ── IK 3: Lift (straight up from pick position) ───────────────────────
        lift_angles = self._ik(
            bx, by, LIFT_HEIGHT_MM,
            label="lift above pick"
        )
        if lift_angles is None:
            return

        # ── IK 4: Bin approach (above bin) ───────────────────────────────────
        bin_approach_angles = self._ik(
            bin_x, bin_y, bin_z + BIN_APPROACH_HEIGHT_MM,
            label="approach above bin"
        )
        if bin_approach_angles is None:
            return

        # ── IK 5: Bin drop ───────────────────────────────────────────────────
        bin_drop_angles = self._ik(
            bin_x, bin_y, bin_z,
            label="bin drop height"
        )
        if bin_drop_angles is None:
            return

        # All IK solutions found — store plan
        self._plan = PickPlan(
            ball                = ball,
            approach_angles     = approach_angles,
            pick_angles         = pick_angles,
            lift_angles         = lift_angles,
            bin_approach_angles = bin_approach_angles,
            bin_drop_angles     = bin_drop_angles,
        )

        self._log("PLAN", "All IK solutions computed. Ready to pick.")
        self._transition(State.PICK)

    # ── PICK ──────────────────────────────────────────────────────────────────

    def _state_pick(self) -> None:
        """
        PICK — physically execute the pick sequence:
          1. Open gripper
          2. Move to approach position (above ball)
          3. Descend slowly to ball contact
          4. Close gripper
          5. Wait for grip to settle
        """
        plan = self._plan
        color = plan.ball['color']

        self._log("PICK", f"Picking {color} ball...")

        try:
            # Ensure gripper is open before approaching
            self._log("PICK", "Opening gripper...")
            self._arm.open_gripper()

            # Move to approach height above ball (full speed smooth move)
            self._log("PICK", "Moving to approach position...")
            traj.move_to(self._arm, plan.approach_angles)

            # Descend slowly to ball (short duration = slow careful descent)
            self._log("PICK", "Descending to ball...")
            traj.move_to(self._arm, plan.pick_angles,
                         duration=0.6, steps=15)

            # Close gripper and wait for it to settle
            self._log("PICK", "Closing gripper...")
            self._arm.close_gripper()
            time.sleep(GRIPPER_WAIT)

        except RuntimeError as e:
            self._fail(f"Serial error during PICK: {e}")
            return

        self._log("PICK", "Ball gripped.")
        self._transition(State.CARRY)

    # ── CARRY ─────────────────────────────────────────────────────────────────

    def _state_carry(self) -> None:
        """
        CARRY — lift ball and move toward bin.
          1. Lift straight up to safe height
          2. Move to above the bin (full speed)

        The lift step is critical — without it the arm swings the ball
        through the table surface when rotating to the bin.
        """
        plan = self._plan
        color = plan.ball['color']

        self._log("CARRY", f"Carrying {color} ball to bin...")

        try:
            # Lift straight up first
            self._log("CARRY", "Lifting...")
            traj.move_to(self._arm, plan.lift_angles, duration=0.8)

            # Rotate and move to above the bin
            self._log("CARRY", "Moving to bin...")
            traj.move_to(self._arm, plan.bin_approach_angles)

        except RuntimeError as e:
            self._fail(f"Serial error during CARRY: {e}")
            return

        self._transition(State.PLACE)

    # ── PLACE ─────────────────────────────────────────────────────────────────

    def _state_place(self) -> None:
        """
        PLACE — descend into bin and release ball.
          1. Descend slowly into the bin
          2. Open gripper
          3. Wait for ball to fall
          4. Lift back up out of bin
        """
        plan = self._plan
        color = plan.ball['color']

        self._log("PLACE", f"Placing {color} ball in bin...")

        try:
            # Descend into bin slowly
            self._log("PLACE", "Descending into bin...")
            traj.move_to(self._arm, plan.bin_drop_angles,
                         duration=0.6, steps=15)

            # Open gripper — release ball
            self._log("PLACE", "Releasing ball...")
            self._arm.open_gripper()
            time.sleep(GRIPPER_WAIT)

            # Lift out of bin before moving home (avoid knocking the bin)
            self._log("PLACE", "Lifting out of bin...")
            traj.move_to(self._arm, plan.bin_approach_angles,
                         duration=0.6)

        except RuntimeError as e:
            self._fail(f"Serial error during PLACE: {e}")
            return

        self._cycles_done += 1
        self._log("PLACE",
                  f"Ball placed. Cycle {self._cycles_done} complete.")
        self._transition(State.HOME)

    # ── HOME ──────────────────────────────────────────────────────────────────

    def _state_home(self) -> None:
        """
        HOME — return arm to home position.
        Clears plan, resets retry counters, decides whether to continue.
        """
        self._log("HOME", "Returning to home...")

        try:
            traj.home(self._arm)
        except RuntimeError as e:
            # Non-fatal — log and continue
            self._log("HOME", f"Warning during home move: {e}")

        self._plan = None

        if self._once:
            self._log("HOME", "--once flag set. Stopping after one cycle.")
            self._transition(State.DONE)
        else:
            # Small pause so camera can see the workspace clearly
            time.sleep(0.5)
            self._transition(State.SCAN)

    # ── ERROR ─────────────────────────────────────────────────────────────────

    def _state_error(self) -> None:
        """
        ERROR — something went wrong.
        Attempt to recover by returning to home and resuming the cycle.
        After 3 consecutive errors, stop.
        """
        self._errors += 1
        self._log("ERROR",
                  f"Error #{self._errors}: {self._last_error_msg}")

        if self._errors >= 3:
            self._log("ERROR",
                      "3 consecutive errors — stopping for safety.")
            self._transition(State.DONE)
            return

        self._log("ERROR", "Attempting recovery — returning to home...")
        try:
            self._arm.open_gripper()
            traj.home(self._arm)
        except Exception as e:
            self._log("ERROR", f"Recovery also failed: {e}")
            self._transition(State.DONE)
            return

        self._plan = None
        self._log("ERROR", "Recovered. Resuming scan.")
        time.sleep(1.0)
        self._transition(State.SCAN)

    # =========================================================================
    # HELPERS
    # =========================================================================

    def _ik(self, x: float, y: float, z: float,
            label: str = '') -> Optional[list[float]]:
        """
        Run inverse kinematics. Logs result. Returns None and calls _fail()
        if IK cannot find a solution.
        """
        angles = kin.inverse_kinematics(x, y, z,
                                        wrist_pitch_deg=IK_PITCH)
        if angles is None:
            self._fail(
                f"IK failed for {label} at "
                f"({x:+.1f}, {y:+.1f}, {z:.1f})mm. "
                f"Position may be out of reach."
            )
            return None

        if DEBUG:
            self._log("IK",
                      f"{label}: "
                      f"({x:+.1f},{y:+.1f},{z:.1f})mm → "
                      f"[{', '.join(f'{a:.1f}' for a in angles)}]°")
        return angles

    def _fail(self, message: str) -> None:
        """Record an error and transition to ERROR state."""
        self._last_error_msg = message
        print(f"  [FAIL] {message}")
        self._transition(State.ERROR)

    def _transition(self, new_state: State) -> None:
        """Log and execute a state transition."""
        if DEBUG and new_state != self._state:
            print(f"  → {new_state.name}")
        self._state = new_state

    def _log(self, state: str, message: str) -> None:
        """Print a formatted log line."""
        print(f"[{state:8s}] {message}")

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def state(self) -> State:
        return self._state

    @property
    def cycles_done(self) -> int:
        return self._cycles_done

    @property
    def errors(self) -> int:
        return self._errors

    @property
    def last_error(self) -> str:
        return self._last_error_msg

    @property
    def last_detections(self) -> list:
        return getattr(self, '_detections_in_zone', [])

    @property
    def selected_ball(self):
        return getattr(self, '_selected_ball', None)

    @property
    def current_plan(self):
        return self._plan


# =============================================================================
# ENTRY POINT
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='TechnoMax 2026 — Pick and Place State Machine'
    )
    parser.add_argument(
        '--mock', action='store_true',
        help='Use mock Arduino (no hardware needed). Default: False'
    )
    parser.add_argument(
        '--once', action='store_true',
        help='Stop after one successful pick-and-place cycle'
    )
    parser.add_argument(
        '--color', type=str, default=None,
        choices=['red', 'blue', 'green', 'yellow', 'orange'],
        help='Only pick balls of this colour'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sm = StateMachine(
        mock         = args.mock,
        once         = args.once,
        target_color = args.color,
    )

    sm.setup()
    sm.run()  # teardown() is called inside run() via finally block


if __name__ == '__main__':
    main()