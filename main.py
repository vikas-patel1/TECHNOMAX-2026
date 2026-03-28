# =============================================================================
# main.py
# Entry point for TechnoMax 2026 pick-and-place robot arm.
#
# HOW TO RUN:
#   Real hardware, full dashboard:   python src/main.py
#   Mock (no arm):                   python src/main.py --mock
#   One cycle only:                  python src/main.py --once
#   Specific colour:                 python src/main.py --color green
#   No dashboard (headless):         python src/main.py --no-dashboard
#
# DASHBOARD WINDOWS:
#   [1] Main dashboard  — state, FPS, ball counts, arm status, alerts
#   [2] Camera feed     — live frame with ball detection overlays
#   [3] HSV mask        — binary mask for the selected active colour
#
# KEYBOARD SHORTCUTS (click any dashboard window first):
#   SPACE  — pause / resume the state machine
#   Q / ESC — quit cleanly (arm returns home)
#   1-5    — cycle HSV mask display between colours
#   D      — toggle DEBUG output in terminal
# =============================================================================

from __future__ import annotations

import sys
import time
import threading
import argparse
import importlib.util
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
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
    raise ImportError(f"Cannot find {filename}")


sm_mod  = _load('state_machine.py')
vis_mod = _load('vision.py')

StateMachine = sm_mod.StateMachine
State        = sm_mod.State

# ── Config ────────────────────────────────────────────────────────────────────
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
    raise RuntimeError("config.py not found.")

HSV_RANGES   = _cfg.HSV_RANGES
PICK_ZONE    = _cfg.PICK_ZONE
DEBUG        = _cfg.DEBUG


# =============================================================================
# DASHBOARD CONSTANTS
# =============================================================================

# Dashboard main panel size
DASH_W = 640
DASH_H = 520

# Camera feed display size
CAM_W  = 640
CAM_H  = 480

# HSV mask display size
MASK_W = 320
MASK_H = 240

# Colour palette (BGR)
CLR = {
    'bg':      (18,  18,  28),
    'panel':   (28,  28,  42),
    'border':  (60,  60,  90),
    'white':   (240, 240, 240),
    'dim':     (120, 120, 140),
    'green':   (80,  200, 80),
    'red':     (80,  80,  220),
    'amber':   (40,  160, 220),
    'teal':    (180, 200, 80),
    'blue':    (220, 140, 40),
    'purple':  (200, 80,  180),
    'ok':      (60,  200, 60),
    'warn':    (40,  180, 220),
    'err':     (60,  60,  220),
}

# Colour -> BGR for ball labels
BALL_BGR = {
    'red':    (60,  60,  220),
    'blue':   (220, 120, 40),
    'green':  (40,  200, 40),
    'yellow': (40,  220, 220),
    'orange': (40,  140, 255),
}

# State -> colour mapping
STATE_CLR = {
    State.IDLE:   CLR['dim'],
    State.SCAN:   CLR['teal'],
    State.DETECT: CLR['teal'],
    State.PLAN:   CLR['amber'],
    State.PICK:   CLR['green'],
    State.CARRY:  CLR['green'],
    State.PLACE:  CLR['blue'],
    State.HOME:   CLR['purple'],
    State.ERROR:  CLR['err'],
    State.DONE:   CLR['dim'],
}

COLOUR_NAMES = list(HSV_RANGES.keys())   # for HSV mask cycling


# =============================================================================
# DASHBOARD RENDERER
# =============================================================================

class Dashboard:
    """
    Renders a three-window OpenCV debug dashboard.

    Window 1 — Main panel: state, FPS, ball counts, arm status, alerts
    Window 2 — Camera feed: annotated live frame
    Window 3 — HSV mask: binary mask for currently selected colour
    """

    WIN_DASH = "TechnoMax 2026 — Dashboard"
    WIN_CAM  = "Camera Feed"
    WIN_MASK = "HSV Mask"

    def __init__(self) -> None:
        self._fps_times: list[float] = []
        self._fps:       float       = 0.0
        self._mask_idx:  int         = 0      # which colour's mask to show
        self._paused:    bool        = False
        self._alerts:    list[str]   = []
        self._alert_ts:  list[float] = []
        self._frame_count: int       = 0

        # Last captured raw data for rendering
        self._last_frame:      Optional[np.ndarray] = None
        self._last_detections: list[dict]            = []
        self._last_mask:       Optional[np.ndarray] = None

        cv2.namedWindow(self.WIN_DASH, cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.WIN_CAM,  cv2.WINDOW_NORMAL)
        cv2.namedWindow(self.WIN_MASK, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(self.WIN_DASH, DASH_W, DASH_H)
        cv2.resizeWindow(self.WIN_CAM,  CAM_W,  CAM_H)
        cv2.resizeWindow(self.WIN_MASK, MASK_W, MASK_H)

        # Position windows side by side
        cv2.moveWindow(self.WIN_DASH, 0,          0)
        cv2.moveWindow(self.WIN_CAM,  DASH_W + 10, 0)
        cv2.moveWindow(self.WIN_MASK, DASH_W + 10, CAM_H + 40)

    # ── Public update call (called every frame from main loop) ────────────────

    def update(self,
               sm:          StateMachine,
               frame:       Optional[np.ndarray],
               detections:  list[dict]) -> int:
        """
        Render all windows and return the pressed key code.
        Returns -1 if no key pressed.
        Call cv2.waitKey(1) is handled inside here.
        """
        self._update_fps()
        self._last_frame      = frame
        self._last_detections = detections

        # Expire old alerts (older than 4 seconds)
        now = time.time()
        pairs = [(a, t) for a, t in zip(self._alerts, self._alert_ts)
                 if now - t < 4.0]
        self._alerts   = [p[0] for p in pairs]
        self._alert_ts = [p[1] for p in pairs]

        # Add alerts from state machine
        if sm.state == State.ERROR and sm.last_error:
            self._add_alert(sm.last_error)

        # Check for out-of-reach detections
        for d in detections:
            if not d['in_pick_zone']:
                self._add_alert(
                    f"{d['color']} ball outside pick zone "
                    f"({d['x']:.0f},{d['y']:.0f})mm"
                )

        # Build HSV mask for selected colour
        if frame is not None:
            hsv  = vis_mod.preprocess_frame(frame)
            color_key = COLOUR_NAMES[self._mask_idx % len(COLOUR_NAMES)]
            self._last_mask = vis_mod.build_color_mask(hsv, color_key)
        else:
            self._last_mask = None

        # Render all three windows
        self._render_dashboard(sm)
        self._render_camera(detections)
        self._render_mask()

        key = cv2.waitKey(1) & 0xFF
        return key

    def add_alert(self, message: str) -> None:
        """Add an alert from external code (e.g. main loop)."""
        self._add_alert(message)

    def is_paused(self) -> bool:
        return self._paused

    def toggle_pause(self) -> None:
        self._paused = not self._paused

    def next_mask_colour(self) -> None:
        self._mask_idx = (self._mask_idx + 1) % max(1, len(COLOUR_NAMES))

    def set_mask_colour(self, idx: int) -> None:
        self._mask_idx = idx % max(1, len(COLOUR_NAMES))

    def close(self) -> None:
        cv2.destroyAllWindows()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _add_alert(self, msg: str) -> None:
        # Deduplicate — don't add same message twice in 2s
        now = time.time()
        for existing, ts in zip(self._alerts, self._alert_ts):
            if existing == msg and now - ts < 2.0:
                return
        self._alerts.append(msg)
        self._alert_ts.append(now)
        # Keep max 5 alerts
        if len(self._alerts) > 5:
            self._alerts   = self._alerts[-5:]
            self._alert_ts = self._alert_ts[-5:]

    def _update_fps(self) -> None:
        now = time.time()
        self._fps_times.append(now)
        # Keep only last 30 frames for FPS calc
        self._fps_times = [t for t in self._fps_times if now - t < 2.0]
        if len(self._fps_times) > 1:
            self._fps = len(self._fps_times) / (
                self._fps_times[-1] - self._fps_times[0] + 1e-6)
        self._frame_count += 1

    # ── Window 1: Main dashboard ──────────────────────────────────────────────

    def _render_dashboard(self, sm: StateMachine) -> None:
        canvas = np.full((DASH_H, DASH_W, 3), CLR['bg'], dtype=np.uint8)

        def rect(x, y, w, h, color=CLR['panel'], radius=6):
            cv2.rectangle(canvas, (x, y), (x+w, y+h), color, -1)
            cv2.rectangle(canvas, (x, y), (x+w, y+h), CLR['border'], 1)

        def txt(text, x, y, color=CLR['white'], scale=0.5, bold=False):
            thickness = 2 if bold else 1
            cv2.putText(canvas, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                        thickness, cv2.LINE_AA)

        # ── Title bar ─────────────────────────────────────────────────────────
        cv2.rectangle(canvas, (0, 0), (DASH_W, 36), (35, 25, 55), -1)
        txt("TechnoMax 2026  |  Pick & Place Robot Arm",
            10, 24, CLR['white'], 0.55, bold=True)
        pause_lbl = "  [PAUSED]" if self._paused else ""
        txt(f"FPS: {self._fps:.1f}{pause_lbl}",
            DASH_W - 160, 24,
            CLR['amber'] if self._paused else CLR['teal'], 0.5)

        # ── State panel ───────────────────────────────────────────────────────
        rect(10, 44, DASH_W - 20, 68)
        txt("CURRENT STATE", 20, 64, CLR['dim'], 0.42)
        state_name = sm.state.name
        state_col  = STATE_CLR.get(sm.state, CLR['white'])
        txt(state_name, 20, 100, state_col, 0.9, bold=True)

        # State description
        descriptions = {
            State.IDLE:   "Waiting to start",
            State.SCAN:   "Capturing frame and detecting balls",
            State.DETECT: "Selecting which ball to pick",
            State.PLAN:   "Computing inverse kinematics",
            State.PICK:   "Moving to ball and gripping",
            State.CARRY:  "Lifting and rotating to bin",
            State.PLACE:  "Descending into bin and releasing",
            State.HOME:   "Returning arm to home position",
            State.ERROR:  "Error — attempting recovery",
            State.DONE:   "Session complete",
        }
        txt(descriptions.get(sm.state, ""), 180, 100, CLR['dim'], 0.42)

        # ── Stats row ─────────────────────────────────────────────────────────
        rect(10, 122, 140, 56)
        rect(158, 122, 140, 56)
        rect(306, 122, 140, 56)
        rect(454, 122, 170, 56)

        txt("CYCLES",    20,  140, CLR['dim'],   0.38)
        txt(str(sm.cycles_done), 20, 166, CLR['green'],  0.8, bold=True)

        txt("ERRORS",    168, 140, CLR['dim'],   0.38)
        err_col = CLR['err'] if sm.errors > 0 else CLR['ok']
        txt(str(sm.errors), 168, 166, err_col, 0.8, bold=True)

        txt("FRAME",     316, 140, CLR['dim'],   0.38)
        txt(str(self._frame_count), 316, 166, CLR['white'], 0.8)

        txt("MASK COLOUR", 464, 140, CLR['dim'], 0.38)
        active_color = COLOUR_NAMES[self._mask_idx % len(COLOUR_NAMES)]
        txt(active_color.upper(), 464, 166,
            BALL_BGR.get(active_color, CLR['white']), 0.7, bold=True)

        # ── Ball detections panel ─────────────────────────────────────────────
        rect(10, 188, DASH_W - 20, 140)
        txt("BALLS IN FRAME", 20, 208, CLR['dim'], 0.42)

        detections = self._last_detections
        if not detections:
            txt("No balls detected", 20, 238, CLR['dim'], 0.5)
        else:
            # Count by colour
            counts: dict[str, int] = {}
            for d in detections:
                counts[d['color']] = counts.get(d['color'], 0) + 1

            # Draw one row per detected colour
            row_y = 228
            for d in detections:
                col_bgr = BALL_BGR.get(d['color'], CLR['white'])
                in_zone = d['in_pick_zone']

                # Colour dot
                cv2.circle(canvas, (28, row_y - 5), 7, col_bgr, -1)

                # Ball info
                zone_str = "IN ZONE" if in_zone else "OUT OF REACH"
                zone_col = CLR['ok'] if in_zone else CLR['err']
                info = (f"{d['color'].upper():<8s}  "
                        f"X={d['x']:+6.1f}mm  Y={d['y']:+6.1f}mm  "
                        f"circ={d['circularity']:.2f}")
                txt(info, 44, row_y, col_bgr, 0.42)
                txt(zone_str, DASH_W - 110, row_y, zone_col, 0.38, bold=True)
                row_y += 22
                if row_y > 318:
                    txt("...", 44, row_y, CLR['dim'], 0.4)
                    break

        # ── Pick zone info ────────────────────────────────────────────────────
        rect(10, 338, DASH_W - 20, 44)
        txt("PICK ZONE", 20, 356, CLR['dim'], 0.38)
        pz = PICK_ZONE
        txt(f"X: {pz['x_min']:.0f} to {pz['x_max']:.0f} mm  "
            f"Y: {pz['y_min']:.0f} to {pz['y_max']:.0f} mm  "
            f"Z: {pz['z_min']:.0f} to {pz['z_max']:.0f} mm",
            20, 372, CLR['dim'], 0.40)

        # ── Selected ball ─────────────────────────────────────────────────────
        sel = sm.selected_ball
        if sel and sm.state in (State.PLAN, State.PICK, State.CARRY, State.PLACE):
            rect(10, 392, DASH_W - 20, 38)
            txt("TARGET BALL", 20, 408, CLR['dim'], 0.38)
            col_bgr = BALL_BGR.get(sel['color'], CLR['white'])
            txt(f"{sel['color'].upper()}  "
                f"({sel['x']:+.1f}, {sel['y']:+.1f})mm",
                20, 422, col_bgr, 0.48, bold=True)

        # ── Alerts ────────────────────────────────────────────────────────────
        alert_y_start = 440
        if self._alerts:
            rect(10, alert_y_start - 8, DASH_W - 20,
                 min(len(self._alerts), 4) * 22 + 16, CLR['panel'])
            for i, alert in enumerate(self._alerts[-4:]):
                age = time.time() - self._alert_ts[-(4 - i)]
                alpha = max(0.3, 1.0 - age / 4.0)
                a_col = tuple(int(c * alpha) for c in CLR['warn'])
                txt(f"! {alert}", 20,
                    alert_y_start + i * 22,
                    a_col, 0.38)

        # ── Keyboard hint bar ─────────────────────────────────────────────────
        cv2.rectangle(canvas,
                      (0, DASH_H - 22), (DASH_W, DASH_H),
                      (25, 20, 40), -1)
        txt("SPACE=pause  Q=quit  1-5=mask colour  D=debug",
            10, DASH_H - 7, CLR['dim'], 0.35)

        cv2.imshow(self.WIN_DASH, canvas)

    # ── Window 2: Camera feed ─────────────────────────────────────────────────

    def _render_camera(self, detections: list[dict]) -> None:
        frame = self._last_frame
        if frame is None:
            blank = np.full((CAM_H, CAM_W, 3), CLR['bg'], dtype=np.uint8)
            cv2.putText(blank, "No camera frame",
                        (CAM_W // 2 - 80, CAM_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, CLR['dim'], 1)
            cv2.imshow(self.WIN_CAM, blank)
            return

        display = vis_mod.draw_debug_overlay(frame, detections)

        # Draw pick zone boundary on the frame
        # The pick zone is in robot mm — we cannot draw it directly without
        # the inverse homography. Instead draw a text label on the frame.
        h, w = display.shape[:2]
        cv2.putText(display,
                    f"FPS: {self._fps:.1f}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    CLR['teal'], 1, cv2.LINE_AA)

        # Paused overlay
        if self._paused:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)
            cv2.putText(display, "PAUSED",
                        (w // 2 - 60, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                        CLR['amber'], 3, cv2.LINE_AA)

        display_resized = cv2.resize(display, (CAM_W, CAM_H))
        cv2.imshow(self.WIN_CAM, display_resized)

    # ── Window 3: HSV mask ────────────────────────────────────────────────────

    def _render_mask(self) -> None:
        color_key = COLOUR_NAMES[self._mask_idx % len(COLOUR_NAMES)]
        mask = self._last_mask

        if mask is None:
            blank = np.full((MASK_H, MASK_W, 3), CLR['bg'], dtype=np.uint8)
            cv2.putText(blank, "No mask",
                        (60, MASK_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, CLR['dim'], 1)
            cv2.imshow(self.WIN_MASK, blank)
            return

        # Convert binary mask to colour display (white on coloured bg)
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Tint white pixels with the ball colour
        tint = BALL_BGR.get(color_key, (200, 200, 200))
        colored = mask_rgb.copy()
        colored[mask > 0] = tint

        # Resize to display size
        colored = cv2.resize(colored, (MASK_W, MASK_H))

        # Label
        cv2.putText(colored,
                    f"HSV MASK: {color_key.upper()}",
                    (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    CLR['white'], 1, cv2.LINE_AA)

        # Pixel count
        nonzero = int(np.count_nonzero(mask))
        cv2.putText(colored,
                    f"white px: {nonzero}",
                    (6, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    CLR['dim'], 1, cv2.LINE_AA)

        cv2.imshow(self.WIN_MASK, colored)


# =============================================================================
# MAIN LOOP
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='TechnoMax 2026 — Main Entry Point'
    )
    parser.add_argument('--mock',         action='store_true',
                        help='Use mock Arduino (no hardware)')
    parser.add_argument('--once',         action='store_true',
                        help='Stop after one pick-and-place cycle')
    parser.add_argument('--color',        type=str, default=None,
                        choices=['red','blue','green','yellow','orange'],
                        help='Only pick this colour')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Run headless — no OpenCV windows')
    return parser.parse_args()


def run_with_dashboard(args: argparse.Namespace) -> None:
    """Main loop with OpenCV dashboard running in the main thread."""

    sm   = StateMachine(mock=args.mock,
                        once=args.once,
                        target_color=args.color)
    dash = Dashboard()

    print()
    print("=" * 56)
    print("  TechnoMax 2026 — Starting with Dashboard")
    print("  SPACE=pause  Q=quit  1-5=HSV mask colour")
    print("=" * 56)
    print()

    sm.setup()

    sm._state = sm_mod.State.IDLE
    running = True
    last_frame      = None
    last_detections: list = []

    try:
        while running and sm.state != State.DONE:

            # ── Capture latest camera frame for dashboard ──────────────────
            if sm._cam and sm._cam.is_open:
                frame = sm._cam.read_frame()
                if frame is not None:
                    last_frame = frame
                # Get last detections from state machine (non-blocking)
                last_detections = sm.last_detections

            # ── Tick state machine one step (unless paused) ────────────────
            if not dash.is_paused():
                sm._step()

            # ── Render dashboard ───────────────────────────────────────────
            key = dash.update(sm, last_frame, last_detections)

            # ── Handle keyboard ────────────────────────────────────────────
            if key in (ord('q'), ord('Q'), 27):     # Q or ESC
                print("\n[main] Quit key pressed.")
                running = False

            elif key == ord(' '):                   # SPACE — pause/resume
                dash.toggle_pause()
                state = "PAUSED" if dash.is_paused() else "RESUMED"
                print(f"[main] {state}")

            elif key in (ord('1'), ord('2'), ord('3'),
                         ord('4'), ord('5')):        # 1-5 cycle mask colour
                idx = key - ord('1')
                dash.set_mask_colour(idx)
                if idx < len(COLOUR_NAMES):
                    print(f"[main] HSV mask: {COLOUR_NAMES[idx]}")

            elif key in (ord('d'), ord('D')):        # D — toggle debug
                import vision as v_module
                v_module.DEBUG = not v_module.DEBUG
                print(f"[main] Debug: {v_module.DEBUG}")

    except KeyboardInterrupt:
        print("\n[main] Ctrl-C received.")

    finally:
        print("[main] Shutting down...")
        dash.close()
        sm.teardown()
        print("[main] Done.")


def run_headless(args: argparse.Namespace) -> None:
    """Run without dashboard — just the state machine loop."""
    sm = StateMachine(mock=args.mock,
                      once=args.once,
                      target_color=args.color)
    sm.setup()
    sm.run()   # teardown() called inside run()


def main() -> None:
    args = parse_args()

    if args.no_dashboard:
        run_headless(args)
    else:
        run_with_dashboard(args)


if __name__ == '__main__':
    main()