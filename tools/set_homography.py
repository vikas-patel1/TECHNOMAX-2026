# =============================================================================
# tools/set_homography.py
# Compute the homography matrix by clicking 4 known reference points.
#
# WORKS WITH ANY OF THESE SETUPS — just edit ROBOT_POINTS below:
#
#   Setup A — Plain white sheet, 4 corner tape marks (RECOMMENDED)
#     - Place small tape crosses or coloured stickers at the 4 corners
#       of your white workspace sheet while it is in its final position
#     - Measure each corner's distance from the arm base with a ruler
#     - Enter those measurements as ROBOT_POINTS below
#
#   Setup B — Checkerboard border sheet
#     - Use the 4 inner corners of the checkerboard border as reference
#     - Count squares × square size to get exact mm positions
#
#   Setup C — Dot grid (original approach)
#     - Use the 4 corner circles of the asymmetric circle grid
#     - Count grid spacings × 25mm to get exact mm positions
#
# HOW TO RUN:
#   1. Mount camera, place white sheet flat in final position
#   2. Edit ROBOT_POINTS below with your measured corner positions
#   3. python tools/set_homography.py
#   4. Click the 4 marks IN ORDER (P1 → P2 → P3 → P4)
#      A coloured dot appears at each click to confirm
#   5. Press ENTER to compute and save. Press R to re-click. Q to quit.
#
# MEASURING YOUR CORNER POSITIONS:
#   - Origin (0, 0) = arm base centre at table surface
#   - +Y = forward, away from arm (top of image when correctly mounted)
#   - +X = arm's LEFT side (left of image when correctly mounted)
#   - Measure straight-line distances with a ruler
#   - Be precise — 1mm measurement error = 1mm position error everywhere
#
# EXAMPLE for a 40×30cm sheet centred in front of the arm:
#   If the sheet's front edge is 80mm from the base and the sheet is
#   centred left-right, the corners would be approximately:
#     P1 (top-left):     X=+200, Y=+380
#     P2 (top-right):    X=-200, Y=+380
#     P3 (bottom-right): X=-200, Y=+ 80
#     P4 (bottom-left):  X=+200, Y=+ 80
#   (Adjust to YOUR actual measurements)
#
# WHY 4 CORNERS BEAT CLUSTERED POINTS:
#   Homography accuracy improves when reference points are spread as
#   far apart as possible. Sheet corners at 400mm apart give much
#   better interpolation accuracy than 4 dots clustered in 100mm.
# =============================================================================

import sys
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
for _p in [_PROJECT / 'src', _HERE, _PROJECT]:
    s = str(_p)
    if _p.exists() and s not in sys.path:
        sys.path.insert(0, s)

import cv2
import numpy as np
import importlib.util


def _load(filename):
    for d in sys.path:
        p = Path(d) / filename
        if p.exists():
            spec = importlib.util.spec_from_file_location(filename[:-3], p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise ImportError(f"Cannot find {filename}")


vision = _load('vision.py')


# =============================================================================
# EDIT THIS SECTION — measure your actual corner positions with a ruler
# =============================================================================
#
# These are the robot-frame (X, Y) positions in mm of your 4 reference marks.
# Origin = arm base centre. +Y = forward. +X = arm's left.
#
# HOW TO MEASURE:
#   1. Place sheet in final position on table
#   2. Mark the arm base centre point on the table with a pen dot
#   3. Use ruler to measure from that dot to each corner mark
#   4. X = left/right distance (+ = toward arm's left, - = toward right)
#   5. Y = forward distance (always positive — corners are in front of arm)
#
# The values below are EXAMPLES — replace with your measured values.
# Corner order: P1=top-left, P2=top-right, P3=bottom-right, P4=bottom-left
# (top = far from arm = large Y, bottom = close to arm = small Y)

ROBOT_POINTS = np.array([
    [+200.0, +420.0],   # P1 — top-left  corner of sheet (far-left)
    [-200.0, +420.0],   # P2 — top-right corner of sheet (far-right)
    [-200.0, +120.0],    # P3 — bottom-right corner       (near-right)
    [+200.0, +120.0],    # P4 — bottom-left corner        (near-left)
], dtype=np.float32)

# Labels shown on screen during clicking
POINT_LABELS = [
    'P1 — top-left  (far-left corner)',
    'P2 — top-right (far-right corner)',
    'P3 — bottom-right (near-right)',
    'P4 — bottom-left  (near-left)',
]

# Dot colours on screen: orange, blue, green, red
CLICK_COLORS = [(0, 165, 255), (255, 80, 0), (0, 200, 0), (0, 0, 220)]

# =============================================================================
# END OF EDIT SECTION
# =============================================================================


clicked_pixels: list = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_pixels) < 4:
        clicked_pixels.append([x, y])
        lbl = POINT_LABELS[len(clicked_pixels) - 1]
        print(f"  Clicked: pixel ({x:4d}, {y:4d})  →  {lbl}")


def _verify_and_save(pixel_pts: np.ndarray) -> bool:
    """Compute H, verify back-projection, save if good. Returns True on success."""
    H, mask = cv2.findHomography(pixel_pts, ROBOT_POINTS)
    if H is None:
        print("ERROR: findHomography failed.")
        print("  Check that your 4 points are not collinear (not all on one line).")
        return False

    print()
    print("Verification — pixel → robot mm (should match your measured values):")
    print(f"  {'Point':<6}  {'Clicked px':>12}  {'Computed mm':>14}  {'Expected mm':>14}  {'Error':>8}")
    max_err = 0.0
    for i, (px, py) in enumerate(pixel_pts):
        pt  = np.array([[[float(px), float(py)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H)
        cx, cy = out[0][0]
        ex, ey = ROBOT_POINTS[i]
        err = np.sqrt((cx - ex)**2 + (cy - ey)**2)
        max_err = max(max_err, err)
        status = "OK" if err < 2.0 else "WARN"
        print(f"  P{i+1:<5}  ({px:5.0f},{py:5.0f})px  "
              f"({cx:+7.1f},{cy:+7.1f})mm  "
              f"({ex:+7.1f},{ey:+7.1f})mm  "
              f"{err:6.2f}mm {status}")

    print()
    if max_err > 5.0:
        print(f"WARNING: max error {max_err:.1f}mm is high.")
        print("  This usually means ROBOT_POINTS values don't match physical measurements.")
        print("  Re-measure with a ruler and update ROBOT_POINTS in this file.")
    elif max_err > 2.0:
        print(f"Max error {max_err:.1f}mm — acceptable but consider re-measuring.")
    else:
        print(f"Max error {max_err:.2f}mm — excellent accuracy.")

    vision.save_homography(H, vision.HOMOGRAPHY_FILE)
    print()
    print("Homography saved. Vision system ready.")
    print("Next step: python tools/tune_hsv.py")
    return True


def _draw_guide_overlay(display: np.ndarray, n_clicked: int) -> np.ndarray:
    """Draw a guide showing which point to click next."""
    h, w = display.shape[:2]

    # Semi-transparent instruction bar at top
    overlay = display.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (20, 20, 20), -1)
    display = cv2.addWeighted(overlay, 0.6, display, 0.4, 0)

    if n_clicked < 4:
        msg  = f"Click {POINT_LABELS[n_clicked]}  |  R=reset  Q=quit"
        prog = f"Step {n_clicked + 1} of 4"
    else:
        msg  = "All 4 points clicked — press ENTER to save  |  R=reset"
        prog = "4 / 4"

    cv2.putText(display, msg, (12, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(display, prog, (w - 90, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1, cv2.LINE_AA)

    # Draw already-clicked points
    for i, (px, py) in enumerate(clicked_pixels):
        col = CLICK_COLORS[i]
        cv2.circle(display, (px, py), 10, col, 2)
        cv2.circle(display, (px, py), 3,  col, -1)
        lbl = f"P{i+1}  ({ROBOT_POINTS[i][0]:+.0f},{ROBOT_POINTS[i][1]:+.0f})mm"
        # Offset label so it does not overlap the dot
        tx = px + 14 if px < w - 180 else px - 190
        ty = py - 12 if py > 30 else py + 22
        cv2.putText(display, lbl, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # Draw lines connecting clicked points in order
    for i in range(len(clicked_pixels) - 1):
        cv2.line(display,
                 tuple(clicked_pixels[i]),
                 tuple(clicked_pixels[i + 1]),
                 (200, 200, 200), 1, cv2.LINE_AA)
    if len(clicked_pixels) == 4:
        cv2.line(display,
                 tuple(clicked_pixels[3]),
                 tuple(clicked_pixels[0]),
                 (200, 200, 200), 1, cv2.LINE_AA)

    return display


def main() -> None:
    global clicked_pixels

    print("=" * 60)
    print("HOMOGRAPHY SETUP TOOL")
    print("=" * 60)
    print()
    print("Reference points configured:")
    for i, (lbl, pt) in enumerate(zip(POINT_LABELS, ROBOT_POINTS)):
        print(f"  P{i+1}: {lbl:<40s}  X={pt[0]:+.0f}mm  Y={pt[1]:+.0f}mm")
    print()
    print("If these don't match your physical measurements,")
    print("edit ROBOT_POINTS in this file before continuing.")
    print()

    cam = vision.Camera()
    cam.open()

    cv2.namedWindow("Set Homography — click 4 corner marks in order")
    cv2.setMouseCallback(
        "Set Homography — click 4 corner marks in order", on_mouse
    )

    while True:
        frame = cam.read_frame()
        if frame is None:
            continue

        display = _draw_guide_overlay(frame.copy(), len(clicked_pixels))
        cv2.imshow("Set Homography — click 4 corner marks in order", display)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            print("Quit without saving.")
            break

        if key in (13,) and len(clicked_pixels) == 4:          # ENTER
            pixel_pts = np.array(clicked_pixels, dtype=np.float32)
            _verify_and_save(pixel_pts)
            break

        if key in (ord('r'), ord('R')):
            clicked_pixels = []
            print("Reset — click P1 again.")

    cam.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()