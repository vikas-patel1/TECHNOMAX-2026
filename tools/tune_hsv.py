# =============================================================================
# tools/tune_hsv.py
# Live HSV range tuner for ball color detection.
#
# HOW TO USE:
#   1. Run:  python tools/tune_hsv.py
#   2. Select a color with keys:  R=red  B=blue  G=green  Y=yellow
#   3. Hold the actual ball in front of the camera.
#   4. Drag the sliders until ONLY the ball is white in the mask window.
#      Everything else should be black.
#   5. Press S to print the current values — copy them to config.py.
#   6. Press Q or ESC to quit.
#
# TIPS:
#   - Tune under your actual demo lighting — not bright lab then dim corridor
#   - Saturation minimum (S_min) is your most powerful noise filter.
#     Raise it until table surface and shadows disappear.
#   - For red: you need TWO ranges because red wraps around 0° in HSV.
#     Tune H_min/H_max for 0–10 first, then tune the 170–180 range.
#   - After tuning, test by moving the ball to all corners of the pick zone.
#     It should stay white in the mask everywhere.
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

COLORS = ['red', 'blue', 'green', 'yellow']
current_color_idx = 0

# Default starting ranges (from config.py)
DEFAULTS = {
    'red':    {'h_min':   0, 'h_max':  10, 's_min': 120, 's_max': 255, 'v_min': 70, 'v_max': 255},
    'blue':   {'h_min': 100, 'h_max': 130, 's_min': 120, 's_max': 255, 'v_min': 70, 'v_max': 255},
    'green':  {'h_min':  40, 'h_max':  80, 's_min':  80, 's_max': 255, 'v_min': 70, 'v_max': 255},
    'yellow': {'h_min':  20, 'h_max':  35, 's_min': 120, 's_max': 255, 'v_min': 70, 'v_max': 255},
}

WIN_CTRL = "HSV Controls"
WIN_ORIG = "Original"
WIN_MASK = "Mask"

def create_trackbars(color: str) -> None:
    cv2.destroyWindow(WIN_CTRL)
    cv2.namedWindow(WIN_CTRL)
    d = DEFAULTS[color]
    cv2.createTrackbar("H min", WIN_CTRL, d['h_min'], 179, lambda x: None)
    cv2.createTrackbar("H max", WIN_CTRL, d['h_max'], 179, lambda x: None)
    cv2.createTrackbar("S min", WIN_CTRL, d['s_min'], 255, lambda x: None)
    cv2.createTrackbar("S max", WIN_CTRL, d['s_max'], 255, lambda x: None)
    cv2.createTrackbar("V min", WIN_CTRL, d['v_min'], 255, lambda x: None)
    cv2.createTrackbar("V max", WIN_CTRL, d['v_max'], 255, lambda x: None)

def get_trackbar_values() -> dict:
    return {
        'h_min': cv2.getTrackbarPos("H min", WIN_CTRL),
        'h_max': cv2.getTrackbarPos("H max", WIN_CTRL),
        's_min': cv2.getTrackbarPos("S min", WIN_CTRL),
        's_max': cv2.getTrackbarPos("S max", WIN_CTRL),
        'v_min': cv2.getTrackbarPos("V min", WIN_CTRL),
        'v_max': cv2.getTrackbarPos("V max", WIN_CTRL),
    }

def print_values(color: str, v: dict) -> None:
    print(f"\n# config.py values for '{color}':")
    if color == 'red':
        print(f"    '{color}': [")
        print(f"        (np.array([  0, {v['s_min']:3d}, {v['v_min']:3d}]), "
              f"np.array([ {v['h_max']:2d}, 255, 255])),  # low-hue range")
        print(f"        (np.array([{179-v['h_max']:3d}, {v['s_min']:3d}, {v['v_min']:3d}]), "
              f"np.array([180, 255, 255])),  # high-hue range (wraps)")
        print(f"    ],")
    else:
        print(f"    '{color}': [")
        print(f"        (np.array([{v['h_min']:3d}, {v['s_min']:3d}, {v['v_min']:3d}]), "
              f"np.array([{v['h_max']:3d}, {v['s_max']:3d}, {v['v_max']:3d}])),")
        print(f"    ],")

def main():
    global current_color_idx

    cap = cv2.VideoCapture(vision.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {vision.CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  vision.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, vision.CAMERA_HEIGHT)
    for _ in range(10):
        cap.read()

    cv2.namedWindow(WIN_ORIG)
    cv2.namedWindow(WIN_MASK)
    create_trackbars(COLORS[current_color_idx])

    # Load calibration for undistort (optional but improves accuracy)
    K, dist = None, None
    result = vision.load_calibration(vision.CALIBRATION_FILE)
    if result:
        K, dist = result

    print("HSV Tuner  |  R=red  B=blue  G=green  Y=yellow  S=save values  Q=quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        if K is not None:
            frame = vision.undistort_frame(frame, K, dist)
        frame = vision.orient_frame(frame)

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7,7), 0), cv2.COLOR_BGR2HSV)

        v = get_trackbar_values()
        lo = np.array([v['h_min'], v['s_min'], v['v_min']])
        hi = np.array([v['h_max'], v['s_max'], v['v_max']])

        mask = cv2.inRange(hsv, lo, hi)
        if COLORS[current_color_idx] == 'red':
            # Also show high-hue range for red
            lo2 = np.array([179 - v['h_max'], v['s_min'], v['v_min']])
            hi2 = np.array([180, 255, 255])
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo2, hi2))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Draw contours on original
        display = frame.copy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > vision.MIN_BALL_AREA_PX:
                cv2.drawContours(display, [cnt], -1, (0, 255, 0), 2)

        color = COLORS[current_color_idx]
        cv2.putText(display,
                    f"Tuning: {color}  |  R/B/G/Y to switch  S to print  Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow(WIN_ORIG, display)
        cv2.imshow(WIN_MASK, mask)

        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q'), ord('Q')):
            break
        elif key == ord('r'):
            current_color_idx = 0
            create_trackbars(COLORS[current_color_idx])
        elif key == ord('b'):
            current_color_idx = 1
            create_trackbars(COLORS[current_color_idx])
        elif key == ord('g'):
            current_color_idx = 2
            create_trackbars(COLORS[current_color_idx])
        elif key == ord('y'):
            current_color_idx = 3
            create_trackbars(COLORS[current_color_idx])
        elif key in (ord('s'), ord('S')):
            print_values(COLORS[current_color_idx], get_trackbar_values())

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()