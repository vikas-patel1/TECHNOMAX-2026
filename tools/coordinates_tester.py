# =============================================================================
# tools/coordinate_tester.py
# Click anywhere on camera feed → get robot (X, Y) coordinates
# =============================================================================
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent

sys.path.insert(0, str(_PROJECT / "src"))
sys.path.insert(0, str(_PROJECT))

import cv2
import numpy as np
from vision import Camera, pixel_to_robot, draw_debug_overlay

clicked_point = None


def on_mouse(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


def main():
    global clicked_point

    cam = Camera()
    cam.open()

    if cam._H is None:
        print("ERROR: Homography not loaded. Run set_homography.py first.")
        return

    cv2.namedWindow("Coordinate Tester")
    cv2.setMouseCallback("Coordinate Tester", on_mouse)

    while True:
        frame = cam.read_frame()
        if frame is None:
            continue

        display = frame.copy()

        if clicked_point is not None:
            cx, cy = clicked_point

            # Convert to robot coordinates
            x_mm, y_mm, z_mm = pixel_to_robot(cx, cy, cam._H)

            # Draw point
            cv2.circle(display, (cx, cy), 6, (0, 255, 0), -1)

            # Show coordinates
            text = f"X={x_mm:.1f} mm  Y={y_mm:.1f} mm"
            cv2.putText(display, text, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2, cv2.LINE_AA)

            print(f"Clicked Pixel: ({cx}, {cy}) → Robot: ({x_mm:.1f}, {y_mm:.1f}) mm")

            # Reset after showing once (optional)
            clicked_point = None

        cv2.imshow("Coordinate Tester", display)
        key = cv2.waitKey(30) & 0xFF

        if key in (27, ord('q')):
            break

    cam.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
