# =============================================================================
# test_vision.py
# Standalone test suite for vision.py
#
# HOW TO RUN  (from project root):
#   python tests/test_vision.py
#
# Tests are split into two groups:
#   Group A — pure function tests (no camera, no files, instant)
#   Group B — integration tests with synthetic data (no camera needed)
#
# Group C (live camera tests) are skipped unless --camera flag is given:
#   python tests/test_vision.py --camera
#
# EXIT CODES:
#   0 — all tests passed
#   1 — one or more tests failed
# =============================================================================

import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import cv2

# ── Path setup ────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SRC     = _PROJECT / 'src'

for _p in [_SRC, _HERE, _PROJECT]:
    s = str(_p)
    if _p.exists() and s not in sys.path:
        sys.path.insert(0, s)

import importlib.util
def _load(filename):
    for d in sys.path:
        p = Path(d) / filename
        if p.exists():
            spec = importlib.util.spec_from_file_location(filename[:-3], p)
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
            return m
    raise ImportError(f"Cannot find {filename}. Make sure it is in src/")

vision = _load('vision.py')

USE_CAMERA = '--camera' in sys.argv

# ── Test helpers ──────────────────────────────────────────────────────────────
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

def close(a, b, tol=1.0):
    return abs(a - b) <= tol

# ── Synthetic test data helpers ───────────────────────────────────────────────

def make_synthetic_frame(ball_color_bgr: tuple,
                         cx: int, cy: int,
                         radius: int = 30,
                         size: tuple = (1280, 720)) -> np.ndarray:
    """Create a black frame with a filled circle — synthetic ball."""
    frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    cv2.circle(frame, (cx, cy), radius, ball_color_bgr, -1)
    return frame

def make_perfect_homography() -> np.ndarray:
    """
    Build a homography H that maps known pixel points to known robot mm.
    Camera centred at (640, 360), 1px = 0.5mm, Y-axis inverted (image down = robot -Y).
    Robot origin at image bottom-centre.
    """
    # pixel_point → robot_mm:
    #   cx=640, cy=360  →  x=0,   y=150  (image centre → robot mid-zone)
    #   cx=840, cy=360  →  x=-100, y=150  (100px right → robot -X)
    #   cx=440, cy=360  →  x=+100, y=150  (100px left  → robot +X)
    #   cx=640, cy=160  →  x=0,   y=250  (200px up   → robot +Y more)
    pixel_pts = np.array([
        [640, 360], [840, 360], [440, 360], [640, 160]
    ], dtype=np.float32)
    robot_pts = np.array([
        [0,   150], [-100, 150], [100, 150], [0, 250]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(pixel_pts, robot_pts)
    return H

# =============================================================================
# GROUP A — pure function tests (no camera, no files)
# =============================================================================

def test_orient_frame_no_rotation() -> None:
    print("\n[A1] orient_frame() — no rotation, no flip")
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    # Temporarily set rotation to 0
    original_rot  = vision.CAMERA_ROTATION
    original_flip = vision.CAMERA_FLIP_H
    vision.CAMERA_ROTATION = 0
    vision.CAMERA_FLIP_H   = False
    result = vision.orient_frame(frame)
    vision.CAMERA_ROTATION = original_rot
    vision.CAMERA_FLIP_H   = original_flip
    check("frame shape unchanged with rotation=0",
          result.shape == frame.shape, f"{result.shape}")
    check("frame content unchanged",
          np.array_equal(result, frame))


def test_orient_frame_180() -> None:
    print("\n[A2] orient_frame() — 180 degree rotation")
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[0, 0] = [255, 0, 0]   # top-left = blue pixel

    original_rot  = vision.CAMERA_ROTATION
    vision.CAMERA_ROTATION = 180
    result = vision.orient_frame(frame)
    vision.CAMERA_ROTATION = original_rot

    check("shape preserved after 180 rotation",
          result.shape == frame.shape)
    check("top-left pixel moved to bottom-right after 180 rotation",
          list(result[99, 199]) == [255, 0, 0],
          f"got {list(result[99,199])}")


def test_preprocess_frame() -> None:
    print("\n[A3] preprocess_frame() — blur + HSV conversion")
    frame = make_synthetic_frame((0, 0, 220), cx=400, cy=300, radius=40)
    hsv   = vision.preprocess_frame(frame)

    check("output has same spatial dims",
          hsv.shape[:2] == frame.shape[:2])
    check("output has 3 channels (HSV)",
          hsv.shape[2] == 3,  f"got {hsv.shape[2]}")
    # Red BGR=(0,0,220) → HSV should have H near 0 or 179
    ball_h = int(hsv[300, 400, 0])
    check("red ball H channel near 0 or 179",
          ball_h <= 10 or ball_h >= 169,
          f"H={ball_h}")


def test_build_color_mask_red() -> None:
    print("\n[A4] build_color_mask() — red ball detected")
    # BGR (0,0,220) = pure red → HSV H≈0, S=255, V=220
    frame = make_synthetic_frame((0, 0, 220), cx=640, cy=360, radius=35)
    hsv   = vision.preprocess_frame(frame)
    mask  = vision.build_color_mask(hsv, 'red')

    check("mask is 2D binary",
          len(mask.shape) == 2, f"shape={mask.shape}")
    ball_px = int(mask[360, 640])
    check("ball centroid pixel is white in mask",
          ball_px == 255, f"got {ball_px}")
    background_px = int(mask[10, 10])
    check("background pixel is black in mask",
          background_px == 0, f"got {background_px}")


def test_build_color_mask_blue() -> None:
    print("\n[A5] build_color_mask() — blue ball detected")
    # BGR (200, 50, 0) = pure blue → HSV H≈120
    frame = make_synthetic_frame((200, 50, 0), cx=500, cy=300, radius=30)
    hsv   = vision.preprocess_frame(frame)
    mask  = vision.build_color_mask(hsv, 'blue')

    ball_px = int(mask[300, 500])
    check("blue ball centroid is white in mask",
          ball_px == 255, f"got {ball_px}")


def test_build_color_mask_wrong_color() -> None:
    print("\n[A6] build_color_mask() — wrong color returns clean mask")
    frame = make_synthetic_frame((0, 0, 220), cx=640, cy=360, radius=35)
    hsv   = vision.preprocess_frame(frame)
    mask  = vision.build_color_mask(hsv, 'blue')  # red frame, blue mask

    nonzero = int(np.count_nonzero(mask))
    check("blue mask nearly empty for red frame",
          nonzero < 100, f"nonzero={nonzero}px")


def test_find_best_ball_found() -> None:
    print("\n[A7] find_best_ball() — detects circular blob")
    mask = np.zeros((720, 1280), dtype=np.uint8)
    cv2.circle(mask, (640, 360), 35, 255, -1)   # 35px radius circle

    result = vision.find_best_ball(mask)
    check("ball found",           result is not None)
    check("centroid cx near 640", result is not None and close(result['cx'], 640, 5),
          f"cx={result['cx'] if result else 'N/A'}")
    check("centroid cy near 360", result is not None and close(result['cy'], 360, 5),
          f"cy={result['cy'] if result else 'N/A'}")
    check("circularity >= 0.8",   result is not None and result['circularity'] >= 0.80,
          f"circ={result['circularity']:.3f}" if result else "N/A")


def test_find_best_ball_noise_rejected() -> None:
    print("\n[A8] find_best_ball() — noise blobs rejected")
    mask = np.zeros((720, 1280), dtype=np.uint8)
    # Add 5 tiny noise dots (area < MIN_BALL_AREA_PX)
    for x in range(100, 600, 100):
        cv2.circle(mask, (x, 100), 4, 255, -1)

    result = vision.find_best_ball(mask)
    check("tiny noise blobs rejected", result is None,
          f"got {result}")


def test_find_best_ball_irregular_rejected() -> None:
    print("\n[A9] find_best_ball() — non-circular blob rejected")
    mask = np.zeros((720, 1280), dtype=np.uint8)
    # Draw a thin rectangle — very low circularity
    cv2.rectangle(mask, (100, 100), (500, 120), 255, -1)

    result = vision.find_best_ball(mask)
    check("thin rectangle rejected (low circularity)", result is None,
          f"got {result}")


def test_pixel_to_robot() -> None:
    print("\n[A10] pixel_to_robot() — correct coordinate mapping")
    H = make_perfect_homography()

    # Test: image centre → robot (0, 150)
    x, y, z = vision.pixel_to_robot(640, 360, H)
    check("image centre maps to robot (0, ~150)mm",
          close(x, 0, 3) and close(y, 150, 3),
          f"got ({x:.1f},{y:.1f})")

    # Test: 100px right of centre → robot -X direction
    x2, y2, _ = vision.pixel_to_robot(740, 360, H)
    check("right of centre → negative X",
          x2 < x,
          f"x2={x2:.1f} should be < x={x:.1f}")

    # Test: 100px left of centre → robot +X direction
    x3, y3, _ = vision.pixel_to_robot(540, 360, H)
    check("left of centre → positive X",
          x3 > x,
          f"x3={x3:.1f} should be > x={x:.1f}")

    # Test: Z always = TABLE_HEIGHT_MM
    check("Z always TABLE_HEIGHT_MM",
          close(z, vision.TABLE_HEIGHT_MM, 0.01),
          f"z={z}")


def test_is_in_pick_zone() -> None:
    print("\n[A11] is_in_pick_zone() — boundary checks")
    # Use default pick zone from config
    pz = vision.PICK_ZONE
    cx = (pz['x_min'] + pz['x_max']) / 2
    cy = (pz['y_min'] + pz['y_max']) / 2
    cz = (pz['z_min'] + pz['z_max']) / 2

    check("centre of pick zone is inside",
          vision.is_in_pick_zone(cx, cy, cz))
    check("X below x_min is outside",
          not vision.is_in_pick_zone(pz['x_min']-1, cy, cz))
    check("X above x_max is outside",
          not vision.is_in_pick_zone(pz['x_max']+1, cy, cz))
    check("Y below y_min is outside",
          not vision.is_in_pick_zone(cx, pz['y_min']-1, cz))


def test_calibration_save_load() -> None:
    print("\n[A12] save/load calibration roundtrip")
    K    = np.array([[800,0,640],[0,800,360],[0,0,1]], dtype=np.float64)
    dist = np.array([[0.1,-0.05,0.001,0.001,0.02]], dtype=np.float64)
    rms  = 0.42

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'test_cal.json')
        vision.save_calibration(K, dist, rms, path)
        result = vision.load_calibration(path)

    check("load returns tuple of 2",
          result is not None and len(result) == 2)
    K2, dist2 = result
    check("K matrix roundtrips correctly",
          np.allclose(K, K2, atol=1e-6), f"max diff={np.max(np.abs(K-K2)):.2e}")
    check("dist roundtrips correctly",
          np.allclose(dist, dist2, atol=1e-6))


def test_homography_save_load() -> None:
    print("\n[A13] save/load homography roundtrip")
    H = make_perfect_homography()

    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / 'test_H.npy')
        vision.save_homography(H, path)
        H2 = vision.load_homography(path)

    check("H roundtrips correctly",
          H2 is not None and np.allclose(H, H2, atol=1e-8),
          f"max diff={np.max(np.abs(H-(H2 if H2 is not None else 0))):.2e}")


def test_load_nonexistent_files() -> None:
    print("\n[A14] load missing files returns None gracefully")
    check("missing calibration returns None",
          vision.load_calibration('/nonexistent/path.json') is None)
    check("missing homography returns None",
          vision.load_homography('/nonexistent/path.npy') is None)


# =============================================================================
# GROUP B — integration tests with synthetic data (no live camera)
# =============================================================================

def test_detect_balls_synthetic() -> None:
    print("\n[B1] detect_balls() — synthetic red ball in frame")
    H = make_perfect_homography()

    cam = vision.Camera.__new__(vision.Camera)
    cam._H    = H
    cam._K    = None
    cam._dist = None

    # Synthetic red ball at image centre (640, 360) → robot (~0, ~150)mm
    frame = make_synthetic_frame((0, 0, 220), cx=640, cy=360, radius=35)
    original_debug = vision.DEBUG
    vision.DEBUG = False
    detections = cam.detect_balls(frame=frame)
    vision.DEBUG = original_debug

    check("at least one ball detected",
          len(detections) >= 1, f"got {len(detections)}")
    if detections:
        d = detections[0]
        check("detected color is red",
              d['color'] == 'red', f"got {d['color']}")
        check("centroid cx near 640",
              close(d['cx'], 640, 10), f"cx={d['cx']}")
        check("centroid cy near 360",
              close(d['cy'], 360, 10), f"cy={d['cy']}")
        check("x_mm near 0",
              close(d['x'], 0, 15), f"x={d['x']}")
        check("y_mm near 150",
              close(d['y'], 150, 15), f"y={d['y']}")
        check("detection has required keys",
              all(k in d for k in ['color','x','y','z','cx','cy','area','circularity','in_pick_zone']))


def test_detect_balls_no_ball() -> None:
    print("\n[B2] detect_balls() — empty frame returns empty list")
    H = make_perfect_homography()

    cam = vision.Camera.__new__(vision.Camera)
    cam._H = H; cam._K = None; cam._dist = None

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    original_debug = vision.DEBUG
    vision.DEBUG = False
    detections = cam.detect_balls(frame=frame)
    vision.DEBUG = original_debug

    check("empty frame → empty detections",
          len(detections) == 0, f"got {len(detections)}")


def test_detect_balls_no_homography() -> None:
    print("\n[B3] detect_balls() — raises RuntimeError without homography")
    cam = vision.Camera.__new__(vision.Camera)
    cam._H = None; cam._K = None; cam._dist = None

    frame = make_synthetic_frame((0, 0, 220), cx=640, cy=360, radius=35)
    try:
        cam.detect_balls(frame=frame)
        check("RuntimeError raised without homography", False,
              "no exception raised")
    except RuntimeError:
        check("RuntimeError raised without homography", True)


def test_detect_color_specific() -> None:
    print("\n[B4] detect_color() — returns correct color")
    H = make_perfect_homography()

    cam = vision.Camera.__new__(vision.Camera)
    cam._H = H; cam._K = None; cam._dist = None

    frame = make_synthetic_frame((0, 0, 220), cx=640, cy=360, radius=35)
    original_debug = vision.DEBUG
    vision.DEBUG = False
    red_result  = cam.detect_color('red',  frame=frame)
    blue_result = cam.detect_color('blue', frame=frame)
    vision.DEBUG = original_debug

    check("detect_color('red') finds red ball",  red_result is not None)
    check("detect_color('blue') finds nothing",  blue_result is None)


def test_draw_debug_overlay() -> None:
    print("\n[B5] draw_debug_overlay() — returns frame without modifying original")
    frame = make_synthetic_frame((0, 0, 220), cx=640, cy=360, radius=35)
    original = frame.copy()

    detections = [{
        'color': 'red', 'x': 100.0, 'y': 150.0, 'z': 0.0,
        'cx': 640, 'cy': 360, 'area': 3848.0, 'circularity': 0.91,
        'in_pick_zone': True,
    }]

    result = vision.draw_debug_overlay(frame, detections)

    check("original frame not modified",
          np.array_equal(frame, original))
    check("result has same shape as input",
          result.shape == frame.shape)
    check("result differs from input (overlay was drawn)",
          not np.array_equal(result, frame))


# =============================================================================
# GROUP C — live camera tests (skipped without --camera)
# =============================================================================

def test_camera_open_close() -> None:
    print("\n[C1] Camera.open() and close()")
    cam = vision.Camera()
    try:
        cam.open()
        check("camera opens successfully",   cam.is_open)
        frame = cam.read_frame()
        check("read_frame() returns ndarray", frame is not None and isinstance(frame, np.ndarray))
        check("frame has 3 channels",         frame is not None and frame.shape[2] == 3)
        cam.close()
        check("camera closes cleanly",        not cam.is_open)
    except RuntimeError as e:
        print(f"  SKIP: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    SEP = "=" * 62

    print(SEP)
    print("VISION TEST SUITE")
    print(SEP)

    print("\nGROUP A — Pure function tests (no hardware)")
    print("-" * 62)
    test_orient_frame_no_rotation()
    test_orient_frame_180()
    test_preprocess_frame()
    test_build_color_mask_red()
    test_build_color_mask_blue()
    test_build_color_mask_wrong_color()
    test_find_best_ball_found()
    test_find_best_ball_noise_rejected()
    test_find_best_ball_irregular_rejected()
    test_pixel_to_robot()
    test_is_in_pick_zone()
    test_calibration_save_load()
    test_homography_save_load()
    test_load_nonexistent_files()

    print("\nGROUP B — Integration tests with synthetic data")
    print("-" * 62)
    test_detect_balls_synthetic()
    test_detect_balls_no_ball()
    test_detect_balls_no_homography()
    test_detect_color_specific()
    test_draw_debug_overlay()

    if USE_CAMERA:
        print("\nGROUP C — Live camera tests (--camera flag)")
        print("-" * 62)
        test_camera_open_close()
    else:
        print("\nGROUP C — skipped (run with --camera to test live camera)")

    total = _PASS + _FAIL
    print(f"\n{SEP}")
    print(f"RESULT:  {_PASS}/{total} tests passed")
    if _FAIL == 0:
        print("ALL TESTS PASSED")
        print()
        print("Before first real run:")
        print("  1. python tools/calibrate_camera.py  (15 circle grid photos)")
        print("  2. python tools/set_homography.py    (click 4 table points)")
        print("  3. python tools/tune_hsv.py          (tune colors under demo lighting)")
        print("  4. python tests/test_vision.py --camera  (verify live feed)")
    else:
        print(f"{_FAIL} FAILURE(S)")
    print(SEP)

    sys.exit(0 if _FAIL == 0 else 1)

if __name__ == '__main__':
    main()