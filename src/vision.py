# =============================================================================
# vision.py
# Camera pipeline for ball detection and pixel → robot-frame coordinate mapping.
#
# RESPONSIBILITIES:
#   - Open and warm up the USB camera
#   - Undistort each frame using saved calibration data
#   - Apply orientation correction (rotation / flip) to align with robot frame
#   - Detect coloured balls via HSV masking and contour filtering
#   - Map ball pixel centroids to robot-frame (X, Y, Z) mm via homography
#   - Provide calibration helpers (load / save calibration and homography)
#
# DOES NOT:
#   - Know anything about joint angles or IK
#   - Import robot_serial or trajectory
#   - Make any arm movement decisions
#
# PUBLIC API (what state_machine.py calls):
#   cam   = Camera()
#   cam.open()
#   balls = cam.detect_balls()           # → list of BallDetection dicts
#   cam.close()
#
#   Each BallDetection:
#   {
#     'color' : 'red'|'blue'|'green'|'yellow',
#     'x'     : float,   # mm in robot base frame
#     'y'     : float,   # mm in robot base frame
#     'z'     : float,   # mm (= TABLE_HEIGHT_MM, constant)
#     'cx'    : int,     # pixel column in undistorted frame (for debug overlay)
#     'cy'    : int,     # pixel row  in undistorted frame (for debug overlay)
#     'area'  : float,   # contour area in px² (larger = more confident)
#   }
#
# TOOLS (run standalone to set up the vision system):
#   python tools/calibrate_camera.py   — compute K and dist from circle grid photos
#   python tools/set_homography.py     — click 4 table points to compute H
#   python tools/tune_hsv.py           — live slider tool to set HSV ranges
#
# DEPENDENCIES:
#   pip install opencv-python numpy
# =============================================================================

from __future__ import annotations

import json
import time
import importlib.util
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# =============================================================================
# CONFIG LOADING — same 4-path search as all other modules
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
    CAMERA_INDEX         = _cfg.CAMERA_INDEX
    CAMERA_WIDTH         = _cfg.CAMERA_WIDTH
    CAMERA_HEIGHT        = _cfg.CAMERA_HEIGHT
    CAMERA_ROTATION      = _cfg.CAMERA_ROTATION
    CAMERA_FLIP_H        = _cfg.CAMERA_FLIP_H
    CAMERA_WARMUP_FRAMES = _cfg.CAMERA_WARMUP_FRAMES
    CALIBRATION_FILE     = _cfg.CALIBRATION_FILE
    HOMOGRAPHY_FILE      = _cfg.HOMOGRAPHY_FILE
    TABLE_HEIGHT_MM      = _cfg.TABLE_HEIGHT_MM
    HSV_RANGES           = _cfg.HSV_RANGES
    MIN_BALL_AREA_PX     = _cfg.MIN_BALL_AREA_PX
    MAX_BALL_AREA_PX     = _cfg.MAX_BALL_AREA_PX
    MIN_CIRCULARITY      = _cfg.MIN_CIRCULARITY
    PICK_ZONE            = _cfg.PICK_ZONE
    DEBUG                = _cfg.DEBUG
    print("[vision] loaded config")
else:
    print("[vision] config not found — using built-in defaults")
    import numpy as _np
    CAMERA_INDEX         = 0
    CAMERA_WIDTH         = 1280
    CAMERA_HEIGHT        = 720
    CAMERA_ROTATION      = 0
    CAMERA_FLIP_H        = False
    CAMERA_WARMUP_FRAMES = 10
    CALIBRATION_FILE     = 'calibration/camera_calibration.json'
    HOMOGRAPHY_FILE      = 'calibration/homography.npy'
    TABLE_HEIGHT_MM      = 0.0
    HSV_RANGES           = {
        'red':    [(_np.array([0,120,70]),  _np.array([10,255,255])),
                   (_np.array([170,120,70]),_np.array([180,255,255]))],
        'blue':   [(_np.array([100,120,70]),_np.array([130,255,255]))],
        'green':  [(_np.array([40,80,70]),  _np.array([80,255,255]))],
        'yellow': [(_np.array([20,120,70]), _np.array([35,255,255]))],
    }
    MIN_BALL_AREA_PX     = 500
    MAX_BALL_AREA_PX     = 50000
    MIN_CIRCULARITY      = 0.60
    PICK_ZONE            = {'x_min':50,'x_max':220,'y_min':-180,'y_max':180,'z_min':0,'z_max':40}
    DEBUG                = True


# =============================================================================
# SECTION 1 — CALIBRATION DATA: load / save helpers
# =============================================================================

def load_calibration(path: str = CALIBRATION_FILE
                     ) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Load camera matrix K and distortion coefficients from JSON.

    Returns
    -------
    (K, dist)  or  None if file does not exist.

    K    shape (3,3)  — camera intrinsic matrix
    dist shape (1,5)  — distortion coefficients [k1,k2,p1,p2,k3]
    """
    p = Path(path)
    if not p.exists():
        if DEBUG:
            print(f"[vision] calibration file not found: {path}")
        return None
    with open(p) as f:
        data = json.load(f)
    K    = np.array(data['camera_matrix'],       dtype=np.float64)
    dist = np.array(data['distortion_coeffs'],   dtype=np.float64)
    if DEBUG:
        print(f"[vision] loaded calibration from {path}")
    return K, dist


def save_calibration(K: np.ndarray, dist: np.ndarray,
                     rms: float,
                     path: str = CALIBRATION_FILE) -> None:
    """Save calibration results to JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        'camera_matrix':     K.tolist(),
        'distortion_coeffs': dist.tolist(),
        'rms_reprojection_error': rms,
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"[vision] calibration saved → {path}  (RMS={rms:.4f}px)")


def load_homography(path: str = HOMOGRAPHY_FILE) -> Optional[np.ndarray]:
    """
    Load homography matrix H from .npy file.

    Returns
    -------
    H  shape (3,3)  or  None if file does not exist.
    """
    p = Path(path)
    if not p.exists():
        if DEBUG:
            print(f"[vision] homography file not found: {path}")
        return None
    H = np.load(str(p))
    if DEBUG:
        print(f"[vision] loaded homography from {path}")
    return H


def save_homography(H: np.ndarray,
                    path: str = HOMOGRAPHY_FILE) -> None:
    """Save homography matrix to .npy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, H)
    print(f"[vision] homography saved → {path}")


# =============================================================================
# SECTION 2 — FRAME PROCESSING: pure functions (no camera, fully testable)
# =============================================================================

def orient_frame(frame: np.ndarray) -> np.ndarray:
    """
    Apply rotation and horizontal flip to align camera frame with robot frame.

    Robot frame convention:
      Image top    → robot +Y (forward, away from arm)
      Image left   → robot +X (arm's left side)

    Controlled by CAMERA_ROTATION (0/90/180/270) and CAMERA_FLIP_H in config.py.
    Set these after running the physical verification test:
      1. Command arm to J1=90, J2=45, J3=90, J4=-45
      2. Attach marker to gripper tip
      3. Marker should appear at TOP-CENTRE of image
      4. Command J1=0 → marker should move to LEFT side
    """
    if CAMERA_ROTATION == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif CAMERA_ROTATION == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif CAMERA_ROTATION == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if CAMERA_FLIP_H:
        frame = cv2.flip(frame, 1)
    return frame


def undistort_frame(frame: np.ndarray,
                    K: np.ndarray,
                    dist: np.ndarray) -> np.ndarray:
    """
    Remove lens distortion from a raw camera frame.

    Uses the optimal new camera matrix so no black borders appear.
    Must be called on every frame before any detection.
    """
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undistorted = cv2.undistort(frame, K, dist, newCameraMatrix=new_K)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        # Crop to valid region and resize back to original resolution
        cropped = undistorted[y:y+rh, x:x+rw]
        undistorted = cv2.resize(cropped, (w, h),
                                 interpolation=cv2.INTER_LINEAR)
    return undistorted


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Blur and convert to HSV ready for color masking.

    GaussianBlur kernel (7,7):
      - Removes single-pixel noise that creates spurious small contours
      - Small enough not to blur ball edges significantly
      - Must be odd × odd

    BGR→HSV conversion:
      - Hue channel encodes colour independently of brightness
      - A red ball in bright light and dim light have same Hue, different BGR
      - Critical for consistent detection across lighting changes
    """
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    return hsv


def build_color_mask(hsv: np.ndarray, color: str) -> np.ndarray:
    """
    Build a binary mask for one color using HSV thresholding.

    Red is special — it wraps around 0° in the HSV hue wheel so it
    needs two ranges (near 0° and near 180°) combined with bitwise OR.

    Morphological operations:
      OPEN  (erode then dilate) — kills small noise blobs
      CLOSE (dilate then erode) — fills holes from specular reflections
    """
    ranges = HSV_RANGES.get(color, [])
    if not ranges:
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def find_best_ball(mask: np.ndarray) -> Optional[dict]:
    """
    Find the best ball candidate in a binary mask.

    Filtering steps:
      1. Area: reject blobs outside [MIN_BALL_AREA_PX, MAX_BALL_AREA_PX]
         — too small = noise, too large = wrong detection
      2. Circularity = 4π·area / perimeter²
         — balls score 0.7–0.9, random blobs score 0.1–0.4
      3. Best candidate = largest area that passes both filters
         — handles partial occlusion (larger remnant = more ball visible)

    Returns
    -------
    {'cx': int, 'cy': int, 'area': float, 'circularity': float}
    or None if no valid ball found.
    """
    contours, _ = cv2.findContours(mask,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_BALL_AREA_PX or area > MAX_BALL_AREA_PX:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 1e-6:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < MIN_CIRCULARITY:
            continue

        if area > best_area:
            M  = cv2.moments(cnt)
            if M['m00'] < 1e-6:
                continue
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            best = {
                'cx':          cx,
                'cy':          cy,
                'area':        area,
                'circularity': circularity,
            }
            best_area = area

    return best


def pixel_to_robot(cx: int, cy: int,
                   H: np.ndarray) -> tuple[float, float, float]:
    """
    Convert a pixel centroid to robot-frame (X, Y, Z) mm using homography.

    The homography H was computed from:
      cv2.findHomography(pixel_points, robot_points)
    where robot_points are in mm in the robot base frame.

    Z is always TABLE_HEIGHT_MM because the camera cannot measure depth.
    All balls sit on the table surface so Z is constant and known.

    Parameters
    ----------
    cx, cy : pixel coordinates in the undistorted, oriented frame
    H      : 3×3 homography matrix from load_homography()

    Returns
    -------
    (X_mm, Y_mm, Z_mm)
    """
    pt  = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pt, H)
    x_mm = float(out[0][0][0])
    y_mm = float(out[0][0][1])
    return x_mm, y_mm, TABLE_HEIGHT_MM


def is_in_pick_zone(x_mm: float, y_mm: float, z_mm: float) -> bool:
    """Return True if (x,y,z) is inside the configured pick zone."""
    return (PICK_ZONE['x_min'] <= x_mm <= PICK_ZONE['x_max'] and
            PICK_ZONE['y_min'] <= y_mm <= PICK_ZONE['y_max'] and
            PICK_ZONE['z_min'] <= z_mm <= PICK_ZONE['z_max'])


def draw_debug_overlay(frame: np.ndarray,
                       detections: list[dict]) -> np.ndarray:
    """
    Draw circles and labels on the frame for debugging.

    Does not modify the original frame — returns a copy.
    Safe to call even if detections is empty.
    """
    COLOR_BGR = {
        'red':    (0,   0,   220),
        'blue':   (220, 50,  50),
        'green':  (30,  180, 30),
        'yellow': (0,   200, 200),
    }
    out = frame.copy()
    for d in detections:
        color_bgr = COLOR_BGR.get(d['color'], (200, 200, 200))
        cx, cy    = d['cx'], d['cy']
        r         = int(np.sqrt(d['area'] / np.pi))

        cv2.circle(out, (cx, cy), r, color_bgr, 2)
        cv2.circle(out, (cx, cy), 4, color_bgr, -1)

        label = (f"{d['color']}  "
                 f"({d['x']:.0f},{d['y']:.0f})mm  "
                 f"circ={d['circularity']:.2f}")
        cv2.putText(out, label, (cx - r, cy - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1,
                    cv2.LINE_AA)
    return out


# =============================================================================
# SECTION 3 — CAMERA CLASS: hardware wrapper
# =============================================================================

class Camera:
    """
    USB camera manager with calibration, orientation, and ball detection.

    Usage
    -----
    cam = Camera()
    cam.open()                    # opens device, warms up, loads calib + H
    balls = cam.detect_balls()    # returns list of BallDetection dicts
    frame = cam.read_frame()      # returns undistorted, oriented frame
    cam.close()

    Without calibration files the camera still works but positions
    will be less accurate at the edges of the pick zone.
    Without homography the detect_balls() call raises RuntimeError.
    """

    def __init__(self,
                 camera_index:      int = CAMERA_INDEX,
                 calibration_file:  str = CALIBRATION_FILE,
                 homography_file:   str = HOMOGRAPHY_FILE) -> None:
        self._index     = camera_index
        self._cal_path  = calibration_file
        self._hom_path  = homography_file
        self._cap:      Optional[cv2.VideoCapture] = None
        self._K:        Optional[np.ndarray] = None
        self._dist:     Optional[np.ndarray] = None
        self._H:        Optional[np.ndarray] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """
        Open the camera, set resolution, warm up, load calibration and H.

        Raises RuntimeError if the camera device cannot be opened.
        Calibration and homography are loaded if files exist — if they
        don't exist the camera opens anyway (useful during initial setup).
        """
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera index {self._index}. "
                f"Check USB connection and CAMERA_INDEX in config.py."
            )

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Warmup — discard first N frames while auto-exposure settles
        print(f"[Camera] warming up ({CAMERA_WARMUP_FRAMES} frames)...")
        for _ in range(CAMERA_WARMUP_FRAMES):
            self._cap.read()
        print("[Camera] ready")

        # Load calibration (optional)
        result = load_calibration(self._cal_path)
        if result is not None:
            self._K, self._dist = result
        else:
            print("[Camera] WARNING: no calibration — positions less accurate at edges")

        # Load homography (required for detect_balls)
        self._H = load_homography(self._hom_path)
        if self._H is None:
            print("[Camera] WARNING: no homography — run tools/set_homography.py first")

    def close(self) -> None:
        """Release the camera device."""
        if self._cap and self._cap.isOpened():
            self._cap.release()
        print("[Camera] closed")

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ── Frame capture ─────────────────────────────────────────────────────────

    def read_raw(self) -> Optional[np.ndarray]:
        """Read one raw BGR frame. Returns None on failure."""
        if not self._cap:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read one frame, apply undistort and orientation correction.

        Returns the processed BGR frame ready for detection, or None
        if the camera read failed.
        This is the frame state_machine.py should use for debug display.
        """
        frame = self.read_raw()
        if frame is None:
            return None

        # Undistort if calibration is loaded
        if self._K is not None and self._dist is not None:
            frame = undistort_frame(frame, self._K, self._dist)

        # Orient to match robot frame
        frame = orient_frame(frame)
        return frame

    # ── Detection ─────────────────────────────────────────────────────────────

    def detect_balls(self,
                     frame: Optional[np.ndarray] = None
                     ) -> list[dict]:
        """
        Detect all coloured balls in one camera frame.

        Parameters
        ----------
        frame : optional pre-captured frame (e.g. for testing with static images).
                If None, reads a fresh frame from the camera.

        Returns
        -------
        List of detections, one per detected ball, sorted by area descending
        (most confidently detected ball first).

        Each detection dict:
          'color'       : str    — 'red'|'blue'|'green'|'yellow'
          'x'           : float  — mm in robot base frame
          'y'           : float  — mm in robot base frame
          'z'           : float  — mm (TABLE_HEIGHT_MM)
          'cx'          : int    — pixel column (undistorted frame)
          'cy'          : int    — pixel row    (undistorted frame)
          'area'        : float  — contour area in px²
          'circularity' : float  — 0–1, higher = more circular
          'in_pick_zone': bool   — True if inside configured pick zone

        Raises
        ------
        RuntimeError if homography is not loaded.
        """
        if self._H is None:
            raise RuntimeError(
                "Homography not loaded. "
                "Run tools/set_homography.py first, then restart."
            )

        # Get processed frame
        if frame is None:
            frame = self.read_frame()
        if frame is None:
            if DEBUG:
                print("[Camera] detect_balls: frame read failed")
            return []

        # Preprocess
        hsv = preprocess_frame(frame)

        detections: list[dict] = []

        for color in HSV_RANGES:
            mask = build_color_mask(hsv, color)
            ball = find_best_ball(mask)
            if ball is None:
                continue

            x_mm, y_mm, z_mm = pixel_to_robot(
                ball['cx'], ball['cy'], self._H
            )

            detection: dict = {
                'color':       color,
                'x':           round(x_mm, 1),
                'y':           round(y_mm, 1),
                'z':           round(z_mm, 1),
                'cx':          ball['cx'],
                'cy':          ball['cy'],
                'area':        round(ball['area'], 1),
                'circularity': round(ball['circularity'], 3),
                'in_pick_zone': is_in_pick_zone(x_mm, y_mm, z_mm),
            }
            detections.append(detection)

            if DEBUG:
                zone = "IN zone" if detection['in_pick_zone'] else "outside zone"
                print(f"[Camera] {color:6s}  "
                      f"({x_mm:6.1f},{y_mm:6.1f})mm  "
                      f"circ={ball['circularity']:.2f}  "
                      f"area={ball['area']:.0f}px²  {zone}")

        # Sort by area descending — most confident detection first
        detections.sort(key=lambda d: d['area'], reverse=True)
        return detections

    def detect_color(self, color: str,
                     frame: Optional[np.ndarray] = None
                     ) -> Optional[dict]:
        """
        Detect a single specific color ball.

        Convenience wrapper used by the state machine when it knows
        which colour it is looking for.

        Returns the detection dict or None if not found.
        """
        balls = self.detect_balls(frame)
        for b in balls:
            if b['color'] == color:
                return b
        return None

    def capture_frame_for_calibration(self) -> Optional[np.ndarray]:
        """
        Read a raw frame suitable for calibration (no undistort applied).
        Used by calibrate_camera.py tool.
        """
        return self.read_raw()

    # ── Calibration setters (used by tools) ──────────────────────────────────

    def set_calibration(self, K: np.ndarray, dist: np.ndarray) -> None:
        """Set calibration data (called by calibrate_camera.py after solving)."""
        self._K    = K
        self._dist = dist

    def set_homography(self, H: np.ndarray) -> None:
        """Set homography matrix (called by set_homography.py after clicking)."""
        self._H = H