# =============================================================================
# config.py
# Single source of truth for ALL tunable parameters.
#
# Every number your team might ever need to change lives here.
# No magic numbers anywhere else in the codebase.
#
# Usage (from any module in the project):
#   from config.config import LINK_LENGTHS, JOINT_LIMITS, ...   # if in config/
#   from config       import LINK_LENGTHS, JOINT_LIMITS, ...   # if next to src/
# =============================================================================


# =============================================================================
# SECTION 1 — ARM GEOMETRY (millimetres)
# Measure from joint-centre to joint-centre with a ruler on your physical arm.
# =============================================================================

LINK_LENGTHS: dict[str, float] = {
    'L1': 100.0,   # base height   : table surface  ->  shoulder joint (vertical)
    'L2': 120.0,   # upper arm     : shoulder joint ->  elbow joint
    'L3': 100.0,   # forearm       : elbow joint    ->  wrist joint
    'L4':  60.0,   # hand          : wrist joint    ->  end-effector tip
}


# =============================================================================
# SECTION 2 — JOINT LIMITS (degrees)
#
# Set to the actual mechanical stop positions of your servos.
#
# WHY these ranges:
#   J1  0–180   : forward hemisphere. Extend if base servo allows more.
#   J2 -90–180  : MUST allow negative so the arm can aim below shoulder height.
#                 (L1=100mm means any target Z < 100mm needs J2 < 0.)
#   J3 -150–150 : symmetric. Positive = elbow-up, negative = elbow-down.
#   J4 -180–180 : full pitch range for any approach angle.
#
# HOW TO MAP TO YOUR SERVO:
#   Set the servo mechanical zero so that:
#     J2 =   0° → arm pointing straight forward (horizontal)
#     J2 =  90° → arm pointing straight up
#     J2 = -90° → arm pointing straight down
# =============================================================================

JOINT_LIMITS: list[tuple[float, float]] = [
    (  0.0,  180.0),   # J1 — base rotation
    (  0.0,  170.0),   # J2 — shoulder: 0=up, 90=horizontal forward, 170=almost-down
    (  0.0,  170.0),   # J3 — elbow:    0=straight, 90=bent forward toward target
    (-90.0,   90.0),   # J4 — wrist:    negative=tip-down (picking), positive=tip-up
]

# Safe resting position sent on startup and after each task cycle
HOME_ANGLES: list[float] = [90.0, 70.0, 60.0, -45.0]


# =============================================================================
# SECTION 3 — IK SOLVER PARAMETERS
# =============================================================================

IK_TOLERANCE_MM: float  = 2.5    # max acceptable end-effector position error (mm)
IK_MAX_ITER:     int    = 500    # scipy optimizer iteration cap
IK_DEFAULT_PITCH: float = -45.0  # default wrist approach angle (degrees)
                                  #   0   = horizontal
                                  #  -45  = angled down  ← good general pick angle
                                  #  -90  = straight down from above


# =============================================================================
# SECTION 4 — SERIAL COMMUNICATION
# =============================================================================

# Port to use. Set to None for auto-detection (searches for Arduino vendor ID).
# Windows:  'COM3', 'COM4', ...
# Linux:    '/dev/ttyUSB0', '/dev/ttyACM0', ...
SERIAL_PORT:    str | None = None
SERIAL_BAUD:    int        = 115200
SERIAL_TIMEOUT: float      = 2.0    # seconds to wait for a reply packet
SERIAL_RETRIES: int        = 3      # how many times to resend on NAK / timeout


# =============================================================================
# SECTION 5 — CAMERA
# =============================================================================

CAMERA_INDEX:  int = 0       # OpenCV device index (0 = first USB camera)
CAMERA_WIDTH:  int = 1280    # capture resolution
CAMERA_HEIGHT: int = 720

CALIBRATION_FILE: str = 'calibration/camera_calibration.json'
HOMOGRAPHY_FILE:  str = 'calibration/homography.npy'

# Height of the table surface in the robot base frame (mm).
# 0 means the table is at the same height as the robot base.
TABLE_HEIGHT_MM: float = 0.0

# How high above the ball to move before descending to pick (mm)
APPROACH_HEIGHT_MM: float = 40.0


# =============================================================================
# SECTION 6 — HSV COLOR RANGES
# Tune with:  python tools/tune_hsv.py
#
# Format: color_name -> list of (lower_hsv, upper_hsv) tuples
# Red wraps around 0° in HSV so it needs two ranges.
# H: 0-179,  S: 0-255,  V: 0-255  (OpenCV convention)
# =============================================================================

import numpy as np   # noqa: E402 — needed here for np.array

HSV_RANGES: dict[str, list[tuple]] = {
    'red': [
        (np.array([  0, 120,  70]), np.array([ 10, 255, 255])),
        (np.array([170, 120,  70]), np.array([180, 255, 255])),
    ],
    'blue': [
        (np.array([100, 120,  70]), np.array([130, 255, 255])),
    ],
    'green': [
        (np.array([ 40,  80,  70]), np.array([ 80, 255, 255])),
    ],
    'yellow': [
        (np.array([ 20, 120,  70]), np.array([ 35, 255, 255])),
    ],
}

MIN_BALL_AREA_PX:   int   = 500    # ignore blobs smaller than this (noise)
MAX_BALL_AREA_PX:   int   = 50000  # ignore blobs larger than this (whole table)
MIN_CIRCULARITY:    float = 0.60   # 0.0 = any shape, 1.0 = perfect circle


# =============================================================================
# SECTION 7 — BIN DROP POSITIONS (mm, robot base frame)
# Where the arm places each sorted ball.
# Measure and set these once your table layout is finalised.
# =============================================================================

BIN_POSITIONS: dict[str, dict[str, float]] = {
    'red':    {'x': -150.0, 'y': 180.0, 'z': 50.0},
    'blue':   {'x':  150.0, 'y': 180.0, 'z': 50.0},
    'green':  {'x': -150.0, 'y': 300.0, 'z': 50.0},
    'yellow': {'x':  150.0, 'y': 300.0, 'z': 50.0},
}


# =============================================================================
# SECTION 8 — PICK ZONE
# The region on the table where balls are expected to be placed.
# Used by the visualizer and the state machine reachability check.
# =============================================================================

PICK_ZONE: dict[str, float] = {
    'x_min':  50.0,
    'x_max': 220.0,
    'y_min': -180.0,
    'y_max':  180.0,
    'z_min':   0.0,
    'z_max':  40.0,
}


# =============================================================================
# SECTION 9 — TRAJECTORY GENERATOR
# =============================================================================

TRAJECTORY_DURATION_S:    float = 1.5    # seconds for one arm movement
TRAJECTORY_STEPS:         int   = 50     # number of interpolation steps
TRAJECTORY_MAX_VEL_DEG_S: float = 90.0  # max joint angular velocity (deg/s)


# =============================================================================
# SECTION 10 — GRIPPER
# =============================================================================

GRIPPER_OPEN_DEG:  float = 10.0   # servo angle for fully open gripper
GRIPPER_CLOSE_DEG: float = 75.0   # servo angle to grasp a ~40mm ball
GRIPPER_WAIT_S:    float = 0.4    # seconds to wait after commanding gripper


# =============================================================================
# SECTION 11 — DEBUG
# =============================================================================

DEBUG: bool = True   # print joint angles, EE coords, IK results to console
                      # set False for clean demo output