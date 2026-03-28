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
    'L1': 170.0,   # base height   : table surface  ->  shoulder joint (vertical)
    'L2': 120.0,   # upper arm     : shoulder joint ->  elbow joint
    'L3': 120.0,   # forearm       : elbow joint    ->  wrist joint
    'L4': 110.0,   # hand          : wrist joint    ->  end-effector tip
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
HOME_ANGLES: list[float] = [90.0, 20.0, 160.0, -80.0]


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

CAMERA_INDEX:  int = 2       # OpenCV device index (0 = first USB camera)
CAMERA_WIDTH:  int = 1280    # capture resolution
CAMERA_HEIGHT: int = 720

# Orientation correction — set after physical verification test.
# Rotate the frame CLOCKWISE by this many degrees after undistort.
# Valid values: 0, 90, 180, 270
# 0 = no rotation (camera cable points toward arm base, image top = robot +Y)
CAMERA_ROTATION: int = 0

# Set True if image left/right is mirrored relative to robot +X / -X.
# Verify: command J1=0 (arm faces +X/its left).
#   If gripper appears on RIGHT side of image → set True to flip.
CAMERA_FLIP_H: bool = False

CALIBRATION_FILE: str = 'calibration/camera_calibration.json'
HOMOGRAPHY_FILE:  str = 'calibration/homography.npy'

# Height of the table surface in the robot base frame (mm).
# 0 means the table is at the same height as the robot base.
TABLE_HEIGHT_MM: float = 0.0

# How high above the ball to move before descending to pick (mm)
APPROACH_HEIGHT_MM: float = 60.0

# Warmup frames to discard on camera open (auto-exposure settling)
CAMERA_WARMUP_FRAMES: int = 10


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
    'orange': [
        (np.array([  5, 150, 100]), np.array([ 20, 255, 255])),
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
    # All bins within guaranteed reach for default link lengths (L2+L3=220mm).
    # Y_near=120mm — bins placed 150-200mm from base, near corners of sheet.
    # UPDATE x/y/z once you measure your actual arm link lengths.
    'red':    {'x': -100.0, 'y': 150.0, 'z': 30.0},
    'blue':   {'x':  100.0, 'y': 150.0, 'z': 30.0},
    'green':  {'x': -100.0, 'y': 200.0, 'z': 30.0},
    'yellow': {'x':  100.0, 'y': 200.0, 'z': 30.0},
    'orange': {'x':    0.0, 'y': 180.0, 'z': 30.0},
}


# =============================================================================
# SECTION 8 — PICK ZONE
# The region on the table where balls are expected to be placed.
# Used by the visualizer and the state machine reachability check.
# =============================================================================

PICK_ZONE: dict[str, float] = {
    # Y_near = 120mm (measured from J1 axle to sheet near edge)
    # Y_far  = 120 + 300 = 420mm (sheet is 300mm deep)
    # X      = ±200mm (sheet is 400mm wide, centred on arm)
    'x_min': -200.0,
    'x_max':  200.0,
    'y_min':  120.0,
    'y_max':  420.0,
    'z_min':    0.0,
    'z_max':   40.0,
}


# =============================================================================
# SECTION 9 — TRAJECTORY GENERATOR
# =============================================================================

TRAJECTORY_DURATION_S:    float = 1.5    # seconds for one arm movement
TRAJECTORY_STEPS:         int   = 25     # number of interpolation steps
TRAJECTORY_MAX_VEL_DEG_S: float = 90.0  # max joint angular velocity (deg/s)


# =============================================================================
# SECTION 10 — GRIPPER
# =============================================================================

GRIPPER_OPEN_DEG:  float = 75.0    # servo angle for fully open gripper  ← measured
GRIPPER_CLOSE_DEG: float = 120.0   # servo angle to grasp ball            ← measured
GRIPPER_WAIT_S:    float = 0.5     # seconds to wait after commanding gripper


# =============================================================================
# SECTION 11 — DEBUG
# =============================================================================

DEBUG: bool = True   # print joint angles, EE coords, IK results to console
                      # set False for clean demo output

# =============================================================================
# SECTION 12 — SERVO MAPPING
# Converts robot-frame angles (from kinematics.py) to physical servo angles.
#
# SERVO_OFFSETS: added to robot angle to get servo angle.
#   Example: J4 robot=0 means wrist level, but your servo reads 90 at level.
#   So offset = +90: servo_angle = robot_angle + 90
#
# SERVO_REVERSED: if True, servo moves opposite to robot-frame convention.
#   Example: J2 robot=0 means arm up, but servo=180 means arm up.
#   So reversed=True: servo_angle = 180 - robot_angle  (then add offset)
#
# HOW TO FIND YOUR VALUES:
#   Run:  python tools/servo_calibration_guide.py
#   It walks you through each joint and prints these values automatically.
#
# YOUR KNOWN VALUES SO FAR:
#   J4 wrist:    robot 0° = servo 90°   → OFFSET = +90
#   J2 shoulder: robot 0° = servo 180°  → REVERSED = True
# =============================================================================

SERVO_OFFSETS: dict[int, float] = {
    1: 0.0,    # J1 base     — measure with calibration guide
    2: 0.0,    # J2 shoulder — reversed handles this (see below)
    3: 0.0,    # J3 elbow    — measure with calibration guide
    4: 90.0,   # J4 wrist    — robot 0° = servo 90° (your measured value)
    5: 0.0,    # gripper     — usually fine as-is
}

SERVO_REVERSED: dict[int, bool] = {
    1: False,  # J1 base
    2: True,   # J2 shoulder — robot 0°=up but servo 180°=up (your case)
    3: False,  # J3 elbow    — measure with calibration guide
    4: False,  # J4 wrist    — offset handles this, not reversal
    5: False,  # gripper
}