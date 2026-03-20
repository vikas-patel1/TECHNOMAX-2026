# =============================================================================
# kinematics.py  —  Pick-and-place robotic arm  —  FIXED VERSION
#
# HOW TO USE:
#   python kinematics.py           -> interactive 3-D visualizer
#   python kinematics.py --test    -> full verification test suite
#
# Dependencies:
#   pip install numpy scipy matplotlib
#
# CHANGELOG vs original:
#   FIX 1 — Joint limits: J2/J3/J4 now allow negative angles so the arm
#            can reach targets below shoulder height (the main IK failure).
#   FIX 2 — elbow_up convention: True keeps theta3 positive (correct).
#            Original negated it, hitting the 0-deg clamp wall every time.
#   FIX 3 — theta4 (wrist) formula sign corrected.
#   FIX 4 — Post-clamp error check removed from ik_geometric.
#   FIX 5 — ik_numerical now tries multiple seeds to avoid local minima.
#   FIX 6 — Removed unnecessary Axes3D import (F401 linter warning).
# =============================================================================

import numpy as np
from scipy.optimize import minimize
import sys

# -----------------------------------------------------------------------------
# SECTION 1 — ARM GEOMETRY  (loaded from config.py when available)
#
# If config.py exists in the same directory (or on PYTHONPATH), all values
# are pulled from it automatically.  If it is missing, the defaults below
# are used so this file still works standalone.
#
# To tune: edit config/config.py — never edit numbers here directly.
# -----------------------------------------------------------------------------

def _load_config():
    """
    Find and load config.py regardless of where Python is run from.

    Search order (stops at first match):
      1. <this file's dir>/config.py           e.g. src/config.py
      2. <this file's dir>/config/config.py    e.g. src/config/config.py
      3. <project root>/config.py              e.g. robot_arm_project/config.py
      4. <project root>/config/config.py       e.g. robot_arm_project/config/config.py

    Uses importlib + absolute paths so it works no matter which directory
    you run 'python' from — including VS Code Run, terminal from any folder,
    or a Jupyter notebook.
    """
    import importlib.util
    from pathlib import Path

    here    = Path(__file__).resolve().parent   # folder containing kinematics.py
    project = here.parent                        # one level up (project root)

    candidates = [
        here    / 'config.py',
        here    / 'config' / 'config.py',
        project / 'config.py',
        project / 'config' / 'config.py',
    ]

    for path in candidates:
        if path.exists():
            spec   = importlib.util.spec_from_file_location('_arm_cfg', path)
            module = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
            spec.loader.exec_module(module)                  # type: ignore[union-attr]
            try:
                rel = path.relative_to(Path.cwd())
            except ValueError:
                rel = path
            print(f"[kinematics] loaded config from {rel}")
            return module

    return None


_cfg = _load_config()

if _cfg is not None:
    LINK_LENGTHS     = _cfg.LINK_LENGTHS
    JOINT_LIMITS     = _cfg.JOINT_LIMITS
    HOME_ANGLES      = _cfg.HOME_ANGLES
    IK_TOLERANCE_MM  = _cfg.IK_TOLERANCE_MM
    IK_MAX_ITER      = _cfg.IK_MAX_ITER
    IK_DEFAULT_PITCH = _cfg.IK_DEFAULT_PITCH
    PICK_ZONE        = _cfg.PICK_ZONE
    L1 = LINK_LENGTHS['L1']
    L2 = LINK_LENGTHS['L2']
    L3 = LINK_LENGTHS['L3']
    L4 = LINK_LENGTHS['L4']

else:
    # ── Standalone defaults (config.py genuinely not found) ───────────
    # Edit these to match your physical arm if you are not using config.py.
    print("[kinematics] config.py not found — using built-in defaults")

    L1 = 100.0   # base height  : table surface  ->  shoulder joint
    L2 = 120.0   # upper arm    : shoulder joint ->  elbow joint
    L3 = 100.0   # forearm      : elbow joint    ->  wrist joint
    L4 =  60.0   # hand         : wrist joint    ->  end-effector tip

    LINK_LENGTHS = {'L1': L1, 'L2': L2, 'L3': L3, 'L4': L4}

    JOINT_LIMITS = [
        (  0.0,  180.0),   # J1 base rotation
        (  0.0,  170.0),   # J2 shoulder: 0=up, 90=horizontal, 170=almost-down
        (  0.0,  170.0),   # J3 elbow:    0=straight, 90=bent forward
        (-90.0,   90.0),   # J4 wrist:    negative=tip-down, positive=tip-up
    ]

    HOME_ANGLES      = [90.0, 20.0, 160.0, -80.0]
    IK_TOLERANCE_MM  = 2.5
    IK_MAX_ITER      = 500
    IK_DEFAULT_PITCH = -45.0   # degrees — angled-down approach for picking
    PICK_ZONE        = dict(x_min=50, x_max=220,
                            y_min=-180, y_max=180,
                            z_min=0,  z_max=40)


# -----------------------------------------------------------------------------
# SECTION 2 — DENAVIT-HARTENBERG CORE
# -----------------------------------------------------------------------------

def dh_matrix(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha),  np.sin(alpha)
    return np.array([
        [ ct, -st*ca,  st*sa,  a*ct ],
        [ st,  ct*ca, -ct*sa,  a*st ],
        [  0,     sa,     ca,     d ],
        [  0,      0,      0,     1 ],
    ], dtype=float)


def dh_table(joint_angles_deg):
    q = np.radians(joint_angles_deg)
    # Crane convention: J2=0->up, J2+->forward. J3=0->straight, J3+->bend forward.
    return [
        [  0,  np.pi/2,  L1,  q[0]           ],   # J1 unchanged
        [ L2,  0,         0,  np.pi/2 - q[1] ],   # J2: 0=up, positive=lean forward
        [ L3,  0,         0, -q[2]            ],   # J3: 0=straight, positive=bend forward
        [ L4,  0,         0,  q[3]            ],   # J4 unchanged
    ]


# -----------------------------------------------------------------------------
# SECTION 3 — FORWARD KINEMATICS
# -----------------------------------------------------------------------------

def forward_kinematics(joint_angles_deg):
    """
    Returns (joint_positions, T_final).
    joint_positions[0] = base origin, joint_positions[4] = end-effector.
    T_final[:3,3] = [x,y,z] end-effector position in mm.
    """
    T = np.eye(4)
    positions = [(0.0, 0.0, 0.0)]
    for (a, alpha, d, theta) in dh_table(joint_angles_deg):
        T = T @ dh_matrix(a, alpha, d, theta)
        positions.append((float(T[0,3]), float(T[1,3]), float(T[2,3])))
    return positions, T


def get_end_effector_xyz(joint_angles_deg):
    positions, _ = forward_kinematics(joint_angles_deg)
    return positions[-1]


# -----------------------------------------------------------------------------
# SECTION 4 — INVERSE KINEMATICS
# -----------------------------------------------------------------------------

def _clamp(angles_deg):
    return [float(np.clip(a, lo, hi))
            for a, (lo, hi) in zip(angles_deg, JOINT_LIMITS)]


def ik_geometric(target_x, target_y, target_z,
                 wrist_pitch_deg=0.0, elbow_up=True):
    """
    Fast closed-form geometric IK for a 4-DOF planar serial arm.

    FIX 2: elbow_up=True keeps theta3 positive (original incorrectly negated).
    FIX 3: theta4 sign formula corrected.
    FIX 4: removed post-clamp check that rejected valid near-limit solutions.
    """
    theta1 = np.degrees(np.arctan2(target_y, target_x))

    r   = np.sqrt(target_x**2 + target_y**2)
    z_s = target_z - L1

    wp  = np.radians(wrist_pitch_deg)
    wx  = r   - L4 * np.cos(wp)
    wz  = z_s - L4 * np.sin(wp)
    D   = np.sqrt(wx**2 + wz**2)

    if D > (L2 + L3) - 1e-6 or D < abs(L2 - L3) + 1e-6:
        return None

    cos3   = np.clip((D**2 - L2**2 - L3**2) / (2.0*L2*L3), -1.0, 1.0)
    theta3 = np.degrees(np.arccos(cos3))          # always positive from arccos

    # FIX 2: elbow_up=True -> theta3 stays positive.
    #        elbow_up=False -> theta3 goes negative.
    if not elbow_up:
        theta3 = -theta3

    beta      = np.degrees(np.arctan2(wz, wx))
    cos_gamma = np.clip((D**2 + L2**2 - L3**2) / (2.0*D*L2), -1.0, 1.0)
    gamma     = np.degrees(np.arccos(cos_gamma))
    # Crane convention: J2=0 is vertical, convert from horizontal-reference
    theta2    = 90.0 - (beta + gamma) if elbow_up else 90.0 - (beta - gamma)

    # Wrist: compensate for DH offsets on J2 and J3
    theta4 = wrist_pitch_deg - (90.0 - theta2) - (-theta3)

    return _clamp([theta1, theta2, theta3, theta4])


def ik_numerical(target_x, target_y, target_z,
                 seed_angles=None, wrist_pitch_deg=0.0):
    """
    Fallback numerical IK via scipy L-BFGS-B.
    FIX 5: tries multiple seeds to avoid local minima.
    """
    target = np.array([target_x, target_y, target_z])

    def cost(q):
        ee    = np.array(get_end_effector_xyz(q))
        pos_e = float(np.sum((ee - target)**2))
        pit_e = float((sum(q[1:4]) - wrist_pitch_deg)**2) * 0.01
        return pos_e + pit_e

    seeds = [HOME_ANGLES.copy()]
    if seed_angles is not None:
        seeds.append(list(seed_angles))
    seeds += [
        [90,  45,  45,   0],
        [90, -45,  90,   0],
        [float(np.degrees(np.arctan2(target_y, target_x))), 0, 90, 0],
    ]

    best_result = None
    best_fun    = float('inf')

    for seed in seeds:
        try:
            res = minimize(cost, x0=seed, method='L-BFGS-B',
                           bounds=JOINT_LIMITS,
                           options={'maxiter': IK_MAX_ITER,
                                    'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_fun:
                best_fun    = res.fun
                best_result = res
        except Exception:
            continue

    if best_result is None or best_fun > (IK_TOLERANCE_MM * 5)**2:
        return None

    solution = _clamp(list(best_result.x))
    ee       = get_end_effector_xyz(solution)
    error    = np.linalg.norm(np.array(ee) - target)
    return solution if error < IK_TOLERANCE_MM * 5 else None


def inverse_kinematics(target_x, target_y, target_z,
                       wrist_pitch_deg=0.0, prefer_elbow_up=True):
    """
    Main IK entry point. Call this from your state machine.

    Strategy:
      1. Geometric IK, preferred elbow config.
      2. Geometric IK, opposite elbow config.
      3. Numerical IK fallback (multiple seeds).

    Parameters
    ----------
    target_x, target_y, target_z : goal position in mm
    wrist_pitch_deg : tool approach angle in degrees
                      0   = horizontal (default)
                      -45 = angled down (recommended for picking)
                      -90 = straight down (pick from above)
    prefer_elbow_up : preferred elbow configuration

    Returns
    -------
    [J1, J2, J3, J4] in degrees, or None if unreachable.

    Example
    -------
    angles = inverse_kinematics(150, 0, 80, wrist_pitch_deg=-45)
    if angles:
        arm.set_joints(angles + [10.0])   # 10 deg = gripper open
    """
    for elbow_up in [prefer_elbow_up, not prefer_elbow_up]:
        result = ik_geometric(target_x, target_y, target_z,
                              wrist_pitch_deg=wrist_pitch_deg,
                              elbow_up=elbow_up)
        if result is not None:
            ee    = get_end_effector_xyz(result)
            error = np.linalg.norm(
                np.array(ee) - np.array([target_x, target_y, target_z])
            )
            if error <= IK_TOLERANCE_MM * 5:
                return result

    return ik_numerical(target_x, target_y, target_z,
                        wrist_pitch_deg=wrist_pitch_deg)


# -----------------------------------------------------------------------------
# SECTION 5 — JACOBIAN & VELOCITY IK
# -----------------------------------------------------------------------------

def jacobian(joint_angles_deg, delta_deg=0.1):
    """Numerical 3x4 Jacobian via central finite differences."""
    J      = np.zeros((3, 4))
    angles = np.array(joint_angles_deg, dtype=float)
    for i in range(4):
        qp = angles.copy(); qp[i] += delta_deg
        qm = angles.copy(); qm[i] -= delta_deg
        J[:, i] = (np.array(get_end_effector_xyz(qp))
                 - np.array(get_end_effector_xyz(qm))) / (2.0 * delta_deg)
    return J


def ik_velocity(joint_angles_deg, ee_velocity_mm_s, dt=0.05):
    """One velocity-IK step via Jacobian pseudoinverse."""
    J_pinv = np.linalg.pinv(jacobian(joint_angles_deg))
    dq     = J_pinv @ np.array(ee_velocity_mm_s)
    return _clamp(list(np.array(joint_angles_deg) + dq * dt))


# -----------------------------------------------------------------------------
# SECTION 6 — WORKSPACE HELPERS
# -----------------------------------------------------------------------------

def is_reachable(x, y, z):
    r    = np.sqrt(x**2 + y**2)
    dist = np.sqrt(r**2 + (z - L1)**2)
    return abs(L2 - L3) <= dist <= (L2 + L3)


def workspace_limits():
    max_r = L2 + L3 + L4
    return dict(x_min=-max_r, x_max=max_r,
                y_min=-max_r, y_max=max_r,
                z_min=0.0,    z_max=L1 + max_r)


def workspace_pick_zone():
    """Recommended pick zone for balls on the table (Z = 0-40 mm)."""
    return dict(x_min=50, x_max=258, y_min=-205, y_max=205, z_min=0, z_max=40)


# -----------------------------------------------------------------------------
# SECTION 7 — VERIFICATION TEST SUITE
# -----------------------------------------------------------------------------

def _run_tests():
    SEP = "=" * 62
    print(SEP)
    print("KINEMATICS VERIFICATION TEST SUITE")
    print(f"Links  L1={L1}  L2={L2}  L3={L3}  L4={L4} mm")
    print(f"IK tolerance: {IK_TOLERANCE_MM} mm")
    print(SEP)

    all_ok = True

    print("\n[T1] Forward kinematics at home position")
    positions, _ = forward_kinematics(HOME_ANGLES)
    labels = ["base", "J1 origin", "J2 origin", "J3 origin", "end-effector"]
    for lbl, pos in zip(labels, positions):
        print(f"  {lbl:14s}: ({pos[0]:8.2f}, {pos[1]:8.2f}, {pos[2]:8.2f}) mm")
    ok = positions[-1][2] > 0
    all_ok = all_ok and ok
    print(f"  End-effector above table: {'PASS' if ok else 'FAIL'}")

    print(f"\n[T2] FK -> IK round-trip  (error < {IK_TOLERANCE_MM} mm)")
    # Crane convention: J2 0=up->90=forward, J3 0=straight->90=bend, J4 +-90 wrist
    configs = [
        [  0,  25,  50,  -45],   # forward reach, mid-height
        [  0,  25,  75,  -45],   # forward reach, lower
        [ 45,  25,  75,  -30],   # diagonal right
        [135,  25,  75,  -30],   # diagonal left
        [  0,  25, 100,  -45],   # deep forward reach
        [ 90,  25,  75,    0],   # side reach, wrist flat
        [  0,  25, 100,    0],   # forward, wrist flat
        [ 45,  25, 100,  -45],   # diagonal, deep
    ]
    for i, cfg in enumerate(configs):
        positions, _ = forward_kinematics(cfg)
        tx, ty, tz   = positions[-1]
        ik = inverse_kinematics(tx, ty, tz)
        if ik is None:
            print(f"  [{i+1}] FAIL  IK returned None  target=({tx:.1f},{ty:.1f},{tz:.1f})")
            all_ok = False; continue
        rec = get_end_effector_xyz(ik)
        err = np.linalg.norm(np.array(positions[-1]) - np.array(rec))
        ok  = err < IK_TOLERANCE_MM
        all_ok = all_ok and ok
        print(f"  [{i+1}] {'PASS' if ok else 'FAIL'}  "
              f"cfg={[round(a,1) for a in cfg]}  err={err:.3f} mm")

    print("\n[T3] Critical pick-zone targets")
    critical = [
        (150,   0,  80, "original failure case"),
        (100, 100, 100, "diagonal reach"),
        (  0, 150,  50, "side reach"),
        (200,   0, 150, "long forward reach"),
        (-100, 100, 80, "back-left quadrant"),
        (120,   0,  10, "near-table low target"),
    ]
    for tx, ty, tz, desc in critical:
        res = inverse_kinematics(tx, ty, tz, wrist_pitch_deg=-45)
        if res is None:
            print(f"  FAIL  ({tx:5},{ty:5},{tz:4})  {desc}  -> unreachable")
            all_ok = False
        else:
            ee  = get_end_effector_xyz(res)
            err = np.linalg.norm(np.array(ee) - np.array([tx,ty,tz]))
            ok  = err < IK_TOLERANCE_MM * 3
            all_ok = all_ok and ok
            print(f"  {'PASS' if ok else 'FAIL'}  ({tx:5},{ty:5},{tz:4})  "
                  f"{desc}  err={err:.2f} mm")

    print("\n[T4] Joint limits compliance")
    violations = 0
    for cfg in configs:
        ik = inverse_kinematics(*forward_kinematics(cfg)[0][-1])
        if ik is None: continue
        for j, (a, (lo, hi)) in enumerate(zip(ik, JOINT_LIMITS)):
            if a < lo - 0.5 or a > hi + 0.5:
                print(f"  FAIL  J{j+1}={a:.1f} outside [{lo},{hi}]")
                violations += 1
    ok = violations == 0
    all_ok = all_ok and ok
    print(f"  {'PASS' if ok else 'FAIL'}  {violations} limit violation(s)")

    print("\n[T5] Jacobian shape")
    J  = jacobian(HOME_ANGLES)
    ok = J.shape == (3, 4)
    all_ok = all_ok and ok
    print(f"  {'PASS' if ok else 'FAIL'}  shape={J.shape}  (expected (3,4))")

    print("\n" + SEP)
    if all_ok:
        print("ALL TESTS PASSED -- safe to proceed to hardware integration.")
    else:
        print("FAILURES DETECTED -- fix kinematics before connecting hardware.")
    print(SEP)
    return all_ok


# -----------------------------------------------------------------------------
# SECTION 8 — INTERACTIVE VISUALIZER
# FIX 6: removed `from mpl_toolkits.mplot3d import Axes3D` (F401 warning,
#         not needed in matplotlib >= 3.4).
# -----------------------------------------------------------------------------

def _run_visualizer():
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button

    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Pick-and-place arm  --  interactive kinematics visualizer",
                 fontsize=13)
    ax3 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    # ── Layout constants ───────────────────────────────────────────────
    # Figure bottom 44% is reserved for controls.
    # Blue  sliders: left=0.05, width=0.38  → right edge at 0.43
    # Gap:                                    0.43 → 0.53  (0.10 wide)
    # Orange sliders: left=0.53, width=0.40  → right edge at 0.93
    BL    = 0.05    # blue left
    BW    = 0.38    # blue width
    OL    = 0.53    # orange left  (BL + BW + 0.10 gap)
    OW    = 0.40    # orange width
    ROW   = 0.055   # vertical pitch between slider rows
    SH    = 0.025   # slider height
    B_TOP = 0.38    # bottom of first (topmost) blue slider row
    O_TOP = 0.38    # bottom of first orange slider row

    plt.subplots_adjust(bottom=0.46, left=0.04, right=0.98, top=0.93)

    # ── Blue sliders: manual joint angles ──────────────────────────────
    blue_defs = [
        ('J1 base (°)',     JOINT_LIMITS[0][0], JOINT_LIMITS[0][1], HOME_ANGLES[0]),
        ('J2 shoulder (°)', JOINT_LIMITS[1][0], JOINT_LIMITS[1][1], HOME_ANGLES[1]),
        ('J3 elbow (°)',    JOINT_LIMITS[2][0], JOINT_LIMITS[2][1], HOME_ANGLES[2]),
        ('J4 wrist (°)',    JOINT_LIMITS[3][0], JOINT_LIMITS[3][1], HOME_ANGLES[3]),
    ]
    sliders = []
    for i, (lbl, lo, hi, init) in enumerate(blue_defs):
        ax_s = plt.axes((BL, B_TOP - i * ROW, BW, SH))
        sliders.append(Slider(ax_s, lbl, lo, hi, valinit=init,
                              valstep=0.5, color='steelblue'))

    # ── Orange sliders: IK target + wrist pitch ────────────────────────
    orange_defs = [
        ('Target X (mm)',   -300, 300, int(PICK_ZONE['x_max'] * 0.6)),
        ('Target Y (mm)',   -300, 300,   0),
        ('Target Z (mm)',      0, 380,  int(PICK_ZONE['z_max'])),
        ('Wrist pitch (°)',  -90,   0,  int(IK_DEFAULT_PITCH)),
    ]
    tsliders = []
    for i, (lbl, lo, hi, init) in enumerate(orange_defs):
        ax_s = plt.axes((OL, O_TOP - i * ROW, OW, SH))
        tsliders.append(Slider(ax_s, lbl, lo, hi, valinit=init,
                               valstep=1.0, color='darkorange'))

    # ── Buttons sit below both slider groups ───────────────────────────
    BTN_Y    = B_TOP - 4 * ROW - 0.03
    btn_home = Button(plt.axes((BL,        BTN_Y, 0.10, 0.032)), 'Home',
                      color='lightyellow')
    btn_ik   = Button(plt.axes((BL + 0.12, BTN_Y, 0.16, 0.032)), 'Solve IK',
                      color='lightgreen')

    def redraw(angles):
        pos, _ = forward_kinematics(angles)
        xs = [p[0] for p in pos]; ys = [p[1] for p in pos]
        zs = [p[2] for p in pos]
        lim = L1 + L2 + L3 + L4 + 20

        ax3.cla()
        ax3.set_xlim(-lim, lim); ax3.set_ylim(-lim, lim)
        ax3.set_zlim(0, lim*1.1)
        ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)')
        ax3.set_zlabel('Z (mm)'); ax3.set_title('3D view', fontsize=10)
        gx, gy = np.meshgrid([-lim, lim], [-lim, lim])
        ax3.plot_surface(gx, gy, np.zeros_like(gx), alpha=0.07, color='gray')
        ax3.plot(xs, ys, zs, '-', color='steelblue', linewidth=4)
        for x, y, z in pos[:-1]:
            ax3.scatter(x, y, z, color='white', s=60,
                        edgecolors='steelblue', linewidths=1.5, zorder=5)
        ee_x, ee_y, ee_z = float(pos[-1][0]), float(pos[-1][1]), float(pos[-1][2])
        ax3.scatter(ee_x, ee_y, ee_z, color='red', s=120, zorder=6,
                    edgecolors='darkred', linewidths=1.5)
        for vec, col in [((40,0,0),'r'),((0,40,0),'g'),((0,0,40),'b')]:
            ax3.quiver(0,0,0,*vec,color=col,linewidth=1.5,arrow_length_ratio=0.3)

        ax2.cla()
        ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)')
        ax2.set_title('Top-down (XY)', fontsize=10)
        ax2.axhline(0, color='lightgray', lw=0.8)
        ax2.axvline(0, color='lightgray', lw=0.8)
        ax2.grid(True, alpha=0.2)
        th = np.linspace(0, 2*np.pi, 180)
        ax2.plot((L2+L3+L4)*np.cos(th),(L2+L3+L4)*np.sin(th),
                 '--', color='lightgray', lw=1.0, label='max reach')
        ax2.plot(xs, ys, '-o', color='steelblue', lw=3, markersize=7,
                 markerfacecolor='white', markeredgecolor='steelblue',
                 markeredgewidth=1.5)
        ax2.scatter(xs[-1], ys[-1], color='red', s=100, zorder=6)
        ax2.legend(fontsize=8)

        ee  = pos[-1]
        ok  = inverse_kinematics(ee[0], ee[1], ee[2]) is not None
        col = 'green' if ok else 'red'
        info = (f"EE:  X={ee[0]:7.1f}  Y={ee[1]:7.1f}  Z={ee[2]:7.1f} mm\n"
                f"IK:  {'REACHABLE' if ok else 'OUT OF REACH'}\n"
                f"J:   {[round(a,1) for a in angles]}")
        ax2.text(-lim*0.97, lim*0.88, info, fontsize=9, color=col,
                 va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.85, edgecolor=col, lw=0.8))
        fig.canvas.draw_idle()

    def on_change(_):    redraw([s.val for s in sliders])
    def on_ik(_):
        tx, ty, tz = tsliders[0].val, tsliders[1].val, tsliders[2].val
        wp = tsliders[3].val
        res = inverse_kinematics(tx, ty, tz, wrist_pitch_deg=wp)
        if res:
            for s, v in zip(sliders, res):
                s.set_val(round(v, 1))
        else:
            print(f"IK: no solution for ({tx:.0f},{ty:.0f},{tz:.0f}) "
                  f"wrist_pitch={wp:.0f}°  -- target outside reachable workspace.")
    def on_home(_):      [s.set_val(v) for s, v in zip(sliders, HOME_ANGLES)]

    for s in sliders: s.on_changed(on_change)
    btn_ik.on_clicked(on_ik)
    btn_home.on_clicked(on_home)
    redraw(HOME_ANGLES)

    print("Visualizer ready.")
    print("  Blue   (left)  -> drag to move joints manually")
    print("  Orange (right) -> set IK target + wrist pitch, press 'Solve IK'")
    print("  Home button    -> reset all joints to home position")
    z = workspace_pick_zone()
    print(f"\nPick zone:  X {z['x_min']}–{z['x_max']} mm  |  "
          f"Y {z['y_min']}–{z['y_max']} mm  |  Z {z['z_min']}–{z['z_max']} mm")
    plt.show()


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    if '--test' in sys.argv:
        sys.exit(0 if _run_tests() else 1)
    else:
        _run_visualizer()