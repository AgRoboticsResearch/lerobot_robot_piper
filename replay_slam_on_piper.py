#!/usr/bin/env python3
"""Replay EE trajectory on the real Piper arm.

Loads EETrajectory.txt (ee_link delta poses, same ORB_SLAM3 12-float format),
applies delta to the FK-derived start pose, solves IK with warm-start, validates,
and executes on the real arm.

Usage:
    python replay_slam_on_piper.py --dry-run
    python replay_slam_on_piper.py --max-frames 5
    python replay_slam_on_piper.py --speed 0.5
    python replay_slam_on_piper.py --camera-traj path/to/CameraTrajectory.txt
"""

import argparse
import sys
import time
import numpy as np
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics
from lerobot_robot_piper import Piper, PiperConfig

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = str(PROJECT_ROOT / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")
DEFAULT_EE_TRAJ = str(PROJECT_ROOT.parent / "Datasets" / "1777628135" /
                       "episode_001" / "EETrajectory.txt")

INITIAL_JOINTS_DEG = [0.0, 40.11, -45.84, 0.0, 17.19, 0.0]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# URDF joint limits (degrees)
JOINT_LIMITS_DEG = {
    "min": [-150.0, 0.0, -170.0, -100.0, -70.0, -120.0],
    "max": [150.0, 180.0, 0.0, 100.0, 70.0, 120.0],
}
JOINT_LIMIT_TOLERANCE_DEG = 0.5

# Safety thresholds
URDF_VELOCITY_LIMIT_RAD_S = 5.0
URDF_VELOCITY_LIMIT_DEG_S = np.rad2deg(URDF_VELOCITY_LIMIT_RAD_S)
MAX_JOINT_STEP_DEG = 10.0
MAX_IK_POS_ERROR_MM = 10.0
MAX_EE_STEP_MM = 100.0

DEFAULT_DT = 0.05
DEFAULT_SPEED = 1.0
SETTLE_TIME_S = 3.0


# ---------------------------------------------------------------------------
# Trajectory loading
# ---------------------------------------------------------------------------

def load_trajectory(path):
    """Load ORB_SLAM3-format trajectory (12 floats/line) → list of 4×4 matrices."""
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[0, :4] = vals[0:4]
            T[1, :4] = vals[4:8]
            T[2, :4] = vals[8:12]
            poses.append(T)
    return poses


# ---------------------------------------------------------------------------
# EE trajectory: apply EE deltas to FK start pose
# ---------------------------------------------------------------------------

def compute_ee_trajectory_from_ee_deltas(kin, ee_deltas, initial_joints_deg):
    """T_base_ee(t) = T_base_ee_start @ T_ee_delta(t).

    `kin` is a RobotKinematics(target_frame_name="ee_link", ...).
    `ee_deltas` are loaded from EETrajectory.txt (frame 0 = identity).
    """
    T_start = kin.forward_kinematics(np.array(initial_joints_deg))
    ee_poses = [T_start @ T_delta for T_delta in ee_deltas]
    return ee_poses, T_start


def compute_ee_trajectory_from_camera(urdf_path, initial_joints_deg, slam_poses):
    """Replay formula: T_base_ee(t) = T_base_ee_start @ T_ee_cam @ T_w_cam(t) @ T_cam_ee.

    Requires placo for the URDF FK to extract camera→EE transform.
    """
    import placo
    robot = placo.RobotWrapper(urdf_path)
    for jn, val in zip(JOINT_NAMES, initial_joints_deg):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    T_world_cam = robot.get_T_world_frame("camera_link")
    T_world_ee = robot.get_T_world_frame("ee_link")
    T_ee_cam = np.linalg.inv(T_world_ee) @ T_world_cam
    T_cam_ee = np.linalg.inv(T_ee_cam)

    ee_poses = []
    for T_w_cam_t in slam_poses:
        delta_ee = T_ee_cam @ T_w_cam_t @ T_cam_ee
        T_base_ee_t = T_world_ee @ delta_ee
        ee_poses.append(T_base_ee_t)

    print(f"\n  T_ee_cam =\n{np.round(T_ee_cam, 4)}")
    print(f"  T_cam_ee =\n{np.round(T_cam_ee, 4)}")
    return ee_poses, T_world_ee


# ---------------------------------------------------------------------------
# IK waypoints
# ---------------------------------------------------------------------------

def compute_ik_waypoints(kin, ee_poses, initial_joints_deg,
                         position_weight, orientation_weight):
    """Convert ee_link poses → joint angles via warm-start IK."""
    joint_waypoints = []
    ik_errors_mm = []
    prev = np.array(initial_joints_deg, dtype=float)

    print(f"\n{'='*60}")
    print(f"COMPUTING IK WAYPOINTS ({len(ee_poses)} frames)")
    print(f"{'='*60}")

    t0 = time.time()
    for i, T_target in enumerate(ee_poses):
        j_ik = kin.inverse_kinematics(prev, T_target,
                                       position_weight=position_weight,
                                       orientation_weight=orientation_weight)
        joint_waypoints.append(j_ik)

        T_check = kin.forward_kinematics(j_ik)
        err_mm = np.linalg.norm(T_check[:3, 3] - T_target[:3, 3]) * 1000
        ik_errors_mm.append(err_mm)
        prev = j_ik

        if i % 20 == 0 or i == len(ee_poses) - 1:
            print(f"  Frame {i:4d}/{len(ee_poses)}  IK err={err_mm:.3f}mm  "
                  f"J=[{j_ik[0]:+.1f}, {j_ik[1]:+.1f}, {j_ik[2]:+.1f}, "
                  f"{j_ik[3]:+.1f}, {j_ik[4]:+.1f}, {j_ik[5]:+.1f}]")

    elapsed = time.time() - t0
    print(f"\n  Computed {len(joint_waypoints)} waypoints in {elapsed:.1f}s "
          f"({len(joint_waypoints)/elapsed:.0f} frames/s)")
    return joint_waypoints, ik_errors_mm


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_trajectory(joint_waypoints, ee_poses, ik_errors_mm, dt, speed):
    """Pre-execution safety checks. Returns (passed, warnings)."""
    N = len(joint_waypoints)
    effective_dt = dt / speed
    warnings = []
    fatal = []

    print(f"\n{'='*60}")
    print(f"TRAJECTORY VALIDATION")
    print(f"{'='*60}")

    # Joint limits
    limit_violations = 0
    for i, joints in enumerate(joint_waypoints):
        for j in range(6):
            lo = JOINT_LIMITS_DEG["min"][j] - JOINT_LIMIT_TOLERANCE_DEG
            hi = JOINT_LIMITS_DEG["max"][j] + JOINT_LIMIT_TOLERANCE_DEG
            if joints[j] < lo or joints[j] > hi:
                limit_violations += 1
                if joints[j] < JOINT_LIMITS_DEG["min"][j] or joints[j] > JOINT_LIMITS_DEG["max"][j]:
                    fatal.append(f"Frame {i} joint{j+1}={joints[j]:.1f}° outside limit")

    if limit_violations > 0:
        warnings.append(f"Joint limit violations: {limit_violations} occurrences")

    # Joint velocity
    max_vel_deg_s = 0.0
    for i in range(1, N):
        step = np.max(np.abs(np.array(joint_waypoints[i]) - np.array(joint_waypoints[i-1])))
        vel_deg_s = step / effective_dt
        if vel_deg_s > max_vel_deg_s:
            max_vel_deg_s = vel_deg_s
        if vel_deg_s > URDF_VELOCITY_LIMIT_DEG_S:
            fatal.append(f"Frame {i} velocity {vel_deg_s:.0f}°/s exceeds URDF limit")

    max_step = max(
        np.max(np.abs(np.array(joint_waypoints[i]) - np.array(joint_waypoints[i-1])))
        for i in range(1, N)
    ) if N > 1 else 0.0

    # IK convergence
    max_ik_err = max(ik_errors_mm)
    mean_ik_err = np.mean(ik_errors_mm)
    bad_ik = sum(1 for e in ik_errors_mm if e > MAX_IK_POS_ERROR_MM)
    if bad_ik > 0:
        warnings.append(f"IK error > {MAX_IK_POS_ERROR_MM}mm on {bad_ik} frames")

    # Continuity
    for i, joints in enumerate(joint_waypoints):
        if np.any(np.isnan(joints)) or np.any(np.isinf(joints)):
            fatal.append(f"Frame {i}: NaN/Inf in joints")

    first_wp_err = np.max(np.abs(joint_waypoints[0] - np.array(INITIAL_JOINTS_DEG)))
    if first_wp_err > 2.0:
        warnings.append(f"First waypoint deviates from initial by {first_wp_err:.1f}°")

    # EE step
    ee_max_step = max(
        np.linalg.norm(ee_poses[i][:3, 3] - ee_poses[i-1][:3, 3]) * 1000
        for i in range(1, N)
    ) if N > 1 else 0.0
    if ee_max_step > MAX_EE_STEP_MM:
        warnings.append(f"Max EE step {ee_max_step:.1f}mm > {MAX_EE_STEP_MM}mm")

    # Summary
    print(f"  Frames:              {N}")
    print(f"  IK convergence:")
    print(f"    Max pos error:     {max_ik_err:.2f} mm")
    print(f"    Mean pos error:    {mean_ik_err:.2f} mm")
    print(f"    Frames > {MAX_IK_POS_ERROR_MM}mm:    {bad_ik}")
    print(f"  Joint limits:")
    print(f"    Violations:        {limit_violations}")

    joints_arr = np.array(joint_waypoints)
    print(f"  Joint range:")
    for j in range(6):
        lo, hi = JOINT_LIMITS_DEG["min"][j], JOINT_LIMITS_DEG["max"][j]
        margin_lo = np.min(joints_arr[:, j]) - lo
        margin_hi = hi - np.max(joints_arr[:, j])
        print(f"    joint{j+1}: [{np.min(joints_arr[:, j]):+.1f}, "
              f"{np.max(joints_arr[:, j]):+.1f}]°  "
              f"(margin: {margin_lo:.1f}° / {margin_hi:.1f}°)")
    print(f"  Joint velocity:")
    print(f"    Max per-step:      {max_step:.1f}°")
    print(f"    Max velocity:      {max_vel_deg_s:.0f}°/s  "
          f"(limit: {URDF_VELOCITY_LIMIT_DEG_S:.0f}°/s)")
    print(f"  EE step max:         {ee_max_step:.1f} mm")
    print(f"  Effective timestep:  {effective_dt*1000:.1f} ms  "
          f"(dt={dt*1000:.0f}ms / speed={speed})")

    if warnings:
        print(f"\n  --- WARNINGS ---")
        for w in warnings:
            print(f"  [!] {w}")
    else:
        print(f"\n  --- WARNINGS ---\n  None")

    if fatal:
        print(f"\n  --- FATAL ---")
        for f_err in fatal:
            print(f"  [X] {f_err}")
        print(f"\n  VERDICT: UNSAFE")
        # Suggest speed reduction if velocity was the issue
        if any("velocity" in f for f in fatal):
            min_safe_speed = max_vel_deg_s / (URDF_VELOCITY_LIMIT_DEG_S * 0.9)
            print(f"  HINT: use --speed {min_safe_speed:.1f} or lower to stay within velocity limit")
        return False, warnings

    status = "SAFE" if not warnings else "SAFE WITH WARNINGS"
    print(f"\n  VERDICT: {status}")
    return True, warnings


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def execute_trajectory(robot, joint_waypoints, initial_joints_deg, dt, speed,
                       max_frames, no_reverse, dry_run):
    """Send joint waypoints to the real Piper arm."""
    waypoints = joint_waypoints[:max_frames] if max_frames else joint_waypoints
    N = len(waypoints)
    effective_dt = dt / speed
    initial_dict = {f"joint_{i+1}.pos": initial_joints_deg[i] for i in range(6)}

    if dry_run:
        print(f"\n{'='*60}")
        print(f"DRY RUN — skipping execution")
        print(f"{'='*60}")
        return

    print(f"\n{'='*60}")
    print(f"EXECUTING ({N} frames, dt={effective_dt*1000:.0f}ms)")
    print(f"{'='*60}")

    print(f"\n[1] Moving to initial pose...")
    robot.send_action(initial_dict)
    time.sleep(SETTLE_TIME_S)
    obs = robot.get_observation()
    actual = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
    print(f"  Actual: J=[{actual[0]:+.1f}, {actual[1]:+.1f}, {actual[2]:+.1f}, "
          f"{actual[3]:+.1f}, {actual[4]:+.1f}, {actual[5]:+.1f}]")

    input("\nPress Enter to start... (Ctrl+C to stop)")

    print(f"\n[2] Forward trajectory...")
    t_start = time.time()
    prev_joints = np.array(initial_joints_deg, dtype=float)

    try:
        for step, j_ik in enumerate(waypoints):
            if step > 0:
                d = j_ik - prev_joints
                max_d = np.max(np.abs(d))
                if max_d > MAX_JOINT_STEP_DEG:
                    d = d * (MAX_JOINT_STEP_DEG / max_d)
                    j_ik = prev_joints + d

            robot.send_action({f"joint_{i+1}.pos": j_ik[i] for i in range(6)})
            time.sleep(effective_dt)
            prev_joints = j_ik.copy()

            if (step + 1) % 20 == 0:
                obs = robot.get_observation()
                actual = np.array([obs[f"joint_{i+1}.pos"] for i in range(6)])
                print(f"  Frame {step+1:4d}/{N}  "
                      f"J=[{actual[0]:+.1f}, {actual[1]:+.1f}, {actual[2]:+.1f}, "
                      f"{actual[3]:+.1f}, {actual[4]:+.1f}, {actual[5]:+.1f}]")

        elapsed = time.time() - t_start
        print(f"  Forward done in {elapsed:.1f}s ({N/elapsed:.1f} steps/s)")

        if not no_reverse:
            print(f"\n[3] Reverse trajectory...")
            t_start = time.time()
            for j_ik in reversed(waypoints):
                robot.send_action({f"joint_{i+1}.pos": j_ik[i] for i in range(6)})
                time.sleep(effective_dt)
            elapsed = time.time() - t_start
            print(f"  Reverse done in {elapsed:.1f}s ({N/elapsed:.1f} steps/s)")

    except KeyboardInterrupt:
        print("\n\n[!] Emergency stop — holding position.")
        for _ in range(3):
            robot.send_action({f"joint_{i+1}.pos": prev_joints[i] for i in range(6)})
            time.sleep(0.05)

    print("\nDone.")


# ---------------------------------------------------------------------------
# Trajectory statistics
# ---------------------------------------------------------------------------

def print_trajectory_stats(ee_poses, T_start):
    pts = np.array([p[:3, 3] for p in ee_poses])

    def path_len(pts_arr):
        return np.sum(np.linalg.norm(pts_arr[1:] - pts_arr[:-1], axis=1)) * 1000

    ee_len = path_len(pts)
    displacement = np.linalg.norm(pts[-1] - pts[0]) * 1000

    print(f"\n{'='*60}")
    print(f"TRAJECTORY STATISTICS")
    print(f"{'='*60}")
    print(f"  T_base_ee_start: ({T_start[0,3]*1000:.1f}, "
          f"{T_start[1,3]*1000:.1f}, {T_start[2,3]*1000:.1f}) mm")
    print(f"  EE path length:   {ee_len:.1f} mm")
    print(f"  EE displacement:  {displacement:.1f} mm")
    print(f"  EE start: ({pts[0][0]*1000:.1f}, {pts[0][1]*1000:.1f}, {pts[0][2]*1000:.1f}) mm")
    print(f"  EE end:   ({pts[-1][0]*1000:.1f}, {pts[-1][1]*1000:.1f}, {pts[-1][2]*1000:.1f}) mm")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Replay EE trajectory on the real Piper arm")
    parser.add_argument("--traj", default=DEFAULT_EE_TRAJ,
                        help="Path to EETrajectory.txt (ee_link delta poses)")
    parser.add_argument("--camera-traj", default=None,
                        help="Path to CameraTrajectory.txt (use replay formula)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-reverse", action="store_true")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.01)
    parser.add_argument("--initial-joints", type=str, default=None,
                        help="Override initial joints: 'j1,j2,j3,j4,j5,j6' in degrees")
    args = parser.parse_args()

    initial_joints_deg = list(INITIAL_JOINTS_DEG)
    if args.initial_joints is not None:
        values = [float(x.strip()) for x in args.initial_joints.split(",")]
        if len(values) != 6:
            print(f"Error: --initial-joints expects 6 values, got {len(values)}")
            sys.exit(1)
        initial_joints_deg = values

    # --- Load trajectory ---
    traj_path = args.camera_traj or args.traj
    mode = "camera (replay formula)" if args.camera_traj else "EE delta"
    print(f"\n{'='*60}")
    print(f"LOADING TRAJECTORY ({mode})")
    print(f"{'='*60}")
    deltas = load_trajectory(traj_path)
    N = len(deltas)
    print(f"Loaded {N} poses from {traj_path}")

    err_identity = np.linalg.norm(deltas[0] - np.eye(4))
    print(f"  Frame 0 deviation from identity: {err_identity:.6f}")
    if err_identity > 0.01:
        print("  WARNING: Frame 0 is NOT identity.")

    # --- Create kinematics ---
    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_link",
        joint_names=JOINT_NAMES,
    )

    # --- Compute EE poses ---
    if args.camera_traj:
        ee_poses, T_start = compute_ee_trajectory_from_camera(
            URDF_PATH, initial_joints_deg, deltas)
    else:
        ee_poses, T_start = compute_ee_trajectory_from_ee_deltas(
            kin, deltas, initial_joints_deg)

    print_trajectory_stats(ee_poses, T_start)

    # --- IK ---
    joint_waypoints, ik_errors_mm = compute_ik_waypoints(
        kin, ee_poses, initial_joints_deg,
        args.position_weight, args.orientation_weight)

    # --- Validate ---
    ok, _ = validate_trajectory(
        joint_waypoints, ee_poses, ik_errors_mm, args.dt, args.speed)
    if not ok:
        print("\nAborting: failed safety validation.")
        sys.exit(1)

    if args.dry_run:
        print("\nDry run complete.")
        sys.exit(0)

    # --- Execute ---
    config = PiperConfig(
        can_interface="can0",
        include_gripper=False,
        use_degrees=True,
        cameras={},
    )
    robot = Piper(config)
    robot.connect()
    print("Connected.")

    try:
        execute_trajectory(
            robot, joint_waypoints, initial_joints_deg,
            args.dt, args.speed, args.max_frames, args.no_reverse, args.dry_run)
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
