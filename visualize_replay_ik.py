"""Visualize SLAM replay + IK waypoints on Piper robot in placo/meshcat.

Shows:
    - Green:  camera trajectory (raw SLAM in robot base frame)
    - Red:    EE replay trajectory (IK targets)
    - Blue:   IK FK-computed ee_link positions
    - Robot model animated through IK joint waypoints
    - Frames at start/end of each trajectory

Usage:
    python visualize_replay_ik.py
    python visualize_replay_ik.py --traj /path/to/CameraTrajectory.txt
    python visualize_replay_ik.py --step 5 --frame-dt 0.02
"""

import argparse
import time
import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, points_viz
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics

PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = str(PROJECT_ROOT / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")
DEFAULT_TRAJ = str(PROJECT_ROOT.parent / "third_party" / "CameraTrajectory.txt")

INITIAL_JOINTS_DEG = [0.0, 40.11, -45.84, 0.0, 17.19, 0.0]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]


def load_slam_trajectory(path):
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


def compute_ee_trajectory(urdf_path, initial_joints_deg, slam_poses):
    robot = placo.RobotWrapper(urdf_path)
    for jn, val in zip(JOINT_NAMES, initial_joints_deg):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    T_world_cam = robot.get_T_world_frame("camera_link")
    T_world_ee = robot.get_T_world_frame("ee_link")
    T_ee_cam = np.linalg.inv(T_world_ee) @ T_world_cam
    T_cam_ee = np.linalg.inv(T_ee_cam)

    ee_poses = []
    cam_poses = []
    for T_w_cam_t in slam_poses:
        T_base_cam_t = T_world_cam @ T_w_cam_t
        cam_poses.append(T_base_cam_t)

        delta_ee = T_ee_cam @ T_w_cam_t @ T_cam_ee
        T_base_ee_t = T_world_ee @ delta_ee
        ee_poses.append(T_base_ee_t)

    return ee_poses, cam_poses, T_world_ee, T_world_cam


def compute_ik_waypoints(urdf_path, ee_poses, initial_joints_deg):
    kin = RobotKinematics(
        urdf_path=urdf_path,
        target_frame_name="ee_link",
        joint_names=JOINT_NAMES,
    )
    waypoints = []
    prev = np.array(initial_joints_deg, dtype=float)
    for T_target in ee_poses:
        j_ik = kin.inverse_kinematics(prev, T_target,
                                       position_weight=1.0,
                                       orientation_weight=0.01)
        waypoints.append(j_ik)
        prev = j_ik
    return waypoints


def animate_robot(viz, robot, joint_waypoints, step_size, frame_dt, N_total):
    """Animate the placo robot through IK joint waypoints in meshcat."""
    M = len(joint_waypoints)
    print(f"\nPlaying {M} frames at {frame_dt*1000:.0f}ms/frame...")
    print("  (Ctrl+C to stop animation, viewer stays open)\n")

    try:
        for step in range(M):
            joints_deg = joint_waypoints[step]
            for jn, deg in zip(JOINT_NAMES, joints_deg):
                robot.set_joint(jn, np.deg2rad(deg))
            robot.update_kinematics()
            viz.display(robot.state.q)

            j = joints_deg
            cam_idx = step * step_size
            T_ee = robot.get_T_world_frame("ee_link")
            print(f"\r  Frame {cam_idx:4d}/{N_total}  "
                  f"EE=({T_ee[0,3]*1000:.0f}, {T_ee[1,3]*1000:.0f}, {T_ee[2,3]*1000:.0f})mm  "
                  f"J=[{j[0]:+.1f}, {j[1]:+.1f}, {j[2]:+.1f}, "
                  f"{j[3]:+.1f}, {j[4]:+.1f}, {j[5]:+.1f}]",
                  end="", flush=True)
            time.sleep(frame_dt)

    except KeyboardInterrupt:
        pass

    print(f"\r  Frame {min(M * step_size, N_total):4d}/{N_total}  [STOPPED]" + " " * 30)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize SLAM replay + IK on Piper in meshcat")
    parser.add_argument("--traj", default=DEFAULT_TRAJ)
    parser.add_argument("--step", type=int, default=1,
                        help="Show every Nth frame (default 1 = all frames)")
    parser.add_argument("--frame-dt", type=float, default=0.03,
                        help="Seconds between animation frames (default 0.03)")
    args = parser.parse_args()

    # Compute
    print(f"Loading {args.traj}...")
    slam_poses = load_slam_trajectory(args.traj)
    N = len(slam_poses)
    print(f"  {N} SLAM frames")

    ee_poses, cam_poses, T_world_ee, T_world_cam = compute_ee_trajectory(
        URDF_PATH, INITIAL_JOINTS_DEG, slam_poses)

    print(f"Computing IK waypoints (target_frame=ee_link)...")
    joint_waypoints = compute_ik_waypoints(URDF_PATH, ee_poses, INITIAL_JOINTS_DEG)
    print(f"  {len(joint_waypoints)} waypoints computed")

    # Subsample
    slam_poses = slam_poses[::args.step]
    ee_poses = ee_poses[::args.step]
    cam_poses = cam_poses[::args.step]
    joint_waypoints = joint_waypoints[::args.step]
    M = len(slam_poses)
    print(f"  Showing every {args.step} frame(s): {M} animation frames")

    # Stats
    ee_pts = np.array([p[:3, 3] for p in ee_poses])
    cam_pts = np.array([p[:3, 3] for p in cam_poses])

    def path_len(pts):
        return np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)) * 1000

    print(f"\n  Camera path: {path_len(cam_pts):.1f} mm  |  "
          f"EE path: {path_len(ee_pts):.1f} mm  |  "
          f"Displacement: {np.linalg.norm(ee_pts[-1] - ee_pts[0])*1000:.1f} mm")

    jw = np.array(joint_waypoints)
    print(f"  Joint ranges:")
    for j in range(6):
        print(f"    joint{j+1}: [{np.min(jw[:, j]):+.1f}, {np.max(jw[:, j]):+.1f}]°")

    # --- Visualization ---
    robot = placo.RobotWrapper(URDF_PATH)
    for jn, val in zip(JOINT_NAMES, INITIAL_JOINTS_DEG):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    viz = robot_viz(robot)
    viz.display(robot.state.q)

    # Robot link frames
    for f in ["base_link", "link6"]:
        try:
            robot_frame_viz(robot, f)
        except Exception:
            pass

    # ee_link and camera_link at initial pose
    frame_viz("ee_link", T_world_ee)
    frame_viz("camera_link", T_world_cam)

    # Camera trajectory (green dots)
    points_viz("camera_trajectory", [p[:3, 3] for p in cam_poses],
               radius=0.003, color=0x00FF00)

    # EE replay trajectory (red dots)
    points_viz("ee_replay_trajectory", [p[:3, 3] for p in ee_poses],
               radius=0.004, color=0xFF0000)

    # IK FK-computed ee_link positions (blue dots) — verify IK matches target
    kin_viz = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="ee_link",
        joint_names=JOINT_NAMES,
    )
    ik_ee_points = []
    for j_ik in joint_waypoints:
        T = kin_viz.forward_kinematics(j_ik)
        ik_ee_points.append(T[:3, 3].copy())
    points_viz("ik_ee_points", ik_ee_points, radius=0.002, color=0x4444FF)

    # Start/end frames
    frame_viz("cam_start", cam_poses[0])
    frame_viz("cam_end", cam_poses[-1])
    frame_viz("ee_start", ee_poses[0])
    frame_viz("ee_end", ee_poses[-1])

    print(f"\n{'='*60}")
    print(f"VISUALIZATION ({M} frames)")
    print(f"{'='*60}")
    print(f"  GREEN  = Camera trajectory")
    print(f"  RED    = EE replay trajectory (IK target)")
    print(f"  BLUE   = IK FK-computed ee_link positions")
    print(f"  Robot  = Animated through IK joint waypoints")
    print(f"{'='*60}")

    # Animate
    animate_robot(viz, robot, joint_waypoints, args.step, args.frame_dt, N)

    input("\nPress Enter to close viewer.")


if __name__ == "__main__":
    main()
