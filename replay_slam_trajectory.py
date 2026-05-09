"""Replay EE trajectory on Piper robot.

Supports two sources:
  1) SLAM trajectory file (--traj): 12 floats per line (ORB_SLAM3 format)
  2) LeRobot dataset (--dataset): reads observation.state from parquet

Usage:
    python replay_slam_trajectory.py --traj /path/to/CameraTrajectory.txt
    python replay_slam_trajectory.py --dataset /path/to/dataset --episodes 0 1 5
    python replay_slam_trajectory.py --dataset /path/to/dataset              # all episodes
"""

import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, points_viz
from scipy.spatial.transform import Rotation
from pathlib import Path
import os, argparse, glob

PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = str(PROJECT_ROOT / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")

DEFAULT_TRAJ = str(PROJECT_ROOT.parent / "1777548643" / "episode_001" / "CameraTrajectory.txt")
DEFAULT_DATASET = str(PROJECT_ROOT / "Datasets" / "sroi_piper_strawberry_picking")

INITIAL_JOINTS_DEG = [0, 40.11, -45.84, 0, 17.19, 0]
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

EPISODE_COLORS = [
    0xFF0000, 0x00AA00, 0x0066FF, 0xFF8800, 0xAA00FF,
    0x00CCCC, 0xFF00AA, 0x888800, 0x008888, 0x880088,
]


def load_trajectory(path):
    """Load ORB_SLAM3 CameraTrajectory.txt -> list of 4x4 matrices."""
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


def load_dataset_episodes(dataset_path, episode_indices=None):
    """Load episodes from LeRobot dataset parquet files.
    Returns dict {episode_idx: (ee_poses, ee_points, states_array)}.
    """
    import pyarrow.parquet as pq

    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(dataset_path, "data", "**", "*.parquet"), recursive=True))
    assert parquet_files, f"No parquet files found in {dataset_path}"

    df = pq.read_table(parquet_files).to_pandas()
    all_episodes = sorted(df["episode_index"].unique())

    if episode_indices is None:
        episode_indices = all_episodes
    else:
        episode_indices = [e for e in episode_indices if e in all_episodes]

    result = {}
    for ep_idx in episode_indices:
        ep_df = df[df["episode_index"] == ep_idx].sort_values("frame_index")
        states = np.stack(ep_df["observation.state"].values)  # (N, 7)

        # Build 4x4 delta poses from [x, y, z, wx, wy, wz, gripper]
        poses = []
        for s in states:
            T = np.eye(4)
            T[:3, 3] = s[:3]  # translation
            T[:3, :3] = Rotation.from_rotvec(s[3:6]).as_matrix()  # axis-angle -> R
            poses.append(T)
        result[ep_idx] = poses

    return result


def main():
    parser = argparse.ArgumentParser(description="Replay EE trajectory on Piper robot")
    parser.add_argument("--traj", default=None,
                        help="Path to trajectory file (12 floats per line)")
    parser.add_argument("--dataset", default=None,
                        help="Path to LeRobot dataset directory")
    parser.add_argument("--episodes", nargs="*", type=int, default=None,
                        help="Episode indices to visualize (default: all)")
    args = parser.parse_args()

    if not args.traj and not args.dataset:
        args.dataset = DEFAULT_DATASET

    # 1. Load robot and set initial config
    assert os.path.isfile(URDF_PATH), f"URDF not found: {URDF_PATH}"
    robot = placo.RobotWrapper(URDF_PATH)

    for jn, val in zip(JOINT_NAMES, INITIAL_JOINTS_DEG):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    T_base_ee_start = robot.get_T_world_frame("ee_link")
    print(f"Initial EE pos: ({T_base_ee_start[0,3]*1000:.1f}, "
          f"{T_base_ee_start[1,3]*1000:.1f}, {T_base_ee_start[2,3]*1000:.1f}) mm")

    # 2. Load trajectories
    all_trajectories = {}  # {label: (ee_poses, color)}

    if args.traj:
        traj = load_trajectory(args.traj)
        ee_poses = [T_base_ee_start @ T for T in traj]
        all_trajectories[args.traj] = (ee_poses, 0xFF0000)
        print(f"Loaded {len(traj)} poses from {args.traj}")

    if args.dataset:
        episodes = load_dataset_episodes(args.dataset, args.episodes)
        print(f"Loaded {len(episodes)} episodes from {args.dataset}")
        for ep_idx, poses in episodes.items():
            ee_poses = [T_base_ee_start @ T for T in poses]
            color = EPISODE_COLORS[ep_idx % len(EPISODE_COLORS)]
            label = f"ep{ep_idx}"
            all_trajectories[label] = (ee_poses, color)

    # 3. Visualize
    viz = robot_viz(robot)
    viz.display(robot.state.q)

    for f in ["base_link", "link1", "link2", "link3", "link4", "link5", "link6",
              "ee_link", "camera_link"]:
        try:
            robot_frame_viz(robot, f)
        except Exception:
            pass

    for label, (ee_poses, color) in all_trajectories.items():
        ee_points = [T[:3, 3] for T in ee_poses]
        pts = np.array(ee_points)
        path_len = np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)) * 1000

        print(f"\n  [{label}] {len(ee_poses)} frames, {path_len:.1f} mm path")
        print(f"    Start: ({pts[0,0]*1000:.1f}, {pts[0,1]*1000:.1f}, {pts[0,2]*1000:.1f}) mm")
        print(f"    End:   ({pts[-1,0]*1000:.1f}, {pts[-1,1]*1000:.1f}, {pts[-1,2]*1000:.1f}) mm")

        points_viz(f"traj_{label}", ee_points, radius=0.003, color=color)
        frame_viz(f"start_{label}", ee_poses[0])
        frame_viz(f"end_{label}", ee_poses[-1])

    print("\nViewer open. Press Enter to exit.")
    input()


if __name__ == "__main__":
    main()
