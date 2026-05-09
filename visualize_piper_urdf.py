"""Visualize Piper URDF kinematic chain with coordinate frames at each link.

Uses placo for FK, matplotlib for 3D rendering.
Shows the arm at a given joint configuration with RGB axes at each link frame.

Usage:
  python visualize_piper_urdf.py
  python visualize_piper_urdf.py --joints 0 40 -45 0 17 0
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

from lerobot.model.kinematics import RobotKinematics

URDF_PATH = str(Path(__file__).resolve().parents[1] / "third_party" / "agx_arm_urdf" / "piper" / "urdf" / "piper_description.urdf")
LINK_NAMES = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]

AXIS_LENGTH = 0.06  # 60mm axis arrows


def get_link_frames(kin, joint_pos_deg):
    """Get T_world_frame for each link."""
    import placo
    frames = {}

    # Set joints
    joint_pos_rad = np.deg2rad(joint_pos_deg[:len(kin.joint_names)])
    for i, jn in enumerate(kin.joint_names):
        kin.robot.set_joint(jn, joint_pos_rad[i])
    kin.robot.update_kinematics()

    for name in LINK_NAMES:
        try:
            T = kin.robot.get_T_world_frame(name)
            frames[name] = T
        except Exception:
            pass
    return frames


def draw_frame(ax, T, name, length=AXIS_LENGTH, label=True):
    """Draw RGB axes for a 4x4 transform."""
    pos = T[:3, 3]
    R = T[:3, :3]

    colors = ['red', 'green', 'blue']
    labels = ['X', 'Y', 'Z']
    for i in range(3):
        direction = R[:, i] * length
        ax.quiver(pos[0], pos[1], pos[2],
                  direction[0], direction[1], direction[2],
                  color=colors[i], linewidth=2, arrow_length_ratio=0.15)

    if label:
        ax.text(pos[0], pos[1], pos[2] + 0.01, name, fontsize=8, ha='center')


def draw_chain(ax, frames):
    """Draw lines connecting link origins."""
    positions = []
    for name in LINK_NAMES:
        if name in frames:
            positions.append(frames[name][:3, 3])
    positions = np.array(positions)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            'k-o', markersize=4, linewidth=1.5)


def main():
    parser = argparse.ArgumentParser(description="Visualize Piper URDF")
    parser.add_argument("--joints", type=float, nargs=6, default=[0, 40.11, -45.84, 0, 17.19, 0],
                        help="Joint angles in degrees")
    args = parser.parse_args()

    kin = RobotKinematics(
        urdf_path=URDF_PATH,
        target_frame_name="link6",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    )

    configs = {
        "Initial": args.joints,
        "Zero":    [0, 0, 0, 0, 0, 0],
        "Home":    [0, 0, 0, 0, 25, 0],
        "Reach":   [0, 90, 0, 0, 0, 0],
    }

    fig = plt.figure(figsize=(16, 4))
    fig.suptitle("Piper URDF - Link Frames (RGB = XYZ)", fontsize=14)

    for idx, (cfg_name, joints) in enumerate(configs.items()):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')
        joints = np.array(joints, dtype=float)

        frames = get_link_frames(kin, joints)
        draw_chain(ax, frames)
        for name, T in frames.items():
            draw_frame(ax, T, name)

        # Formatting
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{cfg_name}\n{[f"{j:.0f}" for j in joints]}')

        # Equal aspect ratio
        all_pos = np.array([T[:3, 3] for T in frames.values()])
        center = all_pos.mean(axis=0)
        span = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2 + 0.05
        ax.set_xlim(center[0] - span, center[0] + span)
        ax.set_ylim(center[1] - span, center[1] + span)
        ax.set_zlim(center[2] - span, center[2] + span)

        ax.set_box_aspect([1, 1, 1])

    plt.tight_layout()

    # Save
    out = Path(__file__).parent / "piper_urdf_frames.png"
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
