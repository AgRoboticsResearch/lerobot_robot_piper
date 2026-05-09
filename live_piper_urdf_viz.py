"""Live Piper URDF visualization — read real joint angles and mirror on placo/meshcat.

Reads the Piper arm joint state in real-time via the LeRobot SDK,
maps the 6 joint angles onto the URDF model, and updates the placo
meshcat visualizer at ~20 Hz.  The result is a "digital twin" that
moves in sync with the physical robot.

Usage:
    python live_piper_urdf_viz.py                  # default CAN interface (can0)
    python live_piper_urdf_viz.py --can can1       # alternate CAN bus
    python live_piper_urdf_viz.py --hz 30          # faster update rate
"""

import argparse
import time
import signal
import sys

import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, point_viz
from pathlib import Path

from lerobot_robot_piper import Piper, PiperConfig

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
URDF_PATH = str(PROJECT_ROOT / "lerobot_robot_piper" / "urdf" / "piper_description.urdf")
JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
ALL_LINK_NAMES = [
    "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
    "ee_link", "camera_link",
]



def build_robot():
    """Load URDF into placo and return the RobotWrapper."""
    robot = placo.RobotWrapper(URDF_PATH)
    return robot


def connect_piper(can: str) -> Piper:
    """Connect to the real Piper arm (no cameras, degrees mode)."""
    config = PiperConfig(
        can_interface=can,
        include_gripper=False,
        use_degrees=True,
        cameras={},
    )
    robot = Piper(config)
    robot.connect()
    return robot


def read_joint_deg(piper: Piper) -> list[float]:
    """Return 6 joint angles in degrees from the live robot."""
    obs = piper.get_observation()
    return [obs.get(f"joint_{i+1}.pos", 0.0) for i in range(6)]


def main():
    parser = argparse.ArgumentParser(
        description="Live URDF visualization of the real Piper arm"
    )
    parser.add_argument("--can", default="can0", help="CAN interface (default: can0)")
    parser.add_argument("--hz", type=float, default=20.0, help="Update rate in Hz (default: 20)")
    parser.add_argument(
        "--show-frames",
        action="store_true",
        default=True,
        help="Show coordinate frames for key links (default: True)",
    )
    args = parser.parse_args()

    dt = 1.0 / args.hz

    # ------------------------------------------------------------------
    # 1.  Load URDF model
    # ------------------------------------------------------------------
    print(f"[viz] Loading URDF: {URDF_PATH}")
    urdf_robot = build_robot()

    # ------------------------------------------------------------------
    # 2.  Connect to real Piper arm
    # ------------------------------------------------------------------
    print(f"[viz] Connecting to Piper on {args.can} …")
    piper = connect_piper(args.can)
    print("[viz] Connected ✓")

    # ------------------------------------------------------------------
    # 3.  Read initial joint state and initialize URDF
    # ------------------------------------------------------------------
    joints_deg = read_joint_deg(piper)
    for jn, val in zip(JOINT_NAMES, joints_deg):
        urdf_robot.set_joint(jn, np.deg2rad(val))
    urdf_robot.update_kinematics()

    # ------------------------------------------------------------------
    # 4.  Create meshcat visualizer
    # ------------------------------------------------------------------
    viz = robot_viz(urdf_robot)
    viz.display(urdf_robot.state.q)

    # Show reference frames for ALL links
    if args.show_frames:
        for f in ALL_LINK_NAMES:
            try:
                robot_frame_viz(urdf_robot, f)
            except Exception:
                pass

    print(f"[viz] Streaming at {args.hz} Hz — press Ctrl+C to stop\n")

    # ------------------------------------------------------------------
    # 5.  Graceful shutdown via Ctrl+C
    # ------------------------------------------------------------------
    running = True

    def _on_sigint(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _on_sigint)

    # ------------------------------------------------------------------
    # 6.  Main loop — read joints → update URDF → display
    # ------------------------------------------------------------------
    ee_trail: list[list[float]] = []  # optional EE trail
    MAX_TRAIL = 500

    try:
        while running:
            t0 = time.time()

            # Read real joint state
            joints_deg = read_joint_deg(piper)

            # Map to URDF
            for jn, val in zip(JOINT_NAMES, joints_deg):
                urdf_robot.set_joint(jn, np.deg2rad(val))
            urdf_robot.update_kinematics()

            # Push to meshcat
            viz.display(urdf_robot.state.q)

            # robot_frame_viz frames (set at init) auto-update via viz.display()
            # Do NOT call frame_viz() here — it creates duplicate standalone frames
            # that jitter/drift out of sync with the scene-graph-attached ones.

            # EE position trail (small green dots)
            try:
                T_ee = urdf_robot.get_T_world_frame("ee_link")
                ee_trail.append(T_ee[:3, 3].tolist())
                if len(ee_trail) > MAX_TRAIL:
                    ee_trail = ee_trail[-MAX_TRAIL:]
                # Show trail as point cloud
                from placo_utils.visualization import points_viz
                points_viz("ee_trail", ee_trail, radius=0.003, color=0x00FF88)
            except Exception:
                pass

            # Print joint values
            vals_str = "  ".join(
                f"j{i+1}={v:+7.2f}" for i, v in enumerate(joints_deg)
            )
            print(f"\r  {vals_str}", end="", flush=True)

            # Sleep to maintain target rate
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except Exception as e:
        print(f"\n[viz] Error: {e}")
    finally:
        print("\n\n[viz] Disconnecting …")
        piper.disconnect()
        print("[viz] Done.")


if __name__ == "__main__":
    main()
