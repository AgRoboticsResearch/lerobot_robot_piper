"""Visualize Piper URDF at initial pose with meshes + SROI gripper frames.

Usage:
  python visualize_piper_placo.py
"""

import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz
from pathlib import Path
import tempfile, os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MESH_BASE = str(PROJECT_ROOT / "third_party" / "agx_arm_urdf")
URDF_SRC = os.path.join(MESH_BASE, "piper", "urdf", "piper_description.urdf")

# Fix package:// paths to absolute
with open(URDF_SRC) as f:
    content = f.read()
content = content.replace("package://agx_arm_description/agx_arm_urdf/", MESH_BASE + "/")

# Inject SROI gripper frames (ee_link + camera_link)
gripper_links = '''
    <!-- SROI Gripper: ee_link = TCP (gripper tip), 116.517mm along link6 local +Z -->
    <link name="ee_link"/>
    <joint name="ee_joint" type="fixed">
        <origin xyz="0 0 0.116517" rpy="0 0 0"/>
        <parent link="link6"/>
        <child link="ee_link"/>
    </joint>
    <!-- SROI Gripper: camera_link -->
    <link name="camera_link"/>
    <joint name="camera_joint" type="fixed">
        <origin xyz="-0.054333 0.00895 0.013471" rpy="-0.930304 0 1.570793"/>
        <parent link="link6"/>
        <child link="camera_link"/>
    </joint>
'''
content = content.replace("</robot>", gripper_links + "</robot>")
tmp = tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False, dir="/tmp")
tmp.write(content)
tmp.close()
urdf_path = tmp.name

ARM_FRAMES = ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]
GRIPPER_FRAMES = ["ee_link", "camera_link"]

robot = placo.RobotWrapper(urdf_path)

# Set initial pose
for jn, val in zip(
    ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
    [0, 40.11, -45.84, 0, 17.19, 0],
):
    robot.set_joint(jn, np.deg2rad(val))
robot.update_kinematics()

# Visualize
viz = robot_viz(robot)
viz.display(robot.state.q)

# Show arm frames
for frame_name in ARM_FRAMES:
    try:
        robot_frame_viz(robot, frame_name)
    except Exception:
        pass

# Show gripper frames (ee_link = TCP, camera_link)
for frame_name in GRIPPER_FRAMES:
    try:
        T = robot.get_T_world_frame(frame_name)
        frame_viz(frame_name, T)
        pos = T[:3, 3]
        print(f"  {frame_name}: ({pos[0]*1000:.1f}, {pos[1]*1000:.1f}, {pos[2]*1000:.1f}) mm")
    except Exception as e:
        print(f"  {frame_name}: {e}")

# Cleanup temp file on exit
try:
    print("\nViewer open. Press Enter to exit.")
    input()
finally:
    os.unlink(urdf_path)
