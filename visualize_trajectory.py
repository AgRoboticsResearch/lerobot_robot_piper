"""Visualize ORB_SLAM3 CameraTrajectory.txt on the Piper robot.

Shows the SLAM trajectory starting from the camera_link frame.
ORB_SLAM3 format: 12 floats per line (R row-major + t).
ORB_SLAM3 convention: Y-axis negative = forward (camera looks along -Z in SLAM frame).

Usage:
  python visualize_trajectory.py
  python visualize_trajectory.py --traj third_party/1777602851/episode_002/CameraTrajectory.txt
"""

import numpy as np
import placo
from placo_utils.visualization import robot_viz, robot_frame_viz, frame_viz, points_viz
from pathlib import Path
import argparse, tempfile, os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MESH_BASE = str(PROJECT_ROOT / "third_party" / "agx_arm_urdf")
URDF_SRC = os.path.join(MESH_BASE, "piper", "urdf", "piper_description.urdf")


def load_slam_trajectory(path):
    """Load ORB_SLAM3 CameraTrajectory.txt (12 floats: R00 R01 R02 tx R10 R11 R12 ty R20 R21 R22 tz)."""
    poses = []
    with open(path) as f:
        for line in f:
            vals = [float(x) for x in line.strip().split()]
            if len(vals) != 12:
                continue
            T = np.eye(4)
            T[0, 0], T[0, 1], T[0, 2], T[0, 3] = vals[0], vals[1], vals[2], vals[3]
            T[1, 0], T[1, 1], T[1, 2], T[1, 3] = vals[4], vals[5], vals[6], vals[7]
            T[2, 0], T[2, 1], T[2, 2], T[2, 3] = vals[8], vals[9], vals[10], vals[11]
            poses.append(T)
    return poses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", default=str(PROJECT_ROOT / "third_party" / "CameraTrajectory.txt"),
                        help="Path to CameraTrajectory.txt")
    parser.add_argument("--all", action="store_true", help="Show all frames, not just positions")
    args = parser.parse_args()

    # Load trajectory
    traj = load_slam_trajectory(args.traj)
    print(f"Loaded {len(traj)} poses from {args.traj}")

    # Load robot URDF with meshes
    with open(URDF_SRC) as f:
        content = f.read()
    content = content.replace("package://agx_arm_description/agx_arm_urdf/", MESH_BASE + "/")
    # Inject gripper frames
    content = content.replace("</robot>", '''
    <link name="ee_link"/>
    <joint name="ee_joint" type="fixed">
        <origin xyz="0 0 0.116517" rpy="0 0 0"/>
        <parent link="link6"/>
        <child link="ee_link"/>
    </joint>
    <link name="camera_link"/>
    <joint name="camera_joint" type="fixed">
        <origin xyz="-0.054333 0.00895 0.013471" rpy="-0.930304 0 1.570793"/>
        <parent link="link6"/>
        <child link="camera_link"/>
    </joint>
</robot>''')
    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", mode="w", delete=False, dir="/tmp")
    tmp.write(content)
    tmp.close()

    robot = placo.RobotWrapper(tmp.name)

    # Set initial pose
    for jn, val in zip(
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        [0, 40.11, -45.84, 0, 17.19, 0],
    ):
        robot.set_joint(jn, np.deg2rad(val))
    robot.update_kinematics()

    # Get camera_link and ee_link poses in world frame
    T_world_cam = robot.get_T_world_frame("camera_link")
    T_world_ee = robot.get_T_world_frame("ee_link")
    print(f"Camera frame pos: ({T_world_cam[0,3]*1000:.1f}, {T_world_cam[1,3]*1000:.1f}, {T_world_cam[2,3]*1000:.1f}) mm")
    print(f"EE frame pos:     ({T_world_ee[0,3]*1000:.1f}, {T_world_ee[1,3]*1000:.1f}, {T_world_ee[2,3]*1000:.1f}) mm")

    # SLAM frame adjustment: flip X to correct left/right reversal
    T_flip = np.eye(4)
    T_flip[0, 0] = -1

    # Transforms between camera and ee_link
    T_cam_ee = np.linalg.inv(T_world_cam) @ T_world_ee  # ee→camera (ee expressed in camera frame)
    R_cam_ee = T_cam_ee[:3, :3]                         # rotation: ee coords → camera coords

    # ---- Explicit axis-remap matrix (camera→ee) ----
    # Physical correspondences:
    #   camera +X = gripper +Y
    #   camera -Y = gripper +Z  →  camera +Y = gripper -Z
    #   camera -Z = gripper +X  →  camera +Z = gripper -X
    # Columns of R_cam2ee are where each camera axis lands in ee frame:
    R_cam2ee = np.array([
        [0,  0, -1],   # camera +X → ee +Y
        [1,  0,  0],   # camera +Y → ee -Z  (so camera -Y → ee +Z)
        [0, -1,  0],   # camera +Z → ee -X  (so camera -Z → ee +X)
    ])
    print(f"\nExplicit axis-remap R_cam2ee (camera→ee):\n{R_cam2ee}")
    print(f"URDF-derived   R_ee_cam (camera→ee):\n{np.round(R_cam_ee.T, 3)}")
    print(f"Difference (ideally zero if 90° alignment): {np.linalg.norm(R_cam2ee - R_cam_ee.T):.3f}")

    # Verify key constraint
    check = R_cam2ee @ np.array([0, -1, 0.])
    print(f"R_cam2ee @ [0,-1,0] = [{check[0]:.1f}, {check[1]:.1f}, {check[2]:.1f}]  (expect [0,0,1]: camera -Y → ee +Z)")

    variants = {}

    for T_slam in traj:
        # Corrected camera motion relative to start (in camera-start frame)
        T_rel = T_flip @ T_slam

        # A) Camera baseline (green) - verified correct
        T_a = T_world_cam @ T_rel
        variants.setdefault("A_camera_baseline", {"poses": [], "points": []})
        variants["A_camera_baseline"]["poses"].append(T_a)
        variants["A_camera_baseline"]["points"].append(T_a[:3, 3].copy())

        # B) Rigid body (red): ee_world = camera_world @ T_cam_ee
        #    Uses URDF-derived T_cam_ee — complete correct transform
        T_b = T_a @ T_cam_ee
        variants.setdefault("B_rigid_body", {"poses": [], "points": []})
        variants["B_rigid_body"]["poses"].append(T_b)
        variants["B_rigid_body"]["points"].append(T_b[:3, 3].copy())

        # Camera-start → ee-start coordinate change:
        #   point_in_ee_start = R_ee_cam @ (point_in_cam_start - t_cam_ee)
        # where t_cam_ee = ee start pos in camera-start frame.
        T_rel_rot = T_rel[:3, :3]
        T_rel_trans = T_rel[:3, 3]
        t_cam_ee_vec = T_cam_ee[:3, 3]

        # E) Explicit axis-remap (yellow): uses ideal 90° R_cam2ee.
        #    Compute EE motion in camera-start frame, then remap to EE-start frame.
        ee_in_cam = T_rel_rot @ t_cam_ee_vec + T_rel_trans
        T_remap = np.eye(4)
        T_remap[:3, :3] = R_cam2ee @ T_rel_rot @ R_cam2ee.T   # conjugate rotation
        T_remap[:3, 3] = R_cam2ee @ (ee_in_cam - t_cam_ee_vec) # remap displacement to EE frame
        T_e = T_world_ee @ T_remap
        variants.setdefault("E_explicit_remap", {"poses": [], "points": []})
        variants["E_explicit_remap"]["poses"].append(T_e)
        variants["E_explicit_remap"]["points"].append(T_e[:3, 3].copy())

        # F) Frame-transform correction (orange): build corrected T_cam→ee with
        #    explicit 90° R_cam2ee + URDF translation, then standard rigid body.
        #    T_world_cam @ T_rel @ T_cam_to_ee_corrected
        #    NOTE: gives same positions as B because ee is at origin of its frame.
        T_cam_ee_corrected = np.eye(4)
        T_cam_ee_corrected[:3, :3] = R_cam2ee
        T_cam_ee_corrected[:3, 3] = t_cam_ee_vec
        T_f = T_world_cam @ T_rel @ T_cam_ee_corrected
        variants.setdefault("F_frame_corrected", {"poses": [], "points": []})
        variants["F_frame_corrected"]["poses"].append(T_f)
        variants["F_frame_corrected"]["points"].append(T_f[:3, 3].copy())

        # G) Full frame chain (cyan): decompose EE→camera→motion→EE as 5-step
        #    chain that reproduces yellow.
        #    T_remap = M1 @ M2 @ T_rel @ M4 @ M5
        #      M5 = [R_cam2ee.T | 0]   rotate EE coords → camera coords
        #      M4 = [I | t]            translate to EE position in camera frame
        #      T_rel                   camera motion in camera frame
        #      M2 = [I | -t]           undo translation
        #      M1 = [R_cam2ee | 0]     rotate camera coords → EE coords
        M1 = np.eye(4); M1[:3, :3] = R_cam2ee
        M2 = np.eye(4); M2[:3, 3] = -t_cam_ee_vec
        M4 = np.eye(4); M4[:3, 3] = t_cam_ee_vec
        M5 = np.eye(4); M5[:3, :3] = R_cam2ee.T
        T_remap_g = M1 @ M2 @ T_rel @ M4 @ M5
        T_g = T_world_ee @ T_remap_g
        variants.setdefault("G_frame_chain", {"poses": [], "points": []})
        variants["G_frame_chain"]["poses"].append(T_g)
        variants["G_frame_chain"]["points"].append(T_g[:3, 3].copy())

        # C) URDF axis-remap (blue): uses URDF R_ee_cam
        #    Should match B exactly
        R_ee_cam = R_cam_ee.T
        T_remap_c = np.eye(4)
        T_remap_c[:3, :3] = R_ee_cam @ T_rel_rot @ R_ee_cam.T
        T_remap_c[:3, 3] = R_ee_cam @ (T_rel_rot @ t_cam_ee_vec + T_rel_trans - t_cam_ee_vec)
        T_c = T_world_ee @ T_remap_c
        variants.setdefault("C_urdf_remap", {"poses": [], "points": []})
        variants["C_urdf_remap"]["poses"].append(T_c)
        variants["C_urdf_remap"]["points"].append(T_c[:3, 3].copy())

    # Sanity checks
    err_bc = np.linalg.norm(variants["B_rigid_body"]["points"][-1] - variants["C_urdf_remap"]["points"][-1]) * 1000
    err_be = np.linalg.norm(variants["B_rigid_body"]["points"][-1] - variants["E_explicit_remap"]["points"][-1]) * 1000
    err_ef = np.linalg.norm(variants["E_explicit_remap"]["points"][-1] - variants["F_frame_corrected"]["points"][-1]) * 1000
    err_ef_start = np.linalg.norm(variants["E_explicit_remap"]["points"][0] - variants["F_frame_corrected"]["points"][0]) * 1000
    err_eg = np.linalg.norm(variants["E_explicit_remap"]["points"][-1] - variants["G_frame_chain"]["points"][-1]) * 1000
    print(f"Variant B↔C (URDF remap, should be 0): {err_bc:.4f} mm")
    print(f"Variant B↔E (explicit remap, 90° vs 53°): {err_be:.1f} mm")
    print(f"Variant E↔F (yellow vs naive frame swap): {err_ef:.1f} mm")
    print(f"Variant E↔G (yellow vs 5-step frame chain): {err_eg:.4f} mm")
    print(f"Trajectory span: {np.linalg.norm(variants['A_camera_baseline']['points'][-1] - variants['A_camera_baseline']['points'][0])*1000:.1f} mm")

    # ============================================================
    # Path length analysis: demonstrate lever-arm effect
    # ============================================================
    def path_length(points):
        """Cumulative Euclidean path length (mm)."""
        pts = np.array(points)
        return np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)) * 1000

    print(f"\n{'='*60}")
    print(f"PATH LENGTH COMPARISON (cumulative sum of ‖Δp‖₂)")
    print(f"{'='*60}")
    len_cam = path_length(variants["A_camera_baseline"]["points"])
    len_ee  = path_length(variants["E_explicit_remap"]["points"])
    len_red = path_length(variants["B_rigid_body"]["points"])
    print(f"  A) camera-only  (green):  {len_cam:8.1f} mm")
    print(f"  E) gripper      (yellow): {len_ee:8.1f} mm  (Δ = {len_ee-len_cam:+.1f} mm,  {len_ee/len_cam*100:.1f}%)")
    print(f"  B) URDF rigid   (red):    {len_red:8.1f} mm  (Δ = {len_red-len_cam:+.1f} mm)")

    # Decompose the gripper displacement step-by-step:
    #   Δt_G = R_cam2ee · Δt_C  +  R_cam2ee · (ΔR_C · t - t)
    #         ╰── translation ──╯  ╰── lever arm ──────────╯
    traj_cam = variants["A_camera_baseline"]["poses"]
    traj_ee  = variants["E_explicit_remap"]["poses"]
    trans_cum = 0.0
    lever_cum = 0.0
    max_lever = 0.0
    max_lever_idx = 0
    for k in range(1, len(traj_cam)):
        d_cam_raw = np.linalg.inv(traj_cam[k-1]) @ traj_cam[k]
        dR_C = d_cam_raw[:3, :3]
        dt_C = d_cam_raw[:3, 3]
        # Translation component
        dt_trans = R_cam2ee @ dt_C
        # Lever-arm component
        dt_lever = R_cam2ee @ (dR_C @ t_cam_ee_vec - t_cam_ee_vec)
        trans_cum += np.linalg.norm(dt_trans) * 1000
        lever_cum += np.linalg.norm(dt_lever) * 1000
        n = np.linalg.norm(dt_lever) * 1000
        if n > max_lever:
            max_lever = n
            max_lever_idx = k

    print(f"\n  Gripper path decomposition (yellow, per-step):")
    print(f"    cumulative ‖translation term‖₂   : {trans_cum:8.1f} mm  ({trans_cum/len_ee*100:.1f}%)")
    print(f"    cumulative ‖lever-arm term‖₂      : {lever_cum:8.1f} mm  ({lever_cum/len_ee*100:.1f}%)")
    print(f"    max single-step lever-arm ‖·‖₂   : {max_lever:8.1f} mm  (frame {max_lever_idx})")
    print(f"    EE offset ‖t_cam_ee‖             : {np.linalg.norm(t_cam_ee_vec)*1000:8.1f} mm  (lever arm radius)")

    # Also decompose T_rel directly to show the raw per-frame breakdown
    print(f"\n  Per-frame displacement decomposition (SLAM camera frame):")
    rel_trans = np.array([np.linalg.norm(np.linalg.inv(traj_cam[i-1]) @ traj_cam[i])[:3, 3]) * 1000 for i in range(1, len(traj_cam))])
    # Compute total rotation (integrated angle) per frame
    total_rot_rad = 0.0
    for i in range(1, len(traj_cam)):
        dR = (np.linalg.inv(traj_cam[i-1]) @ traj_cam[i])[:3, :3]
        tr = np.trace(dR)
        angle = np.arccos(np.clip((tr - 1)/2, -1, 1))
        total_rot_rad += angle
    print(f"    total translational distance      : {np.sum(rel_trans):8.1f} mm")
    print(f"    total integrated rotation angle   : {np.degrees(total_rot_rad):8.1f}°")
    print(f"    avg per-frame ‖translation‖       : {np.mean(rel_trans):8.1f} mm")
    print(f"    max per-frame ‖translation‖       : {np.max(rel_trans):8.1f} mm")
    print(f"    ratio  lever/total in gripper path: {lever_cum/len_ee*100:.0f}%")
    print(f"    → {lever_cum/len_ee*100:.0f}% of the gripper's path length comes from")
    print(f"      rotation × lever-arm, NOT from the camera moving forward.")

    # Visualize robot
    viz = robot_viz(robot)
    viz.display(robot.state.q)

    for f in ["base_link", "link1", "link2", "link3", "link4", "link5", "link6"]:
        try:
            robot_frame_viz(robot, f)
        except Exception:
            pass

    # Show ee_link and camera_link frames
    frame_viz("ee_link", robot.get_T_world_frame("ee_link"))
    frame_viz("camera_link", robot.get_T_world_frame("camera_link"))

    # Show all trajectory variants
    colors = {
        "A_camera_baseline":  0x00FF00,  # green
        "B_rigid_body":       0xFF0000,  # red
        "C_urdf_remap":       0x0000FF,  # blue
        "E_explicit_remap":   0xFFFF00,  # yellow
        "F_frame_corrected":  0xFF8800,  # orange
        "G_frame_chain":      0x00FFFF,  # cyan
    }
    for name, data in variants.items():
        color = colors[name]
        points_viz(name, data["points"], radius=0.003, color=color)
        frame_viz(f"{name}_start", data["poses"][0])
        frame_viz(f"{name}_end", data["poses"][-1])

    print("\nTrajectory variants:")
    print("  A) green  - camera baseline (verified correct)")
    print("  B) red    - rigid body: T_world_cam @ T_rel @ T_cam_ee (URDF)")
    print("  C) blue   - axis remap with URDF R_ee_cam (should = B)")
    print("  E) yellow - explicit axis remap (camera -Y → ee +Z, 90° ideal)")
    print("  F) orange - naive: T_world_cam @ T_rel @ [R_cam2ee | t]  (= B positions)")
    print("  G) cyan   - 5-step chain: M1 @ M2 @ T_rel @ M4 @ M5  (should = E)")

    if args.all:
        # Show a frame every N poses
        poses = variants["A_camera_baseline"]["poses"]
        step = max(1, len(poses) // 20)
        for i in range(0, len(poses), step):
            frame_viz(f"slam_{i:04d}", poses[i])

    try:
        print("\nViewer open. Press Enter to exit.")
        input()
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    main()
