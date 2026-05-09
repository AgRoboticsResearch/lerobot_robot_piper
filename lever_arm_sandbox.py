"""Interactive sandbox: pure rotation × lever-arm → path-length difference.

Drag the rotation slider to see how the same rotation causes the camera and
gripper to travel drastically different distances on a rigid body.

Key insight:
    T_W^C = T_W^G · T_G^C   (constant extrinsic transform)

When the body rotates by θ, the camera and gripper both rotate by θ, but their
linear displacements differ by the lever-arm arc length:
    ‖Δp_G‖ ≈ ‖Δp_C‖ + |t|·θ   (for small θ, rotation around camera)

Usage:
    python lever_arm_sandbox.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ---- Fixed parameters from URDF (camera→gripper offset) ----
# These match the URDF injection in visualize_trajectory.py:
#   camera_link origin is at (-0.054333, 0.00895, 0.013471) in link6 frame
#   ee_link    origin is at (0, 0, 0.116517) in link6 frame
# The camera−ee offset in camera frame is T_cam_ee (computed by the robot).
# Hardcode the value printed by visualize_trajectory.py for this joint config:
T_CAM_EE_TRANSLATION = np.array([0.0595, -0.1165, 0.1082])  # meters, approximate
# For the 2D sandbox we use the XZ (top-down) projection — the dominant components:
LEVER_ARM_2D = np.array([0.0595, 0.1082])  # X, Z in camera frame (meters)
# This is the offset from camera to gripper in the camera's local frame.

# The explicit axis-remap (camera -Y → gripper +Z, camera -Z → gripper +X)
# means rotation in the camera XZ plane maps to gripper XY plane.
# For the sandbox, we work in a simplified 2D plane.

ROTATION_AXIS = "Y"  # camera Y axis (pitch) — the dominant SLAM rotation axis


def rotation_matrix_2d(theta_rad):
    """2D rotation matrix."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, -s], [s, c]])


def build_figure():
    """Build the interactive sandbox figure."""
    plt.ioff()
    fig = plt.figure("Lever-Arm Sandbox — Pure Rotation", figsize=(14, 7))

    # Left: 2D top-down view
    ax_viz = fig.add_axes([0.05, 0.25, 0.50, 0.70])
    ax_viz.set_aspect("equal")
    ax_viz.set_xlim(-0.25, 0.25)
    ax_viz.set_ylim(-0.25, 0.25)
    ax_viz.grid(True, alpha=0.3)
    ax_viz.set_xlabel("X (m) — camera frame")
    ax_viz.set_ylabel("Z (m) — camera frame")
    ax_viz.set_title("Top-down view: rigid body under pure rotation")

    # Draw rotation center (camera origin)
    (cam_pt,) = ax_viz.plot([0], [0], "go", markersize=12, label="Camera (rotation center)", zorder=5)
    (grip_pt,) = ax_viz.plot(
        [LEVER_ARM_2D[0]], [LEVER_ARM_2D[1]], "rs", markersize=12, label="Gripper", zorder=5
    )

    # Rigid body link line
    (body_line,) = ax_viz.plot(
        [0, LEVER_ARM_2D[0]], [0, LEVER_ARM_2D[1]], "k-", linewidth=3, label="Rigid body link"
    )

    # Gripper arc (traced over full slider range)
    (arc_line,) = ax_viz.plot([], [], "r--", linewidth=1, alpha=0.5, label="Gripper arc trace")

    # Ghost positions at previous angle
    (cam_ghost,) = ax_viz.plot([], [], "o", color="lightgreen", markersize=8, alpha=0.4)
    (grip_ghost,) = ax_viz.plot([], [], "s", color="lightcoral", markersize=8, alpha=0.4)
    (body_ghost,) = ax_viz.plot([], [], "-", color="gray", linewidth=2, alpha=0.3)

    # Displacement arrows
    cam_arrow = None
    grip_arrow = None

    ax_viz.legend(loc="upper right", fontsize=8)
    ax_viz.axhline(0, color="gray", linewidth=0.5)
    ax_viz.axvline(0, color="gray", linewidth=0.5)

    # Right: numeric panel
    ax_text = fig.add_axes([0.60, 0.25, 0.37, 0.70])
    ax_text.axis("off")

    # Slider area
    ax_slider_theta = fig.add_axes([0.15, 0.12, 0.70, 0.04])
    slider_theta = Slider(
        ax=ax_slider_theta,
        label="Rotation angle θ (degrees)",
        valmin=0,
        valmax=360,
        valinit=0,
        valstep=0.5,
    )

    # Secondary slider: lever-arm radius scale
    ax_slider_radius = fig.add_axes([0.15, 0.05, 0.70, 0.04])
    slider_radius = Slider(
        ax=ax_slider_radius,
        label="Lever-arm radius scale (×)",
        valmin=0.2,
        valmax=3.0,
        valinit=1.0,
        valstep=0.05,
    )

    # State
    state = {
        "prev_theta": 0.0,
        "prev_radius": 1.0,
        "arc_history": [],  # store all grip positions for trace
    }

    def update(val=None):
        theta_deg = slider_theta.val
        theta_rad = np.deg2rad(theta_deg)
        radius_scale = slider_radius.val

        t = LEVER_ARM_2D * radius_scale
        R = rotation_matrix_2d(theta_rad)

        # Camera stays at origin (rotation center)
        cam_pos = np.array([0.0, 0.0])
        # Gripper rotates around camera
        grip_pos = R @ t

        # Update main positions
        cam_pt.set_data([cam_pos[0]], [cam_pos[1]])
        grip_pt.set_data([grip_pos[0]], [grip_pos[1]])
        body_line.set_data([cam_pos[0], grip_pos[0]], [cam_pos[1], grip_pos[1]])

        # Update arc trace — accumulate arc points
        if abs(theta_rad - state["prev_theta"]) > 0.02 or radius_scale != state["prev_radius"]:
            state["arc_history"].append(grip_pos.copy())
            # Prune old history when radius changes
            if radius_scale != state["prev_radius"]:
                state["arc_history"] = state["arc_history"][-500:]
            if len(state["arc_history"]) > 500:
                state["arc_history"] = state["arc_history"][-500:]
            if len(state["arc_history"]) > 1:
                arc_pts = np.array(state["arc_history"])
                arc_line.set_data(arc_pts[:, 0], arc_pts[:, 1])

        # Ghost at half angle
        theta_half = theta_rad * 0.5
        R_half = rotation_matrix_2d(theta_half)
        grip_half = R_half @ t
        cam_ghost.set_data([0], [0])
        grip_ghost.set_data([grip_half[0]], [grip_half[1]])
        body_ghost.set_data([0, grip_half[0]], [0, grip_half[1]])

        # ---- Numeric computation ----
        # Camera displacement = 0 (rotation around camera)
        cam_disp = 0.0

        # Gripper displacement = |R·t - t|
        grip_disp = np.linalg.norm(grip_pos - t)

        # Lever-arm arc length = |t| * |θ|  (exact arc length for rotation around origin)
        arc_len = np.linalg.norm(t) * abs(theta_rad)

        # Chord length (straight-line displacement) = 2 * |t| * sin(|θ|/2)
        chord_len = 2 * np.linalg.norm(t) * np.sin(abs(theta_rad) / 2)

        # Build text display
        lines = []
        lines.append("=" * 55)
        lines.append("  PURE ROTATION SANDBOX — Lever-Arm Effect")
        lines.append("=" * 55)
        lines.append("")
        lines.append(f"  Rotation angle θ:          {theta_deg:8.1f}°  ({theta_rad:.3f} rad)")
        lines.append(f"  Lever-arm vector |t|:      {np.linalg.norm(t)*1000:8.1f} mm")
        lines.append(f"                     t_x:    {t[0]*1000:8.1f} mm")
        lines.append(f"                     t_z:    {t[1]*1000:8.1f} mm")
        lines.append("")
        lines.append("  ── Displacements ──")
        lines.append(f"  Camera displacement:       {cam_disp*1000:8.1f} mm  (rotation center!)")
        lines.append(f"  Gripper displacement:      {grip_disp*1000:8.1f} mm  (chord)")
        lines.append(f"  Gripper arc length:        {arc_len*1000:8.1f} mm  (|t| × θ)")
        lines.append(f"  Ratio grip/cam:               ∞  (cam stays still)")
        lines.append("")
        lines.append("  ── What this means ──")
        lines.append(f"  The camera {'' if theta_deg < 0.5 else 'rotates ' + f'{theta_deg:.0f}° and '}stays at the same point.")
        if grip_disp > 0.001:
            lines.append(f"  The gripper moves {grip_disp*1000:.0f} mm purely from rotation!")
            lines.append(f"  That's {grip_disp*1000:.0f} mm of 'fake translation' from lever-arm × rotation.")
        lines.append("")
        lines.append("  ── Decomposition ──")
        lines.append(f"  Translation term:           0.0 mm  (no camera translation)")
        lines.append(f"  Lever-arm term:         {grip_disp*1000:8.1f} mm  (100% of gripper path)")
        lines.append("")
        lines.append("  ── Key insight ──")
        lines.append("  Even with ZERO camera translation,")
        lines.append("  pure rotation × lever-arm offset")
        lines.append(f"  produces {grip_disp*1000:.0f} mm of gripper motion.")
        lines.append("")
        lines.append("  → The gripper does NOT 'travel the same")
        lines.append("    distance' as the camera. Not even close.")

        ax_text.clear()
        ax_text.axis("off")
        ax_text.text(
            0.0, 1.0, "\n".join(lines),
            transform=ax_text.transAxes,
            fontfamily="monospace",
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="left",
        )

        state["prev_theta"] = theta_rad
        state["prev_radius"] = radius_scale
        fig.canvas.draw_idle()

    slider_theta.on_changed(update)
    slider_radius.on_changed(update)

    # Initialize
    update(0)
    return fig, slider_theta, slider_radius


def main():
    print("=" * 55)
    print("  Lever-Arm Sandbox — Pure Rotation")
    print("=" * 55)
    print(f"  Lever arm:  {LEVER_ARM_2D[0]*1000:.1f}, {LEVER_ARM_2D[1]*1000:.1f} mm (X, Z in camera frame)")
    print(f"  Arm length: {np.linalg.norm(LEVER_ARM_2D)*1000:.1f} mm")
    print()
    print("  Drag the 'Rotation angle' slider to rotate the rigid body.")
    print("  Camera (green dot) = rotation center → stays still.")
    print("  Gripper (red square) = offset by lever arm → swings in an arc.")
    print("  Watch the right panel to see how much the gripper moves")
    print("  despite ZERO camera translation.")
    print()
    fig, _, _ = build_figure()
    plt.show(block=True)
    print("Window closed.")


if __name__ == "__main__":
    main()
