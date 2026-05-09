#!/usr/bin/env python
"""Diagnostic script for SmolVLA strawberry picking predictions.

Runs inference on selected dataset episodes, compares predicted vs ground truth
action chunks, produces per-dimension error metrics, matplotlib plots,
normalization consistency checks, and a summary text report.

Usage:
  cd /home/hls/codes/lerobot_piper_sroi && uv run --directory lerobot python \
      lerobot_robot_piper/diagnose_predictions.py \
      --pretrained_path outputs/smolvla_umi_strawberry_50k/checkpoints/050000/pretrained_model \
      --dataset_root Datasets/sroi_piper_strawberry_picking \
      --episodes 0 1 2 \
      --output_dir outputs/diagnostic_strawberry_50k
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DIM_NAMES = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper"]
POS_DIMS = [0, 1, 2]
ROT_DIMS = [3, 4, 5]
GRIP_DIM = 6


def load_pipeline(pretrained_path: str, dataset_root: str, device: str = "cuda"):
    """Load dataset with recomputed stats, policy, pre/post processors.

    Returns (ds_meta, ds_dt, policy, preprocessor, postprocessor, dim_names)
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.dataset_tools import recompute_stats
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies import make_policy, make_pre_post_processors

    pretrained_path = str((PROJECT_ROOT / pretrained_path).resolve())
    dataset_root = str((PROJECT_ROOT / dataset_root).resolve())
    repo_id = Path(dataset_root).name

    logger.info(f"Loading dataset metadata from {dataset_root}...")
    ds_meta = LeRobotDatasetMetadata(repo_id, root=dataset_root)

    # Load train_config.json for exact training flags
    config_path = Path(pretrained_path) / "train_config.json"
    if config_path.exists():
        with open(config_path) as f:
            train_config = json.load(f)
        policy_config = train_config.get("policy", {})
        logger.info(f"Loaded train_config.json from {config_path}")
    else:
        policy_config = {}
        logger.warning(f"No train_config.json found at {config_path}, using defaults")

    # Recompute stats with exact same flags as training
    logger.info("Recomputing dataset stats (relative_action, relative_state, derive_state)...")
    ds = LeRobotDataset(repo_id, root=dataset_root)
    ds = recompute_stats(
        ds,
        num_workers=2,
        relative_action=True,
        relative_exclude_joints=["gripper"],
        relative_state=True,
        relative_exclude_state_joints=["gripper"],
        state_obs_steps=2,
        derive_state_from_action=True,
    )
    ds_meta = ds.meta

    # Build SmolVLA config matching training
    cfg = SmolVLAConfig(
        derive_state_from_action=policy_config.get("derive_state_from_action", True),
        use_relative_actions=policy_config.get("use_relative_actions", True),
        relative_exclude_joints=policy_config.get("relative_exclude_joints", ["gripper"]),
        relative_exclude_state_joints=policy_config.get("relative_exclude_state_joints", ["gripper"]),
        device=device,
        resize_imgs_with_padding=tuple(policy_config.get("resize_imgs_with_padding", (512, 512))),
        freeze_vision_encoder=policy_config.get("freeze_vision_encoder", True),
        train_expert_only=policy_config.get("train_expert_only", True),
        train_state_proj=policy_config.get("train_state_proj", True),
        load_vlm_weights=False,
        push_to_hub=False,
        pretrained_path=pretrained_path,
    )

    logger.info(f"Config: derive_state={cfg.derive_state_from_action}, "
                f"relative_actions={cfg.use_relative_actions}")

    # Build policy and processors
    policy = make_policy(cfg=cfg, ds_meta=ds_meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=pretrained_path,
        dataset_stats=ds_meta.stats,
    )

    logger.info(f"Preprocessor: {len(preprocessor.steps)} steps")
    for i, step in enumerate(preprocessor.steps):
        logger.info(f"  [{i}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")
    logger.info(f"Postprocessor: {len(postprocessor.steps)} steps")
    for i, step in enumerate(postprocessor.steps):
        logger.info(f"  [{i}] {type(step).__name__}: enabled={getattr(step, 'enabled', 'N/A')}")

    # Resolve delta timestamps and load dataset with them
    dt = resolve_delta_timestamps(cfg, ds_meta)
    ds_dt = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=dt)

    dim_names = ds_meta.features["action"]["names"]
    if isinstance(dim_names, dict):
        dim_names = list(dim_names.values())
        if len(dim_names) > 0 and isinstance(dim_names[0], list):
            dim_names = dim_names[0]
    if not dim_names or dim_names == ["axes"]:
        n_action = ds_meta.features["action"]["shape"][0]
        dim_names = DIM_NAMES[:n_action]

    return ds_meta, ds_dt, policy, preprocessor, postprocessor, dim_names


def get_episode_frame_ranges(ds) -> dict[int, tuple[int, int]]:
    """Return {episode_index: (start_frame_idx, end_frame_idx)} for the dataset."""
    ranges = {}
    for ep_idx in range(ds.num_episodes):
        ep = ds.meta.episodes[ep_idx]
        start = ep["dataset_from_index"]
        end = ep["dataset_to_index"]
        ranges[ep_idx] = (start, end)
    return ranges


def check_normalization_consistency(preprocessor, postprocessor, ds_meta) -> dict:
    """Verify normalization stats consistency between preprocessor and dataset."""
    result = {
        "consistent": True,
        "issues": [],
        "preproc_action_stats": None,
        "preproc_state_stats": None,
        "dataset_action_stats": None,
        "dataset_state_stats": None,
    }

    # Extract stats from preprocessor normalizer
    for step in preprocessor.steps:
        step_type = type(step).__name__
        if "Normalizer" in step_type and hasattr(step, "_tensor_stats"):
            stats = step._tensor_stats
            if "action" in stats:
                result["preproc_action_stats"] = {
                    "mean": stats["action"]["mean"].cpu().numpy().tolist(),
                    "std": stats["action"]["std"].cpu().numpy().tolist(),
                }
            if "observation.state" in stats:
                result["preproc_state_stats"] = {
                    "mean": stats["observation.state"]["mean"].cpu().numpy().tolist(),
                    "std": stats["observation.state"]["std"].cpu().numpy().tolist(),
                }

    # Extract stats from postprocessor unnormalizer
    for step in postprocessor.steps:
        step_type = type(step).__name__
        if "Unnormalizer" in step_type and hasattr(step, "_tensor_stats"):
            stats = step._tensor_stats
            if "action" in stats:
                postproc_action = {
                    "mean": stats["action"]["mean"].cpu().numpy().tolist(),
                    "std": stats["action"]["std"].cpu().numpy().tolist(),
                }
                # Check preproc vs postproc consistency
                if result["preproc_action_stats"] is not None:
                    pre_mean = np.array(result["preproc_action_stats"]["mean"])
                    post_mean = np.array(postproc_action["mean"])
                    if not np.allclose(pre_mean, post_mean, atol=1e-6):
                        result["consistent"] = False
                        result["issues"].append(
                            f"Preproc/postproc action mean mismatch: "
                            f"pre={pre_mean.tolist()} vs post={post_mean.tolist()}"
                        )
                    pre_std = np.array(result["preproc_action_stats"]["std"])
                    post_std = np.array(postproc_action["std"])
                    if not np.allclose(pre_std, post_std, atol=1e-6):
                        result["consistent"] = False
                        result["issues"].append(
                            f"Preproc/postproc action std mismatch: "
                            f"pre={pre_std.tolist()} vs post={post_std.tolist()}"
                        )

    # Compare against dataset stats
    if ds_meta.stats and "action" in ds_meta.stats:
        ds_action = ds_meta.stats["action"]
        result["dataset_action_stats"] = {
            "mean": ds_action["mean"].tolist() if hasattr(ds_action["mean"], "tolist") else ds_action["mean"],
            "std": ds_action["std"].tolist() if hasattr(ds_action["std"], "tolist") else ds_action["std"],
        }

        if result["preproc_action_stats"] is not None:
            pre_mean = np.array(result["preproc_action_stats"]["mean"])
            ds_mean = np.array(result["dataset_action_stats"]["mean"])
            ds_mean_flat = ds_mean.flatten()
            pre_mean_flat = pre_mean.flatten()
            if len(pre_mean_flat) == len(ds_mean_flat):
                if not np.allclose(pre_mean_flat, ds_mean_flat, atol=1e-4):
                    result["consistent"] = False
                    result["issues"].append(
                        f"Preproc/dataset action mean mismatch: "
                        f"pre={pre_mean_flat.tolist()} vs ds={ds_mean_flat.tolist()}"
                    )

    if ds_meta.stats and "observation.state" in ds_meta.stats:
        ds_state = ds_meta.stats["observation.state"]
        result["dataset_state_stats"] = {
            "mean": ds_state["mean"].tolist() if hasattr(ds_state["mean"], "tolist") else ds_state["mean"],
            "std": ds_state["std"].tolist() if hasattr(ds_state["std"], "tolist") else ds_state["std"],
        }

    # Diagnostic: relative action mean should be ~0 for pos/rot dims
    if result["preproc_action_stats"] is not None:
        mean = np.array(result["preproc_action_stats"]["mean"])
        # For relative actions, pos/rot means should be near zero (gripper excluded from relative)
        for i, name in enumerate(DIM_NAMES[:len(mean.flatten())]):
            idx = i % len(mean.flatten())
            val = mean.flatten()[idx]
            if name != "gripper" and abs(val) > 0.01:
                result["issues"].append(
                    f"Relative action mean for {name} = {val:.4f} (expected ~0 for relative actions)"
                )

    return result


def run_episode_inference(
    policy, preprocessor, postprocessor, ds_dt,
    episode_idx: int, frame_range: tuple[int, int],
    device: str, sample_rate: int = 1, max_frames: int = 0,
) -> dict:
    """Run inference on all (or sampled) frames of one episode.

    Returns dict with pred_actions, gt_actions, pred_normalized_rel, cached_states, etc.
    """
    # Reset pipeline state for fresh episode
    policy.reset()
    for step in preprocessor.steps:
        if hasattr(step, "reset"):
            step.reset()

    start, end = frame_range
    frame_indices = list(range(start, end, sample_rate))
    if max_frames > 0 and len(frame_indices) > max_frames:
        frame_indices = frame_indices[:max_frames]

    all_pred = []
    all_gt = []
    all_pred_norm_rel = []
    all_cached_states = []
    infer_times = []

    for i, frame_idx in enumerate(frame_indices):
        raw = ds_dt[frame_idx]
        batch = {k: v.unsqueeze(0) if hasattr(v, "unsqueeze") else [v]
                 for k, v in raw.items()}
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        gt_action_raw = batch["action"].clone()  # (1, 51, 7)

        t0 = time.perf_counter()
        with torch.no_grad():
            processed = preprocessor(batch)
            pred_actions = policy.predict_action_chunk(processed)
            pred_norm_rel = pred_actions[0].cpu().numpy()  # (50, 7) in normalized relative space
            pred_abs = postprocessor(pred_actions)
        infer_ms = (time.perf_counter() - t0) * 1000
        infer_times.append(infer_ms)

        pred_np = pred_abs[0].cpu().numpy()  # (50, 7) absolute
        gt_np = gt_action_raw[0, 1:].cpu().numpy()  # (50, 7) skip DeriveState frame

        all_pred.append(pred_np)
        all_gt.append(gt_np)
        all_pred_norm_rel.append(pred_norm_rel)

        # Extract cached state from RelativeActionsProcessorStep
        cached_state = None
        for step in preprocessor.steps:
            if hasattr(step, "_last_state") and step._last_state is not None:
                cached_state = step._last_state[0].cpu().numpy() if hasattr(step._last_state, "cpu") else step._last_state
                break
        if cached_state is not None:
            all_cached_states.append(cached_state.copy())

        if (i + 1) % 20 == 0 or i == len(frame_indices) - 1:
            logger.info(f"  Episode {episode_idx}: frame {i+1}/{len(frame_indices)} | "
                        f"infer={infer_ms:.0f}ms")

    return {
        "episode_idx": episode_idx,
        "pred_actions": np.array(all_pred),      # (N, 50, 7)
        "gt_actions": np.array(all_gt),           # (N, 50, 7)
        "pred_normalized_rel": np.array(all_pred_norm_rel),  # (N, 50, 7)
        "cached_states": np.array(all_cached_states) if all_cached_states else None,
        "frame_indices": frame_indices,
        "infer_times_ms": infer_times,
    }


def compute_metrics(pred_actions: np.ndarray, gt_actions: np.ndarray) -> dict:
    """Compute per-dimension error metrics. pred/gt shape: (N, 50, 7)."""
    n_dims = pred_actions.shape[2]
    dim_names = DIM_NAMES[:n_dims]

    error = pred_actions - gt_actions
    abs_error = np.abs(error)

    metrics = {
        "per_dim_mae": {},
        "per_dim_rmse": {},
        "per_dim_max_error": {},
        "per_dim_mean_bias": {},
        "per_dim_pred_range": {},
        "per_dim_gt_range": {},
        "per_dim_t0_mae": {},
        "chunk_error_growth": None,
    }

    for d in range(n_dims):
        name = dim_names[d]
        metrics["per_dim_mae"][name] = float(abs_error[:, :, d].mean())
        metrics["per_dim_rmse"][name] = float(np.sqrt((error[:, :, d] ** 2).mean()))
        metrics["per_dim_max_error"][name] = float(abs_error[:, :, d].max())
        metrics["per_dim_mean_bias"][name] = float(error[:, :, d].mean())
        metrics["per_dim_pred_range"][name] = (
            float(pred_actions[:, :, d].min()),
            float(pred_actions[:, :, d].max()),
        )
        metrics["per_dim_gt_range"][name] = (
            float(gt_actions[:, :, d].min()),
            float(gt_actions[:, :, d].max()),
        )
        # t=0 error only
        metrics["per_dim_t0_mae"][name] = float(abs_error[:, 0, d].mean())

    # Error growth along chunk (average over all frames)
    chunk_growth = abs_error.mean(axis=0)  # (50, 7)
    metrics["chunk_error_growth"] = chunk_growth

    # Position/rotation/gripper aggregates
    pos_mae = np.mean([metrics["per_dim_mae"][dim_names[d]] for d in POS_DIMS if d < n_dims])
    rot_mae = np.mean([metrics["per_dim_mae"][dim_names[d]] for d in ROT_DIMS if d < n_dims])
    metrics["pos_mae_m"] = pos_mae
    metrics["pos_mae_mm"] = pos_mae * 1000
    metrics["rot_mae_rad"] = rot_mae
    metrics["overall_mae"] = float(abs_error.mean())

    return metrics


def plot_episode_trajectories(
    pred_actions: np.ndarray, gt_actions: np.ndarray,
    episode_idx: int, output_dir: Path, dim_names: list,
) -> list[Path]:
    """Generate per-dimension trajectory comparison plots."""
    n_dims = pred_actions.shape[2]
    n_frames = pred_actions.shape[0]
    names = dim_names if dim_names else DIM_NAMES[:n_dims]
    saved = []

    # Figure 1: per-dimension pred vs gt (t=0 actions over the episode)
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    x = np.arange(n_frames)
    for d in range(n_dims):
        ax = axes[d]
        gt_t0 = gt_actions[:, 0, d]
        pred_t0 = pred_actions[:, 0, d]
        gt_lo = gt_actions[:, :, d].min(axis=1)
        gt_hi = gt_actions[:, :, d].max(axis=1)
        pred_lo = pred_actions[:, :, d].min(axis=1)
        pred_hi = pred_actions[:, :, d].max(axis=1)

        ax.fill_between(x, gt_lo, gt_hi, alpha=0.15, color="blue", label="GT chunk range")
        ax.fill_between(x, pred_lo, pred_hi, alpha=0.15, color="red", label="Pred chunk range")
        ax.plot(x, gt_t0, "b-", linewidth=1.5, label="GT t=0")
        ax.plot(x, pred_t0, "r-", linewidth=1.5, alpha=0.8, label="Pred t=0")
        ax.set_ylabel(names[d])
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"Episode {episode_idx}: Predicted vs Ground Truth Actions", fontsize=13)
    fig.tight_layout()
    path = output_dir / f"episode_{episode_idx:03d}_trajectories.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # Figure 2: t=0 absolute error over the episode
    fig, axes = plt.subplots(n_dims, 1, figsize=(14, 3 * n_dims), sharex=True)
    if n_dims == 1:
        axes = [axes]

    for d in range(n_dims):
        ax = axes[d]
        err = np.abs(pred_actions[:, 0, d] - gt_actions[:, 0, d])
        ax.bar(x, err, color="coral", alpha=0.7)
        ax.set_ylabel(f"|err| {names[d]}")
        ax.grid(True, alpha=0.3)
        mean_err = err.mean()
        ax.axhline(mean_err, color="red", linestyle="--", linewidth=1, label=f"mean={mean_err:.4f}")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Frame index")
    fig.suptitle(f"Episode {episode_idx}: t=0 Absolute Error per Frame", fontsize=13)
    fig.tight_layout()
    path = output_dir / f"episode_{episode_idx:03d}_error_per_frame.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    # Figure 3: 3D trajectory (XYZ)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    gt_xyz = gt_actions[:, 0, :3]
    pred_xyz = pred_actions[:, 0, :3]

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], "b-o", markersize=3,
            linewidth=1.5, label="GT", alpha=0.9)
    ax.plot(pred_xyz[:, 0], pred_xyz[:, 1], pred_xyz[:, 2], "r--o", markersize=3,
            linewidth=1.5, label="Pred", alpha=0.7)

    # Start/end markers
    ax.scatter(*gt_xyz[0], color="blue", s=100, marker="*", label="GT start")
    ax.scatter(*gt_xyz[-1], color="blue", s=100, marker="s", label="GT end")
    ax.scatter(*pred_xyz[0], color="red", s=100, marker="*", label="Pred start")
    ax.scatter(*pred_xyz[-1], color="red", s=100, marker="s", label="Pred end")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Episode {episode_idx}: 3D EE Trajectory")
    ax.legend(fontsize=8)
    path = output_dir / f"episode_{episode_idx:03d}_3d_trajectory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved.append(path)

    return saved


def plot_chunk_error_growth(
    chunk_error: np.ndarray, output_dir: Path, dim_names: list,
) -> Path:
    """Plot error growth along the 50-step prediction horizon."""
    n_dims = chunk_error.shape[1]
    names = dim_names if dim_names else DIM_NAMES[:n_dims]
    horizon = chunk_error.shape[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    for d in range(n_dims):
        ax.plot(range(horizon), chunk_error[:, d], "-o", markersize=2,
                linewidth=1.5, label=names[d])

    ax.set_xlabel("Chunk timestep")
    ax.set_ylabel("MAE")
    ax.set_title("Error Growth Along Prediction Horizon (Aggregate)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    path = output_dir / "chunk_error_growth.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_action_ranges(
    pred_actions: np.ndarray, gt_actions: np.ndarray,
    output_dir: Path, dim_names: list,
) -> Path:
    """Bar chart comparing pred range vs gt range per dimension."""
    n_dims = pred_actions.shape[2]
    names = dim_names if dim_names else DIM_NAMES[:n_dims]

    pred_ranges = [pred_actions[:, :, d].max() - pred_actions[:, :, d].min() for d in range(n_dims)]
    gt_ranges = [gt_actions[:, :, d].max() - gt_actions[:, :, d].min() for d in range(n_dims)]

    x = np.arange(n_dims)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, gt_ranges, width, label="Ground Truth", color="steelblue")
    ax.bar(x + width / 2, pred_ranges, width, label="Predicted", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Range (absolute)")
    ax.set_title("Action Range: Predicted vs Ground Truth")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    path = output_dir / "action_ranges.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_normalized_predictions(
    pred_norm_rel: np.ndarray, output_dir: Path, episode_idx: int, dim_names: list,
) -> Path:
    """Plot the model's raw normalized relative output distribution."""
    n_dims = pred_norm_rel.shape[2]
    names = dim_names if dim_names else DIM_NAMES[:n_dims]

    fig, axes = plt.subplots(1, n_dims, figsize=(4 * n_dims, 4))
    if n_dims == 1:
        axes = [axes]

    for d in range(n_dims):
        ax = axes[d]
        vals = pred_norm_rel[:, :, d].flatten()
        ax.hist(vals, bins=50, color="steelblue", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(f"{names[d]}\nmean={vals.mean():.3f} std={vals.std():.3f}")
        ax.set_xlabel("Normalized value")

    fig.suptitle(f"Episode {episode_idx}: Model Output Distribution (Normalized Relative Space)")
    fig.tight_layout()
    path = output_dir / f"episode_{episode_idx:03d}_norm_dist.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_report(
    all_metrics: list[dict], norm_check: dict, output_dir: Path, dim_names: list,
) -> Path:
    """Write a human-readable text summary report."""
    lines = []
    lines.append("=" * 70)
    lines.append("SmolVLA Prediction Diagnostic Report")
    lines.append("=" * 70)
    lines.append("")

    # Normalization check
    lines.append("--- Normalization Consistency Check ---")
    if norm_check["consistent"]:
        lines.append("  PASS: Stats are consistent")
    else:
        lines.append("  FAIL: Inconsistencies detected!")
    for issue in norm_check.get("issues", []):
        lines.append(f"  ! {issue}")
    lines.append("")

    if norm_check.get("preproc_action_stats"):
        lines.append("  Preprocessor action stats:")
        mean = np.array(norm_check["preproc_action_stats"]["mean"]).flatten()
        std = np.array(norm_check["preproc_action_stats"]["std"]).flatten()
        for i, name in enumerate(dim_names[:len(mean)]):
            lines.append(f"    {name}: mean={mean[i]:.6f}, std={std[i]:.6f}")
    lines.append("")

    if norm_check.get("dataset_action_stats"):
        lines.append("  Dataset action stats:")
        mean = np.array(norm_check["dataset_action_stats"]["mean"]).flatten()
        std = np.array(norm_check["dataset_action_stats"]["std"]).flatten()
        for i, name in enumerate(dim_names[:len(mean)]):
            lines.append(f"    {name}: mean={mean[i]:.6f}, std={std[i]:.6f}")
    lines.append("")

    # Per-episode metrics
    lines.append("--- Per-Episode Metrics ---")
    for m in all_metrics:
        ep = m.get("episode_idx", "?")
        lines.append(f"  Episode {ep}:")
        lines.append(f"    Position MAE: {m['pos_mae_mm']:.2f} mm")
        lines.append(f"    Rotation MAE: {m['rot_mae_rad']:.4f} rad")
        lines.append(f"    Overall MAE:  {m['overall_mae']:.6f}")
        lines.append(f"    t=0 Position MAE: "
                     f"{np.mean([m['per_dim_t0_mae'].get(dim_names[d], 0) for d in POS_DIMS if d < len(dim_names)]) * 1000:.2f} mm")
        lines.append("")
        lines.append("    Per-dimension (MAE / RMSE / max-err / bias):")
        for name in dim_names:
            if name in m["per_dim_mae"]:
                lines.append(
                    f"      {name:15s}: MAE={m['per_dim_mae'][name]:.6f}  "
                    f"RMSE={m['per_dim_rmse'][name]:.6f}  "
                    f"max={m['per_dim_max_error'][name]:.6f}  "
                    f"bias={m['per_dim_mean_bias'][name]:+.6f}"
                )
        lines.append("")

        # Range comparison
        lines.append("    Action range comparison (GT → Pred):")
        for name in dim_names:
            if name in m["per_dim_gt_range"]:
                gt_lo, gt_hi = m["per_dim_gt_range"][name]
                pred_lo, pred_hi = m["per_dim_pred_range"][name]
                lines.append(
                    f"      {name:15s}: GT=[{gt_lo:.4f}, {gt_hi:.4f}]  "
                    f"Pred=[{pred_lo:.4f}, {pred_hi:.4f}]"
                )
        lines.append("")

    # Aggregate
    if len(all_metrics) > 1:
        lines.append("--- Aggregate Across Episodes ---")
        avg_pos = np.mean([m["pos_mae_mm"] for m in all_metrics])
        avg_rot = np.mean([m["rot_mae_rad"] for m in all_metrics])
        avg_overall = np.mean([m["overall_mae"] for m in all_metrics])
        lines.append(f"  Avg Position MAE: {avg_pos:.2f} mm")
        lines.append(f"  Avg Rotation MAE: {avg_rot:.4f} rad")
        lines.append(f"  Avg Overall MAE:  {avg_overall:.6f}")
        lines.append("")

    # Diagnosis hints
    lines.append("--- Diagnosis Hints ---")
    if not norm_check["consistent"]:
        lines.append("  [!] Normalization stats mismatch detected. This is likely the root cause.")
        lines.append("      Check that recompute_stats was called with the same flags as training.")
    for m in all_metrics:
        ep = m.get("episode_idx", "?")
        for name in dim_names:
            if name in m["per_dim_pred_range"]:
                pred_lo, pred_hi = m["per_dim_pred_range"][name]
                gt_lo, gt_hi = m["per_dim_gt_range"][name]
                pred_range = pred_hi - pred_lo
                gt_range = gt_hi - gt_lo
                if gt_range > 0.001 and pred_range < gt_range * 0.1:
                    lines.append(
                        f"  [!] Ep{ep} {name}: Prediction range ({pred_range:.6f}) "
                        f"much smaller than GT ({gt_range:.6f}). Model may be outputting near-constant values."
                    )
                if name != "gripper" and abs(m["per_dim_mean_bias"].get(name, 0)) > 0.01:
                    lines.append(
                        f"  [i] Ep{ep} {name}: Systematic bias = {m['per_dim_mean_bias'][name]:+.6f}"
                    )
    lines.append("")

    report_text = "\n".join(lines)
    path = output_dir / "report.txt"
    with open(path, "w") as f:
        f.write(report_text)
    logger.info(f"Report written to {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Diagnose SmolVLA predictions vs ground truth")
    parser.add_argument("--pretrained_path", type=str, required=True,
                        help="Path to trained checkpoint directory")
    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Dataset root path")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2],
                        help="Episode indices to evaluate")
    parser.add_argument("--output_dir", type=str, default="outputs/diagnostic_strawberry_50k",
                        help="Output directory for plots and report")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--sample_rate", type=int, default=1,
                        help="Evaluate every Nth frame (1=all)")
    parser.add_argument("--max_frames_per_episode", type=int, default=0,
                        help="Max frames per episode (0=all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Episodes to evaluate: {args.episodes}")

    # 1. Load pipeline
    ds_meta, ds_dt, policy, preprocessor, postprocessor, dim_names = \
        load_pipeline(args.pretrained_path, args.dataset_root, args.device)

    logger.info(f"Dataset: {ds_dt.num_episodes} episodes, {ds_dt.num_frames} frames")
    logger.info(f"Action dims: {dim_names}")

    # 2. Check normalization
    norm_check = check_normalization_consistency(preprocessor, postprocessor, ds_meta)
    if norm_check["consistent"]:
        logger.info("Normalization check: PASS")
    else:
        logger.warning("Normalization check: FAIL")
        for issue in norm_check["issues"]:
            logger.warning(f"  {issue}")

    # 3. Get episode ranges
    ep_ranges = get_episode_frame_ranges(ds_dt)
    logger.info(f"Available episodes: {sorted(ep_ranges.keys())[:20]}...")

    # 4. Run inference per episode
    all_metrics = []
    all_pred = []
    all_gt = []

    for ep_idx in args.episodes:
        if ep_idx not in ep_ranges:
            logger.warning(f"Episode {ep_idx} not in dataset, skipping")
            continue

        logger.info(f"\nProcessing episode {ep_idx} "
                     f"({ep_ranges[ep_idx][1] - ep_ranges[ep_idx][0]} frames)...")

        result = run_episode_inference(
            policy, preprocessor, postprocessor, ds_dt,
            ep_idx, ep_ranges[ep_idx], args.device,
            sample_rate=args.sample_rate,
            max_frames=args.max_frames_per_episode,
        )

        metrics = compute_metrics(result["pred_actions"], result["gt_actions"])
        metrics["episode_idx"] = ep_idx
        metrics["avg_infer_ms"] = np.mean(result["infer_times_ms"])
        all_metrics.append(metrics)
        all_pred.append(result["pred_actions"])
        all_gt.append(result["gt_actions"])

        # Per-episode plots
        plot_episode_trajectories(
            result["pred_actions"], result["gt_actions"],
            ep_idx, plots_dir, dim_names,
        )
        plot_normalized_predictions(
            result["pred_normalized_rel"], plots_dir, ep_idx, dim_names,
        )

        # Save raw arrays
        save_data = {
            "pred": result["pred_actions"],
            "gt": result["gt_actions"],
            "pred_norm_rel": result["pred_normalized_rel"],
        }
        if result["cached_states"] is not None:
            save_data["cached_states"] = result["cached_states"]
        np.savez(output_dir / f"episode_{ep_idx:03d}.npz", **save_data)

        logger.info(f"  Episode {ep_idx} done: pos MAE={metrics['pos_mae_mm']:.2f}mm, "
                     f"rot MAE={metrics['rot_mae_rad']:.4f}rad, "
                     f"avg infer={metrics['avg_infer_ms']:.0f}ms")

    # 5. Aggregate plots
    if all_pred:
        all_pred_cat = np.concatenate(all_pred, axis=0)
        all_gt_cat = np.concatenate(all_gt, axis=0)
        agg_metrics = compute_metrics(all_pred_cat, all_gt_cat)

        plot_chunk_error_growth(agg_metrics["chunk_error_growth"], plots_dir, dim_names)
        plot_action_ranges(all_pred_cat, all_gt_cat, plots_dir, dim_names)

    # 6. Report
    generate_report(all_metrics, norm_check, output_dir, dim_names)
    logger.info(f"\nDone! Check {output_dir}/report.txt for summary.")


if __name__ == "__main__":
    main()
