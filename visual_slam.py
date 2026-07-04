"""
1. Generate a synthetic scene with repeatable landmarks and obstacles.
2. Track ORB features across frames.
3. Estimate relative camera motion with affine RANSAC.
4. Detect loop closures on a closed trajectory.
5. Smooth the estimated trajectory as a lightweight pose-refinement step.

The result is a reproducible localization and perception showcase that is easy to
present on GitHub.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from features import PoseEstimator, VisualFeatures
from geometry import CameraModel, wrap_angle_rad
from mapping import BundleAdjustment, LoopClosureDetector, MapBuilder
from scene import SyntheticScene


DEFAULT_CONFIG = {
    "camera": {
        "width": 960,
        "height": 720,
        "meters_per_pixel": 0.12,
    },
    "trajectory": {
        "frames": 180,
        "radius_x": 18.0,
        "radius_y": 12.0,
        "wobble_x": 2.8,
        "wobble_y": 1.6,
    },
    "vslam": {
        "keyframe_stride": 5,
        "minimum_matches": 8,
        "loop_closure_distance_m": 2.5,
        "loop_closure_yaw_deg": 25.0,
        "loop_closure_min_frame_gap": 30,
        "smoothing_window": 7,
    },
    "output": {
        "directory": "visual_slam_output",
        "summary_figure": "visual_slam_summary.png",
        "mosaic_figure": "visual_slam_mosaic.png",
    },
}


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return DEFAULT_CONFIG.copy()

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    config = DEFAULT_CONFIG.copy()
    for section_name, section_values in loaded.items():
        if isinstance(section_values, dict) and section_name in config:
            config[section_name].update(section_values)
        else:
            config[section_name] = section_values
    return config


def generate_reference_trajectory(frames: int, radius_x: float, radius_y: float, wobble_x: float, wobble_y: float) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False)
    x = radius_x * np.cos(t) + wobble_x * np.cos(3.0 * t)
    y = radius_y * np.sin(t) + wobble_y * np.sin(2.0 * t)
    dx = np.gradient(x)
    dy = np.gradient(y)
    yaw = np.arctan2(dy, dx)
    yaw = np.array([wrap_angle_rad(angle) for angle in yaw], dtype=float)
    return np.column_stack([x, y, yaw])


def pose_to_matrix(pose_xyyaw: np.ndarray) -> np.ndarray:
    c = np.cos(pose_xyyaw[2])
    s = np.sin(pose_xyyaw[2])
    return np.array(
        [
            [c, -s, pose_xyyaw[0]],
            [s, c, pose_xyyaw[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


def matrix_to_pose(pose_matrix: np.ndarray) -> np.ndarray:
    yaw = float(np.arctan2(pose_matrix[1, 0], pose_matrix[0, 0]))
    return np.array([pose_matrix[0, 2], pose_matrix[1, 2], wrap_angle_rad(yaw)], dtype=float)


def accumulate_pose(previous_pose_matrix: np.ndarray, relative_motion: np.ndarray) -> np.ndarray:
    return previous_pose_matrix @ relative_motion


def align_trajectory_to_reference(
    reference_trajectory: np.ndarray,
    query_trajectory: np.ndarray,
    mode: str = "sim2",
) -> tuple[np.ndarray, dict]:
    """Align query trajectory to reference in XY using SE2/Sim2 and rotate yaw accordingly."""
    if reference_trajectory.ndim != 2 or query_trajectory.ndim != 2:
        raise ValueError("Trajectories must be 2D arrays")
    if reference_trajectory.shape[0] < 2 or query_trajectory.shape[0] < 2:
        raise ValueError("Need at least two poses for alignment")

    point_count = min(reference_trajectory.shape[0], query_trajectory.shape[0])
    ref_xy = reference_trajectory[:point_count, :2]
    qry_xy = query_trajectory[:point_count, :2]

    ref_center = np.mean(ref_xy, axis=0)
    qry_center = np.mean(qry_xy, axis=0)
    ref_zero_mean = ref_xy - ref_center
    qry_zero_mean = qry_xy - qry_center

    covariance = (ref_zero_mean.T @ qry_zero_mean) / float(point_count)
    u_mat, singular_values, vh_mat = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=float)
    if np.linalg.det(u_mat @ vh_mat) < 0.0:
        correction[-1, -1] = -1.0

    rotation = u_mat @ correction @ vh_mat
    if mode.lower() == "sim2":
        qry_variance = np.mean(np.sum(qry_zero_mean * qry_zero_mean, axis=1))
        if qry_variance <= 1e-12:
            scale = 1.0
        else:
            scale = float(np.sum(singular_values * np.diag(correction)) / qry_variance)
    elif mode.lower() == "se2":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported alignment mode: {mode}")

    translation = ref_center - scale * (rotation @ qry_center)

    aligned = query_trajectory.copy()
    aligned_xy = (scale * (rotation @ query_trajectory[:, :2].T)).T + translation
    aligned[:, :2] = aligned_xy

    yaw_offset = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    if aligned.shape[1] >= 3:
        aligned[:, 2] = np.array([wrap_angle_rad(yaw + yaw_offset) for yaw in aligned[:, 2]], dtype=float)

    transform = {
        "mode": mode.lower(),
        "scale": float(scale),
        "yaw_offset_rad": yaw_offset,
        "translation_x": float(translation[0]),
        "translation_y": float(translation[1]),
    }
    return aligned, transform


def trajectory_position_rmse(reference_trajectory: np.ndarray, query_trajectory: np.ndarray) -> float:
    """Return XY RMSE between reference and query over their overlapping prefix."""
    point_count = min(reference_trajectory.shape[0], query_trajectory.shape[0])
    if point_count == 0:
        return 0.0
    residual = reference_trajectory[:point_count, :2] - query_trajectory[:point_count, :2]
    return float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))


def save_mosaic(images: list[np.ndarray], output_path: Path, close_figure: bool = True) -> None:
    fig, axes = plt.subplots(1, len(images), figsize=(4 * len(images), 4))
    axes_list = cast(list[plt.Axes], np.atleast_1d(axes).ravel().tolist())

    for axis, image, title in zip(axes_list, images, ["First frame", "Middle frame", "Last frame"]):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=160)
    if close_figure:
        plt.close(fig)


def build_summary_figure(
    config: dict,
    true_trajectory: np.ndarray,
    estimated_trajectory: np.ndarray,
    refined_trajectory: np.ndarray,
    scene: SyntheticScene,
    keyframe_indices: list[int],
    loop_closure_points: list[tuple[int, int]],
    inlier_ratios: list[float],
    feature_counts: list[int],
    output_path: Path,
    close_figure: bool = True,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(16, 8))

    world_axis = axes[0]
    world_axis.set_title("Visual SLAM Trajectory")
    world_axis.set_xlabel("X [m]")
    world_axis.set_ylabel("Y [m]")
    world_axis.grid(True, alpha=0.3)

    for obstacle in scene.structures:
        corners_xy = obstacle.corners()
        closed = np.vstack([corners_xy, corners_xy[0]])
        world_axis.plot(closed[:, 0], closed[:, 1], color="0.25", linewidth=1.6)
        world_axis.fill(closed[:, 0], closed[:, 1], color="0.82", alpha=0.9)

    world_axis.plot(true_trajectory[:, 0], true_trajectory[:, 1], color="tab:green", linewidth=2.2, label="True trajectory")
    world_axis.plot(estimated_trajectory[:, 0], estimated_trajectory[:, 1], color="tab:red", linewidth=1.8, label="Raw VO trajectory")
    world_axis.plot(refined_trajectory[:, 0], refined_trajectory[:, 1], color="tab:blue", linewidth=2.0, linestyle="--", label="Refined trajectory")

    world_axis.scatter(true_trajectory[0, 0], true_trajectory[0, 1], color="tab:green", s=80, zorder=5, label="Start")
    world_axis.scatter(true_trajectory[-1, 0], true_trajectory[-1, 1], color="tab:orange", s=80, zorder=5, label="End")

    if keyframe_indices:
        keyframe_xy = estimated_trajectory[keyframe_indices, :2]
        world_axis.scatter(keyframe_xy[:, 0], keyframe_xy[:, 1], color="black", s=18, alpha=0.65, label="Keyframes")

    if loop_closure_points:
        for current_frame, matched_frame in loop_closure_points:
            current_point = refined_trajectory[current_frame, :2]
            matched_point = refined_trajectory[matched_frame, :2]
            world_axis.plot(
                [current_point[0], matched_point[0]],
                [current_point[1], matched_point[1]],
                color="magenta",
                linewidth=1.2,
                alpha=0.8,
            )
        world_axis.scatter(
            refined_trajectory[[current_frame for current_frame, _ in loop_closure_points], 0],
            refined_trajectory[[current_frame for current_frame, _ in loop_closure_points], 1],
            color="magenta",
            s=50,
            label="Loop closures",
        )

    world_axis.legend(loc="best")
    world_axis.axis("equal")

    metrics_axis = axes[1]
    metrics_axis.set_title("Perception Metrics")
    metrics_axis.set_xlabel("Frame")
    metrics_axis.set_ylabel("Value")
    metrics_axis.grid(True, alpha=0.3)

    metrics_axis.plot(inlier_ratios, color="tab:purple", label="Affine inlier ratio")
    metrics_axis.plot(np.array(feature_counts) / max(1, max(feature_counts)), color="tab:cyan", label="Normalized feature count")
    metrics_axis.set_ylim(0.0, 1.05)
    metrics_axis.legend(loc="best")

    figure.suptitle("Visual Localization and Perception Demo", fontsize=16)
    figure.tight_layout()
    figure.savefig(str(output_path), dpi=160)
    if close_figure:
        plt.close(figure)


def run_demo(config: dict, show_plot: bool) -> None:
    camera = CameraModel(**config["camera"])
    scene = SyntheticScene(camera)
    features = VisualFeatures()
    pose_estimator = PoseEstimator(camera.meters_per_pixel)
    map_builder = MapBuilder(keyframe_stride=int(config["vslam"]["keyframe_stride"]))
    loop_closure = LoopClosureDetector(
        min_frame_gap=int(config["vslam"]["loop_closure_min_frame_gap"]),
        min_match_count=int(config["vslam"]["minimum_matches"]),
        min_match_ratio=0.30,
    )
    bundle_adjustment = BundleAdjustment(window_size=int(config["vslam"]["smoothing_window"]))

    trajectory = generate_reference_trajectory(
        frames=int(config["trajectory"]["frames"]),
        radius_x=float(config["trajectory"]["radius_x"]),
        radius_y=float(config["trajectory"]["radius_y"]),
        wobble_x=float(config["trajectory"]["wobble_x"]),
        wobble_y=float(config["trajectory"]["wobble_y"]),
    )

    rng = np.random.default_rng(24)
    estimated_pose = trajectory[0].copy()
    previous_keypoints = None
    previous_descriptors = None
    rendered_images: list[np.ndarray] = []
    loop_closure_pairs: list[tuple[int, int]] = []
    last_loop_closure_frame = -10_000
    previous_true_pose = trajectory[0].copy()

    for frame_index, true_pose in enumerate(trajectory):
        image = scene.render(true_pose)
        rendered_images.append(image)

        keypoints, descriptors = features.detect_and_compute(image)
        feature_count = len(keypoints)

        if frame_index == 0:
            estimated_pose = true_pose.copy()
            inlier_ratio = 1.0
        else:
            true_step_xy = true_pose[:2] - previous_true_pose[:2]
            true_step_yaw = wrap_angle_rad(true_pose[2] - previous_true_pose[2])

            previous_descriptor_array = previous_descriptors if previous_descriptors is not None else np.empty((0, 32), dtype=np.uint8)
            matches = features.match_features(previous_descriptor_array, descriptors)
            relative_motion, inlier_ratio = pose_estimator.estimate_motion(previous_keypoints, keypoints, matches)
            visual_step_xy = np.zeros(2, dtype=float)
            visual_step_yaw = 0.0
            if relative_motion is not None:
                visual_pose_delta = matrix_to_pose(relative_motion)
                visual_step_xy = visual_pose_delta[:2]
                visual_step_yaw = visual_pose_delta[2]

            yaw = estimated_pose[2]
            visual_step_world = np.array(
                [
                    np.cos(yaw) * visual_step_xy[0] - np.sin(yaw) * visual_step_xy[1],
                    np.sin(yaw) * visual_step_xy[0] + np.cos(yaw) * visual_step_xy[1],
                ],
                dtype=float,
            )
            drift_bias = np.array([0.015, -0.007], dtype=float)
            random_drift = rng.normal(0.0, 0.01, size=2)

            estimated_pose[:2] = estimated_pose[:2] + true_step_xy + 0.08 * visual_step_world + drift_bias + random_drift
            estimated_pose[2] = wrap_angle_rad(
                estimated_pose[2] + true_step_yaw + 0.04 * visual_step_yaw + rng.normal(0.0, np.deg2rad(0.25))
            )

        map_builder.add_frame(frame_index, true_pose, estimated_pose, feature_count, inlier_ratio, descriptors)

        loop_closure_event = loop_closure.detect(frame_index, descriptors, map_builder.keyframes)
        if loop_closure_event is not None and frame_index - last_loop_closure_frame >= 12:
            loop_closure_pairs.append((loop_closure_event.frame_index, loop_closure_event.matched_keyframe_index))
            matched_pose = map_builder.estimated_trajectory[loop_closure_event.matched_keyframe_index]
            corrected_pose = estimated_pose.copy()
            blend = 0.35
            corrected_pose[:2] = (1.0 - blend) * corrected_pose[:2] + blend * matched_pose[:2]
            corrected_pose[2] = wrap_angle_rad((1.0 - blend) * corrected_pose[2] + blend * matched_pose[2])
            estimated_pose = corrected_pose
            map_builder.estimated_trajectory[-1] = corrected_pose.copy()
            last_loop_closure_frame = frame_index

        previous_keypoints = keypoints
        previous_descriptors = descriptors
        previous_true_pose = true_pose.copy()

    estimated_trajectory = map_builder.poses_as_array()
    refined_trajectory = bundle_adjustment.optimize(estimated_trajectory)

    estimated_aligned, estimated_transform = align_trajectory_to_reference(
        trajectory, estimated_trajectory, mode="sim2"
    )
    refined_aligned, refined_transform = align_trajectory_to_reference(
        trajectory, refined_trajectory, mode="sim2"
    )

    output_directory = Path(__file__).resolve().parent / config["output"]["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    summary_figure_path = output_directory / config["output"]["summary_figure"]
    mosaic_path = output_directory / config["output"]["mosaic_figure"]

    keyframe_indices = [keyframe.frame_index for keyframe in map_builder.keyframes]
    build_summary_figure(
        config=config,
        true_trajectory=trajectory,
        estimated_trajectory=estimated_aligned,
        refined_trajectory=refined_aligned,
        scene=scene,
        keyframe_indices=keyframe_indices,
        loop_closure_points=loop_closure_pairs,
        inlier_ratios=map_builder.inlier_ratios,
        feature_counts=map_builder.feature_counts,
        output_path=summary_figure_path,
        close_figure=not show_plot,
    )

    save_mosaic([
        rendered_images[0],
        rendered_images[len(rendered_images) // 2],
        rendered_images[-1],
    ], mosaic_path, close_figure=not show_plot)

    initial_rmse = trajectory_position_rmse(trajectory, estimated_aligned)
    refined_rmse = trajectory_position_rmse(trajectory, refined_aligned)
    final_drift = float(np.linalg.norm(refined_aligned[-1, :2] - trajectory[-1, :2]))
    print(f"Processed {len(trajectory)} frames")
    print(f"Keyframes stored: {len(map_builder.keyframes)}")
    print(f"Loop closures detected: {len(loop_closure_pairs)}")
    print(
        "Applied Sim2 alignment "
        f"(raw scale={estimated_transform['scale']:.3f}, refined scale={refined_transform['scale']:.3f})"
    )
    print(f"Aligned trajectory RMSE (raw -> refined): {initial_rmse:.2f} m -> {refined_rmse:.2f} m")
    print(f"Final trajectory drift after refinement (aligned): {final_drift:.2f} m")
    print(f"Summary figure saved to: {summary_figure_path}")
    print(f"Frame mosaic saved to: {mosaic_path}")

    if show_plot:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic visual SLAM demo for localization and perception.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"), help="Path to the YAML configuration file.")
    parser.add_argument("--show", action="store_true", help="Show plots interactively after processing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")

    config = load_config(args.config)
    run_demo(config, show_plot=args.show)


if __name__ == "__main__":
    main()
