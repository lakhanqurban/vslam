"""
mapping.py

Trajectory storage, loop closure detection, and lightweight trajectory smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from geometry import wrap_angle_rad


@dataclass
class Keyframe:
    frame_index: int
    pose_xyyaw: np.ndarray
    inlier_ratio: float
    feature_count: int
    descriptors: np.ndarray


class MapBuilder:
    def __init__(self, keyframe_stride: int = 5) -> None:
        self.keyframe_stride = keyframe_stride
        self.keyframes: List[Keyframe] = []
        self.true_trajectory: List[np.ndarray] = []
        self.estimated_trajectory: List[np.ndarray] = []
        self.inlier_ratios: List[float] = []
        self.feature_counts: List[int] = []

    def add_frame(
        self,
        frame_index: int,
        true_pose: np.ndarray,
        estimated_pose: np.ndarray,
        feature_count: int,
        inlier_ratio: float,
        descriptors: np.ndarray,
    ) -> None:
        self.true_trajectory.append(true_pose.copy())
        self.estimated_trajectory.append(estimated_pose.copy())
        self.inlier_ratios.append(inlier_ratio)
        self.feature_counts.append(feature_count)

        if frame_index % self.keyframe_stride == 0:
            self.keyframes.append(
                Keyframe(
                    frame_index=frame_index,
                    pose_xyyaw=estimated_pose.copy(),
                    inlier_ratio=inlier_ratio,
                    feature_count=feature_count,
                    descriptors=descriptors.copy(),
                )
            )

    def poses_as_array(self) -> np.ndarray:
        return np.array(self.estimated_trajectory, dtype=float)


@dataclass
class LoopClosureEvent:
    frame_index: int
    matched_keyframe_index: int
    match_count: float
    match_ratio: float


class LoopClosureDetector:
    def __init__(
        self,
        min_frame_gap: int = 30,
        min_match_count: int = 40,
        min_match_ratio: float = 0.30,
    ) -> None:
        self.min_frame_gap = min_frame_gap
        self.min_match_count = min_match_count
        self.min_match_ratio = min_match_ratio
        self.events: List[LoopClosureEvent] = []

    def detect(self, frame_index: int, current_descriptors: np.ndarray, keyframes: Sequence[Keyframe]) -> Optional[LoopClosureEvent]:
        if len(current_descriptors) == 0:
            return None

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best_event = None
        for keyframe in keyframes:
            if frame_index - keyframe.frame_index < self.min_frame_gap:
                continue
            if len(keyframe.descriptors) == 0:
                continue

            matches = matcher.match(current_descriptors, keyframe.descriptors)
            match_count = len(matches)
            match_ratio = match_count / max(1, min(len(current_descriptors), len(keyframe.descriptors)))

            if match_count >= self.min_match_count and match_ratio >= self.min_match_ratio:
                best_event = LoopClosureEvent(frame_index, keyframe.frame_index, float(match_count), float(match_ratio))
                self.events.append(best_event)
                break

        return best_event


class BundleAdjustment:
    def __init__(self, window_size: int = 7) -> None:
        self.window_size = max(3, window_size | 1)

    def optimize(self, poses_xyyaw: np.ndarray) -> np.ndarray:
        if len(poses_xyyaw) < self.window_size:
            return poses_xyyaw.copy()

        smoothed = poses_xyyaw.copy()
        half_window = self.window_size // 2

        unwrapped_yaw = np.unwrap(smoothed[:, 2])
        for idx in range(len(smoothed)):
            start = max(0, idx - half_window)
            end = min(len(smoothed), idx + half_window + 1)
            smoothed[idx, 0] = np.mean(poses_xyyaw[start:end, 0])
            smoothed[idx, 1] = np.mean(poses_xyyaw[start:end, 1])
            smoothed[idx, 2] = wrap_angle_rad(np.mean(unwrapped_yaw[start:end]))

        drift_xy = smoothed[-1, :2] - smoothed[0, :2]
        drift_yaw = wrap_angle_rad(smoothed[-1, 2] - smoothed[0, 2])
        if len(smoothed) > 1:
            drift_scale = np.linspace(0.0, 1.0, len(smoothed))
            smoothed[:, 0] -= drift_scale * drift_xy[0]
            smoothed[:, 1] -= drift_scale * drift_xy[1]
            smoothed[:, 2] = np.array(
                [wrap_angle_rad(value - scale * drift_yaw) for value, scale in zip(smoothed[:, 2], drift_scale)],
                dtype=float,
            )

        return smoothed
