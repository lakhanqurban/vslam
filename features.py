"""
features.py

Feature detection and motion estimation helpers.
"""

from __future__ import annotations

from math import atan2, cos, sin

import cv2
import numpy as np


class VisualFeatures:
    def __init__(self, feature_type: str = "ORB") -> None:
        self.feature_type = feature_type.upper()
        if self.feature_type != "ORB":
            raise ValueError("The demo currently supports ORB features only.")
        self.detector = cv2.ORB_create(nfeatures=1500, fastThreshold=8)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def detect_and_compute(self, image: np.ndarray):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        if descriptors is None:
            descriptors = np.empty((0, 32), dtype=np.uint8)
        return keypoints, descriptors

    def match_features(self, descriptors1: np.ndarray, descriptors2: np.ndarray):
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        matches = self.matcher.match(descriptors1, descriptors2)
        return sorted(matches, key=lambda match: match.distance)


class PoseEstimator:
    def __init__(self, meters_per_pixel: float) -> None:
        self.meters_per_pixel = meters_per_pixel

    def estimate_motion(self, previous_keypoints, current_keypoints, matches):
        if len(matches) < 8:
            return None, 0.0

        previous_points = np.float32([previous_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        current_points = np.float32([current_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.995,
        )
        if affine is None:
            return None, 0.0

        inlier_ratio = float(inlier_mask.mean()) if inlier_mask is not None and len(inlier_mask) else 0.0

        rotation = affine[:2, :2]
        translation = affine[:2, 2]
        delta_yaw = atan2(rotation[1, 0], rotation[0, 0])

        delta_x = -translation[0] * self.meters_per_pixel
        delta_y = translation[1] * self.meters_per_pixel

        relative_motion = np.array(
            [
                [cos(delta_yaw), -sin(delta_yaw), delta_x],
                [sin(delta_yaw), cos(delta_yaw), delta_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        return relative_motion, inlier_ratio
