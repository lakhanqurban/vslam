"""
scene.py

Synthetic scene generation and rendering for the visual SLAM demo.
"""

from __future__ import annotations

import cv2
from math import cos, sin
import numpy as np

from geometry import CameraModel, RectangleObstacle


class SyntheticScene:
    def __init__(self, camera_model: CameraModel, seed: int = 7) -> None:
        self.camera_model = camera_model
        self.rng = np.random.default_rng(seed)
        self.landmarks_xy = self._create_landmarks()
        self.structures = self._create_structures()

    def _create_landmarks(self) -> np.ndarray:
        dense_grid_x, dense_grid_y = np.meshgrid(np.linspace(-22.0, 22.0, 12), np.linspace(-16.0, 16.0, 9))
        grid_points = np.column_stack([dense_grid_x.ravel(), dense_grid_y.ravel()])
        jitter = self.rng.normal(0.0, 0.5, size=grid_points.shape)
        scatter = self.rng.uniform(low=[-24.0, -18.0], high=[24.0, 18.0], size=(60, 2))
        return np.vstack([grid_points + jitter, scatter])

    def _create_structures(self):
        return [
            RectangleObstacle(-13.0, 10.0, 14.0, 9.0, 0.0),
            RectangleObstacle(9.0, -8.0, 9.0, 14.0, np.deg2rad(8.0)),
            RectangleObstacle(18.5, 12.0, 8.0, 6.0, np.deg2rad(-10.0)),
            RectangleObstacle(-2.5, -8.5, 7.0, 7.0, np.deg2rad(38.0)),
            RectangleObstacle(26.0, 0.0, 6.5, 18.0, 0.0),
            RectangleObstacle(12.0, 18.0, 8.5, 5.0, np.deg2rad(-24.0)),
        ]

    def render(self, pose_xyyaw: np.ndarray) -> np.ndarray:
        image = np.full((self.camera_model.height, self.camera_model.width), 232, dtype=np.uint8)

        for row in range(0, self.camera_model.height, 60):
            cv2.line(image, (0, row), (self.camera_model.width - 1, row), 224, 1, cv2.LINE_AA)
        for col in range(0, self.camera_model.width, 80):
            cv2.line(image, (col, 0), (col, self.camera_model.height - 1), 224, 1, cv2.LINE_AA)

        self._draw_structures(image, pose_xyyaw)
        self._draw_landmarks(image, pose_xyyaw)

        noise = self.rng.normal(0.0, 7.0, size=image.shape).astype(np.float32)
        blurred = cv2.GaussianBlur(image.astype(np.float32) + noise, (5, 5), 0)
        return np.clip(blurred, 0, 255).astype(np.uint8)

    def _draw_structures(self, image: np.ndarray, pose_xyyaw: np.ndarray) -> None:
        for obstacle in self.structures:
            corners_xy = obstacle.corners()
            pixels_xy, visible = self.camera_model.world_to_image(corners_xy, pose_xyyaw)
            if np.count_nonzero(visible) < 3:
                continue

            polygon = np.round(pixels_xy[visible]).astype(np.int32)
            cv2.fillPoly(image, [polygon], 90)
            cv2.polylines(image, [polygon], True, 25, 3, cv2.LINE_AA)

            for start_idx in range(len(polygon)):
                end_idx = (start_idx + 1) % len(polygon)
                cv2.line(image, tuple(polygon[start_idx]), tuple(polygon[end_idx]), 40, 1, cv2.LINE_AA)

    def _draw_landmarks(self, image: np.ndarray, pose_xyyaw: np.ndarray) -> None:
        pixels_xy, visible = self.camera_model.world_to_image(self.landmarks_xy, pose_xyyaw)
        for pixel_x, pixel_y in pixels_xy[visible]:
            point = (int(round(pixel_x)), int(round(pixel_y)))
            cv2.circle(image, point, 4, 35, -1, cv2.LINE_AA)
            cv2.circle(image, point, 1, 250, -1, cv2.LINE_AA)
