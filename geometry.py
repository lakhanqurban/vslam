"""
geometry.py

Shared geometry helpers for the visual SLAM demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin
from typing import Tuple

import numpy as np


def wrap_angle_rad(angle_rad: float) -> float:
    return (angle_rad + np.pi) % (2.0 * np.pi) - np.pi


def rotation_matrix_2d(angle_rad: float) -> np.ndarray:
    c = cos(angle_rad)
    s = sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


@dataclass
class CameraModel:
    width: int = 960
    height: int = 720
    meters_per_pixel: float = 0.12

    def __post_init__(self) -> None:
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0

    def world_to_image(self, points_xy: np.ndarray, pose_xyyaw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if points_xy.size == 0:
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=bool)

        dx = points_xy[:, 0] - pose_xyyaw[0]
        dy = points_xy[:, 1] - pose_xyyaw[1]
        yaw = pose_xyyaw[2]

        local_x = cos(yaw) * dx + sin(yaw) * dy
        local_y = -sin(yaw) * dx + cos(yaw) * dy

        pixel_x = self.cx + local_x / self.meters_per_pixel
        pixel_y = self.cy - local_y / self.meters_per_pixel

        visible = (
            (pixel_x >= 0)
            & (pixel_x < self.width)
            & (pixel_y >= 0)
            & (pixel_y < self.height)
        )
        return np.column_stack([pixel_x, pixel_y]), visible


@dataclass
class RectangleObstacle:
    center_x: float
    center_y: float
    width: float
    height: float
    angle_rad: float = 0.0

    def corners(self) -> np.ndarray:
        half_w = self.width / 2.0
        half_h = self.height / 2.0
        corners = np.array(
            [
                [-half_w, -half_h],
                [half_w, -half_h],
                [half_w, half_h],
                [-half_w, half_h],
            ],
            dtype=float,
        )
        rotated = corners @ rotation_matrix_2d(self.angle_rad).T
        rotated[:, 0] += self.center_x
        rotated[:, 1] += self.center_y
        return rotated
