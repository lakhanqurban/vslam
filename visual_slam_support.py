"""
visual_slam_support.py

Compatibility shim that re-exports the visual SLAM demo modules.
"""

from geometry import CameraModel, RectangleObstacle, rotation_matrix_2d, wrap_angle_rad
from features import PoseEstimator, VisualFeatures
from mapping import Keyframe, LoopClosureDetector, LoopClosureEvent, MapBuilder, BundleAdjustment
from scene import SyntheticScene

