"""StrikeLab: soccer shot evaluation from video."""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Config, RenderConfig, ShotConfig, TrackerConfig
from .geometry import CalibrationError, GoalCalibration
from .physics import GroundCalibration, shot_angle_deg, shot_quality
from .types import Box, FrameDetections, Pose

__all__ = [
    "Box",
    "CalibrationError",
    "Config",
    "FrameDetections",
    "GoalCalibration",
    "GroundCalibration",
    "Pose",
    "RenderConfig",
    "ShotConfig",
    "TrackerConfig",
    "__version__",
    "shot_angle_deg",
    "shot_quality",
]
