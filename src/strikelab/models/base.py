"""The detector contract.

The engine only ever sees FrameDetections, so any backend that can fill one in
works: a YOLO checkpoint, a Roboflow-hosted RF-DETR, or a scripted fixture.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..types import FrameDetections


@runtime_checkable
class Detector(Protocol):
    """Turns one BGR frame into detections."""

    name: str

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameDetections: ...


class DetectorError(RuntimeError):
    """Raised when a backend cannot be constructed or run."""
