"""A detector that replays known detections.

Not a mock in the testing sense - it is a first-class backend. It is how the
demo runs without weights, and how the test suite exercises the real engine
end to end with exact ground truth.
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np

from ..types import FrameDetections
from .base import Detector


class ScriptedDetector(Detector):
    name = "scripted"

    def __init__(self, frames: Iterable[FrameDetections] | Mapping[int, FrameDetections]):
        if isinstance(frames, Mapping):
            self._by_index = dict(frames)
        else:
            self._by_index = {frame.frame_index: frame for frame in frames}

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameDetections:  # noqa: ARG002
        found = self._by_index.get(frame_index)
        if found is not None:
            return found
        return FrameDetections(frame_index=frame_index)
