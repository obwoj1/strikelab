"""Roboflow-hosted detection.

This is the closest analogue to the basketball write-up's setup: a small
detection model fine-tuned on sport-specific data, run through the Roboflow
inference SDK. The advantage over the zero-shot COCO path is that soccer
datasets label the *goalkeeper* separately from outfield players, which is what
lets the engine distinguish a save from a goal.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from ..types import Box, FrameDetections, Pose
from .base import Detector, DetectorError
from .registry import ROBOFLOW_CLASS_MAP, ROBOFLOW_PLAYERS_MODEL


class RoboflowDetector(Detector):
    name = "roboflow"

    def __init__(
        self,
        *,
        model_id: str = ROBOFLOW_PLAYERS_MODEL,
        api_key: str | None = None,
        confidence: float = 0.3,
        pose_detector: Detector | None = None,
    ) -> None:
        key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        if not key:
            raise DetectorError(
                "ROBOFLOW_API_KEY is not set. Put it in your environment or a .env "
                "file, or use --backend ultralytics to stay entirely local."
            )
        try:
            from inference import get_model
        except ImportError as error:  # pragma: no cover - depends on extras
            raise DetectorError(
                "The roboflow backend needs the 'roboflow' extra. "
                "Install it with: pip install 'strikelab[roboflow]'"
            ) from error

        try:
            self.model = get_model(model_id=model_id, api_key=key)
        except Exception as error:  # noqa: BLE001 - surface a readable message
            raise DetectorError(f"Could not load Roboflow model {model_id!r}: {error}") from error

        self.confidence = confidence
        self.pose_detector = pose_detector

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameDetections:
        try:
            result: Any = self.model.infer(frame, confidence=self.confidence)[0]
        except Exception as error:  # noqa: BLE001
            raise DetectorError(f"Roboflow inference failed on frame {frame_index}: {error}") from error

        ball: Box | None = None
        players: list[Box] = []
        keepers: list[Box] = []

        for prediction in getattr(result, "predictions", []):
            label = ROBOFLOW_CLASS_MAP.get(prediction.class_name, prediction.class_name)
            confidence = float(prediction.confidence)
            # Roboflow returns centre/width/height rather than corners.
            half_w = float(prediction.width) / 2.0
            half_h = float(prediction.height) / 2.0
            cx, cy = float(prediction.x), float(prediction.y)
            box = Box(cx - half_w, cy - half_h, cx + half_w, cy + half_h, confidence, label)

            if label == "ball":
                if ball is None or confidence > ball.confidence:
                    ball = box
            elif label == "goalkeeper":
                keepers.append(box)
            elif label == "player":
                players.append(box)

        poses: list[Pose] = []
        if self.pose_detector is not None:
            poses = self.pose_detector.detect(frame, frame_index).poses

        return FrameDetections(
            frame_index=frame_index,
            ball=ball,
            players=players,
            goalkeepers=keepers,
            poses=poses,
        )
