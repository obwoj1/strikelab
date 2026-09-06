"""Runs the engine over an uploaded video and reports progress as it goes.

This is the same sequence as `strikelab analyze`, restructured so it can be
driven from a background thread and interrupted part way.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import Config
from ..engine import ShotEngine
from ..geometry import GoalCalibration
from ..models import DetectorError, build_detector
from ..physics import GroundCalibration
from ..sources.video import VideoFileSource, VideoWriter
from ..stats import summarise
from .jobs import Job

# Update the browser this often. Every frame would flood the event stream.
PROGRESS_EVERY = 5


def probe_video(path: Path) -> dict[str, Any]:
    """Read dimensions and frame count without analysing anything."""
    source = VideoFileSource(path)
    try:
        return {
            "width": source.info.width,
            "height": source.info.height,
            "fps": source.info.fps,
            "frame_count": source.info.frame_count,
        }
    finally:
        source.close()


def extract_frame(path: Path, index: int, out_path: Path) -> Path:
    """Save a single frame as a JPEG, for the calibration screen."""
    import cv2

    source = VideoFileSource(path)
    try:
        target = None
        for current, frame in source.frames():
            if current == index:
                target = frame
                break
        if target is None:
            raise ValueError(f"video has no frame {index}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), target, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return out_path
    finally:
        source.close()


def run_analysis(
    job: Job,
    *,
    video_path: Path,
    output_dir: Path,
    goal_corners: list[list[float]],
    ground_points: list[list[float]] | None = None,
    ground_preset: str = "six-yard-box",
    backend: str = "ultralytics",
    confidence: float = 0.25,
    device: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    write_video: bool = True,
) -> dict[str, Any]:
    """Analyse one video. Returns shots, summary and the annotated video path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = GoalCalibration.from_corners([(p[0], p[1]) for p in goal_corners])
    ground = None
    if ground_points:
        ground = GroundCalibration.from_preset(
            ground_preset, [(p[0], p[1]) for p in ground_points]
        )

    job.report(message="Opening video")
    source = VideoFileSource(video_path)
    info = source.info

    config = _build_config(info.fps, config_overrides)

    job.report(
        frames_total=max(1, info.frame_count),
        message=f"Loading the {backend} model",
    )

    try:
        detector = build_detector(
            backend,
            confidence=confidence,
            **({"device": device} if device and backend == "ultralytics" else {}),
        )
    except DetectorError:
        source.close()
        raise

    engine = ShotEngine(calibration, config=config, ground=ground)

    annotator = None
    writer = None
    annotated_path = output_dir / "annotated.mp4"
    if write_video:
        from ..render import Annotator

        annotator = Annotator(calibration, fps=info.fps)
        writer = VideoWriter(annotated_path, info.width, info.height, info.fps)

    processed = 0
    try:
        job.report(message="Analysing")
        for index, frame in source.frames():
            if job.cancelled:
                break
            detections = detector.detect(frame, index)
            result = engine.process(detections)
            processed += 1

            if writer is not None and annotator is not None:
                writer.write(annotator.draw(frame, result))

            if processed % PROGRESS_EVERY == 0:
                job.report(
                    frames_done=processed,
                    shots_found=len(engine.shots),
                    message=f"Analysing frame {processed}",
                )
    finally:
        if writer is not None:
            writer.close()
        source.close()

    engine.finish()
    job.report(
        frames_done=processed,
        shots_found=len(engine.shots),
        message="Writing results",
    )

    shots = [shot.to_json() for shot in engine.shots]
    summary = summarise(engine.shots).to_json()

    return {
        "shots": shots,
        "summary": summary,
        "annotated_path": str(annotated_path) if write_video else None,
        "frames_processed": processed,
        "calibration": calibration.to_json(),
        "ground_calibration": ground.to_json() if ground else None,
        "cancelled": job.cancelled,
    }


def _build_config(fps: float, overrides: dict[str, Any] | None) -> Config:
    """Apply per-session threshold overrides on top of the defaults."""
    from dataclasses import replace

    config = Config(fps=fps)
    if not overrides:
        return config

    shot_fields = {
        key: value
        for key, value in overrides.items()
        if hasattr(config.shot, key) and value is not None
    }
    tracker_fields = {
        key: value
        for key, value in overrides.items()
        if hasattr(config.tracker, key) and value is not None
    }

    return replace(
        config,
        shot=replace(config.shot, **shot_fields) if shot_fields else config.shot,
        tracker=replace(config.tracker, **tracker_fields) if tracker_fields else config.tracker,
    )
