"""Session aggregates and the JSONL event log."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from ..engine import Shot


@dataclass
class SessionSummary:
    shots: int
    goals: int
    on_target: int
    off_target: int
    saved: int
    blocked: int
    woodwork: int
    unresolved: int
    conversion_rate: float
    on_target_rate: float
    mean_speed_kmh: float
    best_speed_kmh: float
    total_quality: float
    zones: dict[str, int]
    verdicts: dict[str, int]

    def to_json(self) -> dict[str, object]:
        return {
            "shots": self.shots,
            "goals": self.goals,
            "on_target": self.on_target,
            "off_target": self.off_target,
            "saved": self.saved,
            "blocked": self.blocked,
            "woodwork": self.woodwork,
            "unresolved": self.unresolved,
            "conversion_rate": round(self.conversion_rate, 4),
            "on_target_rate": round(self.on_target_rate, 4),
            "mean_speed_kmh": round(self.mean_speed_kmh, 1),
            "best_speed_kmh": round(self.best_speed_kmh, 1),
            "expected_goals_lite": round(self.total_quality, 3),
            "zones": self.zones,
            "verdicts": self.verdicts,
        }


def summarise(shots: Sequence[Shot]) -> SessionSummary:
    verdicts = Counter(shot.verdict for shot in shots)
    zones = Counter(shot.zone for shot in shots if shot.zone)
    speeds = [shot.peak_speed_mps * 3.6 for shot in shots if shot.peak_speed_mps > 0]
    on_target = sum(1 for shot in shots if shot.on_target)
    total = len(shots)
    quality = sum(
        float(shot.quality["score"]) for shot in shots if shot.quality  # type: ignore[index]
    )

    return SessionSummary(
        shots=total,
        goals=verdicts.get("GOAL", 0),
        on_target=on_target,
        off_target=verdicts.get("OFF_TARGET", 0),
        saved=verdicts.get("SAVED", 0),
        blocked=verdicts.get("BLOCKED", 0),
        woodwork=verdicts.get("WOODWORK", 0),
        unresolved=verdicts.get("UNRESOLVED", 0),
        conversion_rate=(verdicts.get("GOAL", 0) / total) if total else 0.0,
        on_target_rate=(on_target / total) if total else 0.0,
        mean_speed_kmh=(sum(speeds) / len(speeds)) if speeds else 0.0,
        best_speed_kmh=max(speeds) if speeds else 0.0,
        total_quality=quality,
        zones=dict(zones),
        verdicts=dict(verdicts),
    )


def write_jsonl(path: str | Path, shots: Iterable[Shot]) -> Path:
    """One JSON object per shot, in the order they happened."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for shot in shots:
            handle.write(json.dumps(shot.to_json()) + "\n")
    return target


def write_summary(path: str | Path, summary: SessionSummary, extra: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"summary": summary.to_json(), **extra}
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def format_table(shots: Sequence[Shot]) -> str:
    """A terminal table of the session, for the CLI."""
    if not shots:
        return "No shots detected."

    header = f"{'#':>3}  {'VERDICT':<11} {'SPEED':>9}  {'PLACEMENT':<15} {'Q':>5}  NOTE"
    lines = [header, "-" * len(header)]
    for shot in shots:
        quality = shot.quality["score"] if shot.quality else 0.0  # type: ignore[index]
        note = shot.notes[0] if shot.notes else ""
        lines.append(
            f"{shot.shot_id:>3}  {shot.verdict:<11} "
            f"{shot.peak_speed_mps * 3.6:>6.0f} kmh  "
            f"{(shot.zone or 'off target'):<15} {quality:>5.3f}  {note[:46]}"
        )
    return "\n".join(lines)
