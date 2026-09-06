"""Background analysis jobs.

Analysis is slow (model inference on every frame), so it cannot run inside a
request. Each job runs on its own thread and publishes progress events that the
browser consumes over Server-Sent Events.

One worker thread per job, capped by a semaphore, is the right shape here: the
work is a single long-running loop per video and the heavy lifting releases the
GIL inside the model and OpenCV.
"""

from __future__ import annotations

import queue
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

MAX_CONCURRENT_JOBS = 2


@dataclass
class JobEvent:
    type: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Job:
    session_id: str
    status: str = "queued"
    progress: float = 0.0
    frames_done: int = 0
    frames_total: int = 0
    shots_found: int = 0
    message: str = "Queued"
    error: str | None = None
    _subscribers: list[queue.Queue[JobEvent]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _cancel: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "status": self.status,
            "progress": round(self.progress, 4),
            "frames_done": self.frames_done,
            "frames_total": self.frames_total,
            "shots_found": self.shots_found,
            "message": self.message,
            "error": self.error,
        }

    def subscribe(self) -> queue.Queue[JobEvent]:
        listener: queue.Queue[JobEvent] = queue.Queue(maxsize=256)
        with self._lock:
            self._subscribers.append(listener)
        # Send current state immediately so a late subscriber is not blind.
        listener.put(JobEvent("progress", self.snapshot()))
        return listener

    def unsubscribe(self, listener: queue.Queue[JobEvent]) -> None:
        with self._lock:
            if listener in self._subscribers:
                self._subscribers.remove(listener)

    def publish(self, event: JobEvent) -> None:
        with self._lock:
            listeners = list(self._subscribers)
        for listener in listeners:
            try:
                listener.put_nowait(event)
            except queue.Full:
                # A browser that stopped reading must not stall the analysis.
                pass

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    # -- progress reporting used by the runner ------------------------------

    def report(
        self,
        *,
        frames_done: int | None = None,
        frames_total: int | None = None,
        shots_found: int | None = None,
        message: str | None = None,
        status: str | None = None,
    ) -> None:
        if frames_done is not None:
            self.frames_done = frames_done
        if frames_total is not None:
            self.frames_total = frames_total
        if shots_found is not None:
            self.shots_found = shots_found
        if message is not None:
            self.message = message
        if status is not None:
            self.status = status
        if self.frames_total > 0:
            self.progress = min(1.0, self.frames_done / self.frames_total)
        self.publish(JobEvent("progress", self.snapshot()))


class JobManager:
    """Owns running jobs and their event streams."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()
        self._slots = threading.Semaphore(max_concurrent)

    def get(self, session_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(session_id)

    def is_running(self, session_id: str) -> bool:
        job = self.get(session_id)
        return job is not None and job.status in {"queued", "running"}

    def start(
        self,
        session_id: str,
        work: Callable[[Job], None],
        *,
        on_finish: Callable[[Job], None] | None = None,
    ) -> Job:
        with self._lock:
            existing = self._jobs.get(session_id)
            if existing and existing.status in {"queued", "running"}:
                return existing
            job = Job(session_id=session_id)
            self._jobs[session_id] = job

        def runner() -> None:
            acquired = self._slots.acquire(timeout=3600)
            try:
                if not acquired:
                    job.status = "failed"
                    job.error = "Timed out waiting for a free analysis slot."
                    job.publish(JobEvent("failed", job.snapshot()))
                    return
                if job.cancelled:
                    job.status = "cancelled"
                    job.publish(JobEvent("cancelled", job.snapshot()))
                    return
                job.report(status="running", message="Starting")
                work(job)
                if job.cancelled:
                    job.status = "cancelled"
                    job.message = "Cancelled"
                    job.publish(JobEvent("cancelled", job.snapshot()))
                else:
                    job.status = "done"
                    job.progress = 1.0
                    job.message = "Complete"
                    job.publish(JobEvent("done", job.snapshot()))
            except Exception as error:  # noqa: BLE001 - surface it, never crash the thread
                job.status = "failed"
                job.error = f"{type(error).__name__}: {error}"
                job.message = "Failed"
                traceback.print_exc()
                job.publish(JobEvent("failed", job.snapshot()))
            finally:
                if acquired:
                    self._slots.release()
                if on_finish is not None:
                    try:
                        on_finish(job)
                    except Exception:  # noqa: BLE001
                        traceback.print_exc()
                job.publish(JobEvent("close", {}))

        threading.Thread(target=runner, name=f"analyze-{session_id}", daemon=True).start()
        return job

    def cleanup(self, keep: int = 50) -> None:
        with self._lock:
            finished = [
                key
                for key, job in self._jobs.items()
                if job.status in {"done", "failed", "cancelled"}
            ]
            for key in finished[:-keep] if len(finished) > keep else []:
                self._jobs.pop(key, None)
