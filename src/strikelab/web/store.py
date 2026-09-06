"""SQLite storage for sessions, shots and players.

Plain sqlite3 rather than an ORM: the schema is small, the queries are simple,
and it keeps the web app dependency-light. The database lives alongside the
uploaded video files under the data directory.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS players (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'new',
    error           TEXT,
    player_id       TEXT REFERENCES players(id) ON DELETE SET NULL,
    video_path      TEXT,
    video_name      TEXT,
    annotated_path  TEXT,
    width           INTEGER,
    height          INTEGER,
    fps             REAL,
    frame_count     INTEGER,
    backend         TEXT,
    goal_corners    TEXT,
    ground_points   TEXT,
    ground_preset   TEXT,
    summary         TEXT,
    share_token     TEXT
);

CREATE TABLE IF NOT EXISTS shots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    shot_index   INTEGER NOT NULL,
    verdict      TEXT NOT NULL,
    confidence   TEXT,
    on_target    INTEGER NOT NULL DEFAULT 0,
    zone         TEXT,
    speed_kmh    REAL,
    quality      REAL,
    goal_x       REAL,
    goal_y       REAL,
    release_frame INTEGER,
    end_frame    INTEGER,
    payload      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_shots_session ON shots(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at DESC);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Store:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def connect(self) -> sqlite3.Connection:
        """One connection per thread; the analysis job runs off the request thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Additive migrations, so an existing database keeps working."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(sessions)")}
        for column, ddl in (
            ("share_token", "ALTER TABLE sessions ADD COLUMN share_token TEXT"),
            ("player_id", "ALTER TABLE sessions ADD COLUMN player_id TEXT"),
        ):
            if column not in existing:
                conn.execute(ddl)
        conn.commit()

    # ------------------------------------------------------------------
    # sessions

    def create_session(self, session_id: str, name: str) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, name, created_at, status) VALUES (?, ?, ?, 'new')",
                (session_id, name, _now()),
            )
        return self.get_session(session_id)  # type: ignore[return-value]

    def update_session(self, session_id: str, **fields: Any) -> None:
        if not fields:
            return
        columns = ", ".join(f"{key} = ?" for key in fields)
        values = [
            json.dumps(value) if isinstance(value, (dict, list)) else value
            for value in fields.values()
        ]
        with self.connect() as conn:
            conn.execute(
                f"UPDATE sessions SET {columns} WHERE id = ?", (*values, session_id)
            )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.connect().execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return _session_row(row) if row else None

    def get_session_by_token(self, token: str) -> dict[str, Any] | None:
        row = self.connect().execute(
            "SELECT * FROM sessions WHERE share_token = ?", (token,)
        ).fetchone()
        return _session_row(row) if row else None

    def list_sessions(self, limit: int = 100, player_id: str | None = None) -> list[dict[str, Any]]:
        if player_id:
            rows = self.connect().execute(
                "SELECT * FROM sessions WHERE player_id = ? ORDER BY created_at DESC LIMIT ?",
                (player_id, limit),
            ).fetchall()
        else:
            rows = self.connect().execute(
                "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_session_row(row) for row in rows]

    def delete_session(self, session_id: str) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM shots WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # ------------------------------------------------------------------
    # shots

    def replace_shots(self, session_id: str, shots: Iterable[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM shots WHERE session_id = ?", (session_id,))
            conn.executemany(
                """INSERT INTO shots (
                       session_id, shot_index, verdict, confidence, on_target, zone,
                       speed_kmh, quality, goal_x, goal_y, release_frame, end_frame, payload
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [_shot_values(session_id, shot) for shot in shots],
            )

    def list_shots(self, session_id: str) -> list[dict[str, Any]]:
        rows = self.connect().execute(
            "SELECT payload FROM shots WHERE session_id = ? ORDER BY shot_index",
            (session_id,),
        ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def shots_for_player(self, player_id: str) -> list[dict[str, Any]]:
        rows = self.connect().execute(
            """SELECT s.payload, ss.created_at, ss.id AS session_id, ss.name AS session_name
               FROM shots s JOIN sessions ss ON ss.id = s.session_id
               WHERE ss.player_id = ? ORDER BY ss.created_at, s.shot_index""",
            (player_id,),
        ).fetchall()
        out = []
        for row in rows:
            payload = json.loads(row["payload"])
            payload["session_id"] = row["session_id"]
            payload["session_name"] = row["session_name"]
            payload["session_created_at"] = row["created_at"]
            out.append(payload)
        return out

    # ------------------------------------------------------------------
    # players

    def create_player(self, player_id: str, name: str) -> dict[str, Any]:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO players (id, name, created_at) VALUES (?, ?, ?)",
                (player_id, name, _now()),
            )
        return {"id": player_id, "name": name, "created_at": _now()}

    def list_players(self) -> list[dict[str, Any]]:
        rows = self.connect().execute(
            """SELECT p.*, COUNT(s.id) AS session_count
               FROM players p LEFT JOIN sessions s ON s.player_id = p.id
               GROUP BY p.id ORDER BY p.name""",
        ).fetchall()
        return [dict(row) for row in rows]

    def get_player(self, player_id: str) -> dict[str, Any] | None:
        row = self.connect().execute(
            "SELECT * FROM players WHERE id = ?", (player_id,)
        ).fetchone()
        return dict(row) if row else None

    def delete_player(self, player_id: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE sessions SET player_id = NULL WHERE player_id = ?", (player_id,))
            conn.execute("DELETE FROM players WHERE id = ?", (player_id,))


def _shot_values(session_id: str, shot: dict[str, Any]) -> tuple[Any, ...]:
    entry = shot.get("entry") or {}
    goal_point = entry.get("goal_plane_m") or [None, None]
    quality = shot.get("quality") or {}
    return (
        session_id,
        shot.get("shot_id", 0),
        shot.get("verdict", "UNRESOLVED"),
        shot.get("confidence"),
        1 if shot.get("on_target") else 0,
        shot.get("zone"),
        shot.get("peak_speed_kmh"),
        quality.get("score"),
        goal_point[0] if goal_point else None,
        goal_point[1] if goal_point else None,
        (shot.get("release") or {}).get("frame"),
        (shot.get("frames") or {}).get("end"),
        json.dumps(shot),
    )


def _session_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for field in ("goal_corners", "ground_points", "summary"):
        if data.get(field):
            try:
                data[field] = json.loads(data[field])
            except (TypeError, json.JSONDecodeError):
                data[field] = None
    return data
