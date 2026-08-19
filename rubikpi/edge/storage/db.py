"""Persistencia local (SQLite): eventos + outbox store-and-forward.

El outbox garantiza la entrega al cloud aunque la red falle: cada mensaje/clip
queda 'pending' hasta confirmarse 'sent', con reintentos.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..types import Event, iso

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id        TEXT PRIMARY KEY,
    device_id       TEXT NOT NULL,
    camera_id       TEXT NOT NULL,
    state           TEXT NOT NULL,
    t_start         TEXT NOT NULL,
    t_end           TEXT,
    max_score       REAL NOT NULL,
    class_id        INTEGER NOT NULL,
    class_name      TEXT NOT NULL,
    clip_path       TEXT,
    clip_object_key TEXT,
    clip_uri        TEXT,
    created_at      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS outbox (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,              -- 'event' | 'clip' | 'telemetry'
    event_id    TEXT,
    payload     TEXT NOT NULL,              -- JSON
    status      TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox(status);
"""


def _now() -> str:
    return iso(datetime.now(timezone.utc))


class Store:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── Eventos ──────────────────────────────────────────────────────────────
    def save_event(self, ev: Event, clip_path: Optional[str]) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO events (event_id, device_id, camera_id, state, t_start, t_end,
                       max_score, class_id, class_name, clip_path, clip_object_key, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(event_id) DO UPDATE SET
                       state=excluded.state, t_end=excluded.t_end, max_score=excluded.max_score,
                       class_id=excluded.class_id, class_name=excluded.class_name,
                       clip_path=excluded.clip_path, clip_object_key=excluded.clip_object_key""",
                (ev.event_id, ev.device_id, ev.camera_id, ev.state, iso(ev.t_start),
                 iso(ev.t_end) if ev.t_end else None, ev.max_score, ev.class_id, ev.class_name,
                 clip_path, ev.clip_object_key, _now()),
            )
            self._conn.commit()

    # ── Outbox ───────────────────────────────────────────────────────────────
    def enqueue(self, kind: str, payload: dict, event_id: Optional[str] = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO outbox (kind, event_id, payload, created_at, updated_at)
                   VALUES (?,?,?,?,?)""",
                (kind, event_id, json.dumps(payload), _now(), _now()),
            )
            self._conn.commit()
            return cur.lastrowid

    def pending(self, limit: int = 50) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(
                "SELECT * FROM outbox WHERE status='pending' ORDER BY id ASC LIMIT ?",
                (limit,),
            ).fetchall())

    def count_pending(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM outbox WHERE status='pending'"
            ).fetchone()[0]

    def mark_sent(self, outbox_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE outbox SET status='sent', updated_at=? WHERE id=?",
                (_now(), outbox_id),
            )
            self._conn.commit()

    def mark_attempt(self, outbox_id: int, max_attempts: int) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE outbox
                   SET attempts=attempts+1,
                       status=CASE WHEN attempts+1 >= ? THEN 'failed' ELSE 'pending' END,
                       updated_at=?
                   WHERE id=?""",
                (max_attempts, _now(), outbox_id),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
