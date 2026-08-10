"""Tipos de datos compartidos dentro del nodo edge."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class Clip:
    """Un clip muestreado de una cámara, listo para inferencia."""
    camera_id: str
    frames: np.ndarray            # (T, H, W, 3) RGB uint8 — para preprocess y para recortar el .mp4
    t_start: datetime
    t_end: datetime


@dataclass
class ClipResult:
    """Salida del modelo por clip (contrato de telemetría)."""
    camera_id: str
    t_start: datetime
    t_end: datetime
    score: float
    class_id: int
    class_probs: List[float]
    latency_ms: float
    ts: datetime = field(default_factory=utcnow)


@dataclass
class Event:
    """Evento de anomalía (apertura/cierre)."""
    event_id: str
    device_id: str
    camera_id: str
    state: str                    # "open" | "closed"
    t_start: datetime
    t_end: Optional[datetime]
    max_score: float
    class_id: int
    class_name: str
    clip_object_key: Optional[str] = None
    clip_uri: Optional[str] = None
