"""Ring buffer en RAM por cámara.

Retiene los últimos N segundos de frames (con timestamp) para:
  - formar clips de inferencia, y
  - recortar el .mp4 del evento con pre/post-roll.
"""
from __future__ import annotations

import threading
from collections import deque
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np


class RingBuffer:
    def __init__(self, max_seconds: float, est_fps: float = 15.0):
        self._maxlen = max(int(max_seconds * est_fps), 32)
        self._buf: deque[Tuple[datetime, np.ndarray]] = deque(maxlen=self._maxlen)
        self._lock = threading.Lock()

    def append(self, ts: datetime, frame_rgb: np.ndarray) -> None:
        with self._lock:
            self._buf.append((ts, frame_rgb))

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def last_n(self, n: int) -> Tuple[List[datetime], np.ndarray]:
        """Devuelve los últimos n frames (timestamps, array (n,H,W,3))."""
        with self._lock:
            items = list(self._buf)[-n:]
        ts = [t for t, _ in items]
        frames = np.stack([f for _, f in items]) if items else np.empty((0,))
        return ts, frames

    def window(self, start: datetime, end: datetime) -> Tuple[List[datetime], np.ndarray]:
        """Frames cuyo timestamp cae en [start, end] (para recortar el clip del evento)."""
        with self._lock:
            items = [(t, f) for t, f in self._buf if start <= t <= end]
        ts = [t for t, _ in items]
        frames = np.stack([f for _, f in items]) if items else np.empty((0,))
        return ts, frames
