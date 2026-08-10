"""Codificador de clips de anomalía.

Al cerrarse un evento, recorta del ring buffer los frames de la ventana
[t_start - pre_roll, t_end + post_roll] y los escribe como .mp4 (H.264).
"""
from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import cv2

from ..capture.ring_buffer import RingBuffer
from ..config import EventsCfg, StorageCfg
from ..types import Event

log = logging.getLogger(__name__)


def object_key_for(event: Event) -> str:
    """Clave de objeto en el storage del cloud: device/camera/fecha/event_id.mp4."""
    day = event.t_start.strftime("%Y-%m-%d")
    return f"{event.device_id}/{event.camera_id}/{day}/{event.event_id.replace(':', '_')}.mp4"


class ClipEncoder:
    def __init__(self, storage: StorageCfg, events: EventsCfg, fps: float = 15.0):
        self.clips_dir = Path(storage.clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.events = events
        self.fps = fps

    def encode(self, event: Event, ring: RingBuffer) -> "Path | None":
        start = event.t_start - timedelta(seconds=self.events.pre_roll_s)
        end = (event.t_end or event.t_start) + timedelta(seconds=self.events.post_roll_s)
        _, frames = ring.window(start, end)

        if frames.size == 0:
            log.warning("Sin frames en la ventana del evento %s; no se genera clip.", event.event_id)
            return None

        out_path = self.clips_dir / Path(object_key_for(event)).name
        h, w = frames.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
        try:
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

        log.info("Clip generado %s (%d frames)", out_path, len(frames))
        return out_path
