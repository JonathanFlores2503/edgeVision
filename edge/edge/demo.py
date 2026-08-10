"""Pipeline de demostración reutilizable.

Toma una fuente de vídeo (webcam / archivo / carpeta de frames), corre el scorer
(simulado con mock=True, o el modelo real TransLowNet) y alimenta un MonitorState
(frames + scores + eventos). Lo usan tanto la demo web como la app de escritorio.
  cd D:/Codes/phdFinal/edge
  python -m tools.live_demo --config config.yaml --source "D:/Codes/TransLowNet_Jetson/Propuesta2/FrameStream" --mock --clip-frames 12
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, Optional

import cv2
import numpy as np

from .config import EdgeConfig, GpsCfg
from .events.clip import ClipEncoder, object_key_for
from .events.engine import EventEngine
from .gps.neo6m import GpsReader
from .inference.params import clip_len_frames
from .monitor.state import MonitorState
from .storage.db import Store
from .capture.ring_buffer import RingBuffer
from .types import Clip, ClipResult, Event, iso, utcnow

log = logging.getLogger(__name__)


def frame_source(src: str) -> Iterator[np.ndarray]:
    """Genera frames RGB en bucle infinito desde webcam / vídeo / carpeta."""
    if src.isdigit():  # webcam
        cap = cv2.VideoCapture(int(src))
        if not cap.isOpened():
            raise SystemExit(f"No se pudo abrir la webcam {src}")
        while True:
            ok, f = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    elif Path(src).is_dir():  # carpeta de frames (en bucle)
        files = sorted(os.listdir(src))
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(src, n)), cv2.COLOR_BGR2RGB)
                for n in files if cv2.imread(os.path.join(src, n)) is not None]
        if not imgs:
            raise SystemExit("Carpeta de frames vacía")
        while True:
            for im in imgs:
                yield im
                time.sleep(1 / 15)
    else:  # archivo de vídeo (en bucle)
        while True:
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise SystemExit(f"No se pudo abrir el vídeo {src}")
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                time.sleep(1 / 25)
            cap.release()


class DemoPipeline:
    def __init__(
        self,
        cfg: EdgeConfig,
        state: MonitorState,
        source: str,
        mock: bool = True,
        camera_id: str = "demo-cam",
        clip_frames: int = 24,
        notifier=None,
        second_camera_id: Optional[str] = None,
    ):
        self.cfg = cfg
        self.state = state
        self.source = source
        self.camera_id = camera_id
        self.mock = mock
        self.notifier = notifier
        # Segunda cámara SIMULADA: misma fuente espejada, procesada aparte.
        self.second_camera_id = second_camera_id
        self._stop = threading.Event()

        self.store = Store(cfg.storage.db_path)
        self.encoder = ClipEncoder(cfg.storage, cfg.events)
        self.ring = RingBuffer(cfg.storage.ring_buffer_s)

        state.set_meta(device_id=cfg.device.id, site=cfg.device.site, model=cfg.model.name)
        # GPS simulado para que el mapa del dashboard se mueva en la demo.
        self.gps = GpsReader(GpsCfg(enabled=False, simulate=True), state)

        if mock:
            self.model = None
            self.class_name = self._mock_class_name
            self.clip_len = clip_frames
        else:
            from .inference.model import TransLowNet
            self.model = TransLowNet(cfg.model)
            self.class_name = self.model.class_name
            self.clip_len = clip_len_frames(cfg.model.name)

        self.engine = EventEngine(cfg.device.id, cfg.events, self.class_name,
                                  self._on_open, self._on_close)

    def _mock_class_name(self, i: int) -> str:
        n = self.cfg.model.class_names
        return n[i] if i < len(n) else f"class_{i}"

    def _infer_clip(self, camera_id, buf, t_start, t_end, idx, phase=0.0):
        """Calcula el ClipResult de un buffer para una cámara dada."""
        if self.model is None:
            score = 0.5 + 0.42 * math.sin(idx / 3.0 + phase)
            return ClipResult(camera_id, t_start, t_end, score, 3,
                              [0.1] * self.cfg.model.n_classes, 1.0)
        return self.model.infer(Clip(camera_id, np.stack(buf), t_start, t_end))

    def _on_open(self, ev: Event):
        ev.clip_object_key = object_key_for(ev)
        self.store.save_event(ev, None)
        self.store.enqueue("event", {"event_id": ev.event_id, "state": "open"}, ev.event_id)
        self.state.add_event(ev)
        if self.notifier is not None:
            gps = self.state.status().get("gps", {})
            self.notifier.notify_event(
                camera_id=ev.camera_id, class_name=ev.class_name, score=ev.max_score,
                when=iso(ev.t_start), lat=gps.get("lat"), lon=gps.get("lon"))

    def _on_close(self, ev: Event):
        ev.clip_object_key = object_key_for(ev)
        out = self.encoder.encode(ev, self.ring)
        self.store.save_event(ev, str(out) if out else None)
        self.state.add_event(ev)

    def stop(self):
        self._stop.set()

    def run(self):
        """Bucle bloqueante; correr en un hilo. Termina al llamar a stop()."""
        log.info("DemoPipeline fuente=%s clip=%d scorer=%s",
                 self.source, self.clip_len, "mock" if self.mock else self.cfg.model.name)
        self.gps.start()
        buf: list = []
        buf2: list = []
        second = self.second_camera_id
        t_start = utcnow()
        idx = 0
        try:
            for frame in frame_source(self.source):
                if self._stop.is_set():
                    break
                now = utcnow()
                self.ring.append(now, frame)
                self.state.update_frame(self.camera_id, frame)
                mirror = None
                if second:
                    mirror = np.ascontiguousarray(frame[:, ::-1, :])  # espejo horizontal
                    self.state.update_frame(second, mirror)
                if not buf:
                    t_start = now
                buf.append(frame)
                if second:
                    buf2.append(mirror)

                if len(buf) >= self.clip_len:
                    r = self._infer_clip(self.camera_id, buf, t_start, now, idx)
                    self.state.update_result(r, self.class_name(r.class_id))
                    self.engine.process(r)
                    if second:
                        r2 = self._infer_clip(second, buf2, t_start, now, idx, phase=1.7)
                        self.state.update_result(r2, self.class_name(r2.class_id))
                        self.engine.process(r2)
                    buf = []
                    buf2 = []
                    idx += 1
        finally:
            self.gps.stop()
            self.store.close()
            log.info("DemoPipeline detenido.")
