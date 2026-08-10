"""Simulador offline del pipeline edge — para verificar sin cámaras ni GPU.

Reutiliza los componentes reales (RingBuffer, EventEngine, ClipEncoder, Store)
pero alimenta clips desde un vídeo o una carpeta de frames, y permite un
scorer 'mock' (sin modelo) para probar toda la fontanería de eventos/clips/outbox.

Ejemplos:
  # Con scorer simulado (no requiere torch):
  python -m tools.simulate_offline --config config.yaml --frames-dir \
      "D:/Codes/TransLowNet_Jetson/Propuesta2/FrameStream" --mock

  # Con el modelo real TransLowNet:
  python -m tools.simulate_offline --config config.yaml --video ruta/al/video.mp4
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from datetime import timedelta
from pathlib import Path
from typing import Iterator, List

import cv2
import numpy as np

from edge.capture.ring_buffer import RingBuffer
from edge.config import load_config
from edge.events.clip import ClipEncoder, object_key_for
from edge.events.engine import EventEngine
from edge.inference.params import clip_len_frames
from edge.storage.db import Store
from edge.types import Clip, ClipResult, Event, utcnow

log = logging.getLogger("simulate")


def _frames_from_dir(path: Path) -> Iterator[np.ndarray]:
    for name in sorted(os.listdir(path)):
        img = cv2.imread(str(path / name))
        if img is not None:
            yield cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _frames_from_video(path: Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames-dir")
    ap.add_argument("--video")
    ap.add_argument("--mock", action="store_true", help="usa scorer simulado (sin torch)")
    ap.add_argument("--camera-id", default="sim-cam")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repite la secuencia de frames N veces (útil si la fuente es corta)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    if args.frames_dir:
        source = list(_frames_from_dir(Path(args.frames_dir)))
    elif args.video:
        source = list(_frames_from_video(Path(args.video)))
    else:
        ap.error("indica --frames-dir o --video")

    if not source:
        ap.error("no se cargaron frames")
    if args.repeat > 1:
        source = source * args.repeat
    log.info("Cargados %d frames de entrada (repeat=%d).", len(source), args.repeat)

    store = Store(cfg.storage.db_path)
    encoder = ClipEncoder(cfg.storage, cfg.events)
    ring = RingBuffer(cfg.storage.ring_buffer_s)

    events: List[Event] = []

    def on_open(ev: Event):
        ev.clip_object_key = object_key_for(ev)
        store.save_event(ev, None)
        store.enqueue("event", {"event_id": ev.event_id, "state": "open"}, ev.event_id)
        events.append(ev)

    def on_close(ev: Event):
        ev.clip_object_key = object_key_for(ev)
        out = encoder.encode(ev, ring)
        store.save_event(ev, str(out) if out else None)
        store.enqueue("event", {"event_id": ev.event_id, "state": "closed"}, ev.event_id)
        if out:
            store.enqueue("clip", {"event_id": ev.event_id, "object_key": ev.clip_object_key,
                                   "clip_path": str(out)}, ev.event_id)
        events.append(ev)

    # Scorer
    if args.mock:
        model = None
        def class_name(i: int) -> str:
            names = cfg.model.class_names
            return names[i] if i < len(names) else f"class_{i}"
    else:
        from edge.inference.model import TransLowNet
        model = TransLowNet(cfg.model)
        class_name = model.class_name

    engine = EventEngine(cfg.device.id, cfg.events, class_name, on_open, on_close)

    clip_len = clip_len_frames(cfg.model.name)
    base = utcnow()
    buf: list = []
    t_start = base
    idx = 0
    for i, frame in enumerate(source):
        ts = base + timedelta(milliseconds=66 * i)  # ~15 fps simulados
        ring.append(ts, frame)
        if not buf:
            t_start = ts
        buf.append(frame)
        if len(buf) >= clip_len:
            clip = Clip(args.camera_id, np.stack(buf), t_start, ts)
            if model is None:
                # score sinusoidal: provoca un pico de anomalía para probar eventos
                score = 0.5 + 0.45 * math.sin(idx / 2.0)
                r = ClipResult(args.camera_id, t_start, ts, score, 3,
                               [0.1] * cfg.model.n_classes, 1.0)
            else:
                r = model.infer(clip)
            log.info("clip %d score=%.3f clase=%d", idx, r.score, r.class_id)
            engine.process(r)
            buf = []
            idx += 1

    store.close()
    log.info("Simulación terminada. Eventos generados: %d. Outbox/clips en %s / %s",
             len(events), cfg.storage.db_path, cfg.storage.clips_dir)


if __name__ == "__main__":
    main()
