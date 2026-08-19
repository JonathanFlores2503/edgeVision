"""Scorer simulado (sin torch ni pesos).

Permite correr el nodo real completo —captura RTSP/USB, eventos, interfaz— para
ver las cámaras activadas sin necesitar GPU ni los pesos de TransLowNet. Genera
scores oscilantes deterministas (mayormente "normal" con algún pico para probar
eventos). Misma interfaz que `TransLowNet`: `infer()` y `class_name()`.
"""
from __future__ import annotations

import math
import time

from ..config import ModelCfg
from ..types import Clip, ClipResult


class MockModel:
    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self._n = 0

    def infer(self, clip: Clip) -> ClipResult:
        t0 = time.perf_counter()
        self._n += 1
        # Score base bajo (~normal) con oscilación suave; algún pico ocasional.
        score = 0.03 + 0.04 * (0.5 + 0.5 * math.sin(self._n / 5.0))
        n_classes = max(self.cfg.n_classes, 1)
        class_id = self._n % n_classes
        probs = [0.0] * n_classes
        probs[class_id] = 1.0
        return ClipResult(
            camera_id=clip.camera_id,
            t_start=clip.t_start,
            t_end=clip.t_end,
            score=score,
            class_id=class_id,
            class_probs=probs,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )

    def class_name(self, class_id: int) -> str:
        names = self.cfg.class_names
        return names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
