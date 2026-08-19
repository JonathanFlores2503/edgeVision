"""Umbral de anomalía por cámara, resuelto en vivo.

Orden de resolución para una cámara:
  1. `thr` propio de la cámara (settings.json -> `camera_thresholds[camera_id]`),
     editable desde el tile en el dashboard. Sirve para RTSP y USB por igual.
  2. Umbral global del dashboard (`alerts.score_threshold`).
  3. `events.score_threshold` de `config.yaml` (default de fábrica).

Cachea los valores y se refresca al guardar config (sin reiniciar el nodo).
"""
from __future__ import annotations

import threading
from typing import Dict, Optional


def _as_float(v, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class ThresholdProvider:
    def __init__(self, settings, default_thr: float):
        self._settings = settings           # Settings o None (si no hay monitor)
        self._default = float(default_thr)  # events.score_threshold (config.yaml)
        self._lock = threading.Lock()
        self._global = self._default
        self._per_cam: Dict[str, float] = {}
        self.refresh()

    def refresh(self) -> None:
        """Relee umbrales desde settings.json (llamar tras guardar config)."""
        g = self._default
        per: Dict[str, float] = {}
        if self._settings is not None:
            d = self._settings.data()
            sg = (d.get("alerts") or {}).get("score_threshold")
            g = _as_float(sg, self._default)
            for cid, v in (d.get("camera_thresholds") or {}).items():
                f = _as_float(v)
                if f is not None:
                    per[cid] = f
        with self._lock:
            self._global = g
            self._per_cam = per

    def for_camera(self, camera_id: str) -> float:
        """Umbral efectivo de una cámara (su thr propio, o el global)."""
        with self._lock:
            return self._per_cam.get(camera_id, self._global)

    def snapshot(self) -> dict:
        """Estado para el dashboard: {global, cameras:{id:thr}}."""
        with self._lock:
            return {"global": self._global, "cameras": dict(self._per_cam)}
