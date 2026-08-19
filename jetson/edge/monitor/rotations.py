"""Rotación de imagen por cámara, resuelta en vivo.

Igual que el umbral: se guarda por `camera_id` en settings.json
(`camera_rotations[camera_id] = grados`) y se edita desde el tile del dashboard,
así sirve para cámaras RTSP **y** USB plug-and-play (que nunca aparecen en la
lista de Configuración).

La rotación se aplica EN LA CAPTURA, antes del ring buffer, por lo que afecta por
igual la vista en vivo, el clip y los frames que analiza el modelo.

Grados válidos: 0, 90, 180, 270. La captura consulta `for_camera()` por frame
(con caché), así que un cambio guardado en el dashboard se aplica sin reiniciar.
"""
from __future__ import annotations

import threading
from typing import Dict


def norm_deg(v, default: int = 0) -> int:
    """Normaliza a {0, 90, 180, 270}. Valor inválido -> default."""
    try:
        d = int(v) % 360
    except (TypeError, ValueError):
        return default
    return d if d in (0, 90, 180, 270) else default


class RotationProvider:
    def __init__(self, settings):
        self._settings = settings            # Settings o None (si no hay monitor)
        self._lock = threading.Lock()
        self._per_cam: Dict[str, int] = {}
        self.refresh()

    def refresh(self) -> None:
        """Relee las rotaciones desde settings.json (llamar tras guardar)."""
        per: Dict[str, int] = {}
        if self._settings is not None:
            d = self._settings.data()
            for cid, v in (d.get("camera_rotations") or {}).items():
                deg = norm_deg(v)
                if deg:  # 0 = sin rotación: no hace falta guardarlo
                    per[cid] = deg
        with self._lock:
            self._per_cam = per

    def for_camera(self, camera_id: str, default: int = 0) -> int:
        """Grados de rotación de una cámara. `default` = fallback (p.ej. config.yaml)."""
        with self._lock:
            return self._per_cam.get(camera_id, default)

    def snapshot(self) -> Dict[str, int]:
        """Estado para el dashboard: {camera_id: grados}."""
        with self._lock:
            return dict(self._per_cam)
