"""Descubrimiento de cámaras USB (plug-and-play) vía V4L2 en Linux.

Enumera `/dev/video*` y filtra a los nodos que realmente entregan vídeo: un mismo
dispositivo UVC suele exponer varios nodos (p.ej. captura + metadatos) y solo
algunos devuelven frames. `probe_capture` abre el nodo y comprueba que entrega al
menos un frame antes de tratarlo como cámara usable.
"""
from __future__ import annotations

import glob
import logging
import re

import cv2

log = logging.getLogger(__name__)

_VIDEO_RE = re.compile(r"(\d+)$")


def _v4l2_backend() -> int:
    return getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)


def device_index(path: str) -> int:
    """Índice numérico de un nodo `/dev/videoN` (N), o 0 si no se reconoce."""
    m = _VIDEO_RE.search(path)
    return int(m.group(1)) if m else 0


def list_usb_devices() -> list[str]:
    """Nodos `/dev/video*` presentes ahora mismo, ordenados por índice.

    Solo lista lo que existe en /dev; no abre nada (rápido y seguro para sondear
    en bucle). El filtrado de "¿captura de verdad?" lo hace `probe_capture`.
    """
    return sorted(glob.glob("/dev/video*"), key=device_index)


def probe_capture(path: str, attempts: int = 5) -> bool:
    """True si el nodo entrega al menos un frame (es de captura, no metadatos).

    Abre y cierra el dispositivo; pensado para sondear nodos *nuevos* una sola
    vez, no en cada escaneo (eso lo controla el CameraManager).
    """
    cap = cv2.VideoCapture(device_index(path), _v4l2_backend())
    try:
        if not cap.isOpened():
            return False
        for _ in range(max(1, attempts)):
            ok, frame = cap.read()
            if ok and frame is not None:
                return True
        return False
    except cv2.error:
        return False
    finally:
        cap.release()
