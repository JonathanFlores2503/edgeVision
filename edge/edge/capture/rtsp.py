"""Captura RTSP por cámara: lee frames, alimenta el ring buffer y forma clips.

Un hilo por cámara. Cada `clip_len` frames ensambla un Clip y lo envía al
InferenceWorker. Reconecta automáticamente si el stream se cae.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from typing import Callable, Optional

import cv2

# OpenCV lee OPENCV_FFMPEG_CAPTURE_OPTIONS al construir el VideoCapture. Como es
# global al proceso, serializamos (set env + abrir) para no pisar la config entre
# cámaras que reconectan en hilos distintos.
_FFMPEG_LOCK = threading.Lock()

# Fuentes de red (se abren por FFmpeg). El resto se trata como cámara local USB.
_NET_RE = re.compile(r"^(rtsp|rtsps|rtmp|http|https|udp|tcp)://", re.I)
_DEV_RE = re.compile(r"(\d+)$")


def is_network_source(url: str) -> bool:
    """True si la URL es un stream de red (RTSP/HTTP/...), False si es USB local."""
    return bool(_NET_RE.match(url.strip()))

from ..config import CameraCfg
from ..inference.params import clip_len_frames
from ..types import Clip, utcnow
from .ring_buffer import RingBuffer

log = logging.getLogger(__name__)

# Grados -> código de rotación de OpenCV. 90/270 intercambian ancho/alto (el
# modelo reescala igual y el clip queda consistente); 180 conserva dimensiones.
_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class CameraCapture:
    def __init__(
        self,
        cam: CameraCfg,
        model_name: str,
        ring_seconds: float,
        on_clip: Callable[[Clip], None],
        on_frame: Optional[Callable[[str, "object"], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        rotate_provider: Optional[Callable[[str, int], int]] = None,
        proc_short_side: int = 0,
        clip_cooldown_s: float = 0.0,
    ):
        self.cam = cam
        self.on_clip = on_clip
        self.on_frame = on_frame
        self.on_status = on_status
        self.clip_len = clip_len_frames(model_name)
        # Cadencia por cámara: tras emitir un clip, no se vuelve a acumular hasta
        # pasados estos segundos. Iguala la producción al worker (back-pressure
        # suave) sin descartar clips. 0 = contiguo.
        self._clip_cooldown_s = float(clip_cooldown_s or 0.0)
        self._clip_resume_t = 0.0            # monotonic; antes de esto no se acumula
        # Reescalado en captura (lado corto). 0 = full-res. Recorta la RAM: todo lo
        # que se guarda (ring buffer, cola, clip, preview) usa ya el frame reducido.
        self._proc_short_side = int(proc_short_side or 0)
        # Rotación de captura: la fuente de verdad es el dashboard (por camera_id),
        # consultada en vivo vía `rotate_provider`. `cam.rotate` (config.yaml) es el
        # fallback cuando no hay dashboard o la cámara no tiene rotación propia.
        self._rotate_provider = rotate_provider
        self._rotate_default = int(getattr(cam, "rotate", 0) or 0)
        self._rotate_deg = -1                # grados actuales en caché (-1 = sin resolver)
        self._rotate_code = None             # código cv2 derivado (None = sin rotación)
        self.ring = RingBuffer(ring_seconds)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"cap-{cam.id}", daemon=True)
        self._frame_count = 0
        self._last_fps_t = time.time()
        self._fps = 0.0

    @property
    def fps(self) -> float:
        return self._fps

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _open(self) -> "cv2.VideoCapture | None":
        url = self.cam.url.strip()
        cap = self._open_network(url) if is_network_source(url) else self._open_local(url)
        if cap is None:
            return None
        # Buffer mínimo: menos latencia y descarta frames atrasados al reconectar.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        return cap

    def _open_network(self, url: str) -> "cv2.VideoCapture | None":
        # Forzamos el transporte RTSP (tcp por defecto). Con UDP, muchas cámaras
        # Hikvision y redes con NAT/firewall cierran el flujo con RST y OpenCV no
        # llega a recibir frames (la cámara nunca aparece en la interfaz), aunque
        # `ffplay` sí funcione. Equivale a `ffplay -rtsp_transport tcp`.
        transport = (getattr(self.cam, "transport", "tcp") or "tcp").lower()
        opts = (
            f"rtsp_transport;{transport}"
            "|stimeout;5000000"        # timeout de socket (µs): un stream caído no bloquea
            "|max_delay;500000"
            "|reorder_queue_size;0"
        )
        with _FFMPEG_LOCK:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log.warning("[%s] No se pudo abrir RTSP %s (transport=%s)",
                        self.cam.id, url, transport)
            cap.release()
            return None
        log.info("[%s] RTSP abierto (transport=%s).", self.cam.id, transport)
        return cap

    def _open_local(self, url: str) -> "cv2.VideoCapture | None":
        # Cámara USB/plug-and-play: índice ("0") o ruta ("/dev/video0").
        # En Linux usamos el backend V4L2 explícito (no FFmpeg, que es para red).
        if url.isdigit():
            src: "int | str" = int(url)
        else:
            m = _DEV_RE.search(url)
            src = int(m.group(1)) if (m and url.startswith("/dev/video")) else url
        backend = getattr(cv2, "CAP_V4L2", cv2.CAP_ANY) if isinstance(src, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(src, backend)
        if not cap.isOpened():
            log.warning("[%s] No se pudo abrir la cámara local %s", self.cam.id, url)
            cap.release()
            return None
        log.info("[%s] Cámara USB abierta (%s).", self.cam.id, url)
        return cap

    def _downscale(self, frame_rgb: "np.ndarray") -> "np.ndarray":
        """Reduce el frame a `proc_short_side` en el lado corto (solo si es mayor;
        nunca amplía), conservando el aspecto. INTER_AREA da mejor calidad al reducir."""
        if self._proc_short_side <= 0:
            return frame_rgb
        h, w = frame_rgb.shape[:2]
        short = min(h, w)
        if short <= self._proc_short_side:
            return frame_rgb
        scale = self._proc_short_side / short
        new_w, new_h = max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)
        return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _refresh_rotation(self) -> None:
        """Resuelve los grados de rotación vigentes (dashboard -> fallback config) y,
        si cambiaron, recalcula el código cv2. Barato: solo recalcula al cambiar."""
        deg = (self._rotate_provider(self.cam.id, self._rotate_default)
               if self._rotate_provider is not None else self._rotate_default)
        if deg != self._rotate_deg:
            self._rotate_deg = deg
            self._rotate_code = _ROTATE_CODES.get(deg)

    def _status(self, status: str) -> None:
        if self.on_status is not None:
            try:
                self.on_status(self.cam.id, status)
            except Exception:  # noqa: BLE001 - el estado nunca debe tumbar la captura
                pass

    def _run(self):
        cap = None
        clip_frames: list = []
        clip_ts_start = None
        live = False
        backoff = 2.0  # backoff exponencial: evita martillar la cámara (algunas, como
                       # Hikvision, bloquean la IP tras muchos intentos seguidos)

        while not self._stop.is_set():
            if cap is None:
                self._status("conectando")
                cap = self._open()
                if cap is None:
                    self._status("sin señal")
                    self._stop.wait(backoff)
                    backoff = min(backoff * 2, 30.0)  # 2,4,8,16,30,30...
                    continue
                backoff = 2.0  # conexión OK -> reinicia el backoff

            ok, frame_bgr = cap.read()
            if not ok:
                log.warning("[%s] Frame perdido; reconectando.", self.cam.id)
                self._status("sin señal")
                live = False
                cap.release()
                cap = None
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30.0)
                continue

            ts = utcnow()
            if not live:
                live = True
                self._status("en vivo")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # Reescalado en el ORIGEN: reduce la RAM de TODO lo que sigue (ring
            # buffer, cola de inferencia, clip y preview). El modelo reescala igual.
            frame_rgb = self._downscale(frame_rgb)
            # Rotación en el ORIGEN: todo lo que sigue (ring buffer, vista en vivo,
            # clip y los frames que recibe el modelo) usa ya el frame rotado.
            self._refresh_rotation()
            if self._rotate_code is not None:
                frame_rgb = cv2.rotate(frame_rgb, self._rotate_code)
            self.ring.append(ts, frame_rgb)
            if self.on_frame is not None:
                self.on_frame(self.cam.id, frame_rgb)

            # Cadencia: durante el cooldown seguimos leyendo (ring buffer + vista en
            # vivo arriba se mantienen), pero NO acumulamos frames para el siguiente
            # clip. Así se deja un hueco entre clips inferidos sin saturar la cola.
            if self._clip_cooldown_s and time.monotonic() < self._clip_resume_t:
                continue

            if not clip_frames:
                clip_ts_start = ts
            clip_frames.append(frame_rgb)

            if len(clip_frames) >= self.clip_len:
                import numpy as np

                clip = Clip(
                    camera_id=self.cam.id,
                    frames=np.stack(clip_frames),
                    t_start=clip_ts_start,
                    t_end=ts,
                )
                self.on_clip(clip)
                clip_frames = []
                if self._clip_cooldown_s:
                    self._clip_resume_t = time.monotonic() + self._clip_cooldown_s

            self._tick_fps()

        if cap is not None:
            cap.release()
        log.info("[%s] Captura detenida.", self.cam.id)

    def _tick_fps(self):
        self._frame_count += 1
        now = time.time()
        if now - self._last_fps_t >= 2.0:
            self._fps = self._frame_count / (now - self._last_fps_t)
            self._frame_count = 0
            self._last_fps_t = now
