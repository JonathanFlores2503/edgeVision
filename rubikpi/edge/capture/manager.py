"""Gestión de cámaras: estáticas (RTSP del config) + USB plug-and-play.

Mantiene las capturas RTSP definidas en `config.yaml` y, si está activado el
auto-detect, lanza un hilo supervisor que escanea `/dev/video*` periódicamente:

  - cámara USB recién conectada  -> abre captura, empieza inferencia, aparece sola
  - cámara USB desconectada      -> detiene captura y la quita de la interfaz

Las cámaras USB reciben un id estable derivado del nodo (p.ej. `usb-video0`), así
no chocan con los ids del config.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Optional

from ..config import CameraCfg, CaptureCfg
from ..types import Clip
from . import discovery
from .rtsp import CameraCapture

log = logging.getLogger(__name__)


def _usb_camera_id(dev_path: str) -> str:
    # /dev/video0 -> usb-video0
    return "usb-" + dev_path.rsplit("/", 1)[-1]


class CameraManager:
    def __init__(
        self,
        *,
        static_cams: List[CameraCfg],
        model_name: str,
        ring_seconds: float,
        on_clip: Callable[[Clip], None],
        on_frame: Optional[Callable[[str, "object"], None]] = None,
        capture_cfg: Optional[CaptureCfg] = None,
        on_camera_removed: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        rotate_provider: Optional[Callable[[str, int], int]] = None,
    ):
        self._static_cams = list(static_cams)
        self._model_name = model_name
        self._ring_seconds = ring_seconds
        self._on_clip = on_clip
        self._on_frame = on_frame
        self._cfg = capture_cfg or CaptureCfg()
        self._on_camera_removed = on_camera_removed
        self._on_status = on_status
        self._rotate_provider = rotate_provider   # rotación por camera_id (dashboard), en vivo

        self._lock = threading.Lock()
        self.captures: Dict[str, CameraCapture] = {}
        self._static_by_id: Dict[str, CameraCfg] = {}   # cámaras RTSP del config/interfaz
        self._static_lock = threading.Lock()            # serializa recargas de config
        self._usb_by_dev: Dict[str, str] = {}   # /dev/videoN -> camera_id
        self._evaluated: set[str] = set()        # nodos ya probados (añadidos o descartados)
        self._stop = threading.Event()
        self._supervisor: Optional[threading.Thread] = None

    # ── API pública ──────────────────────────────────────────────────────────
    def get_capture(self, camera_id: str) -> Optional[CameraCapture]:
        with self._lock:
            return self.captures.get(camera_id)

    def snapshot(self) -> Dict[str, CameraCapture]:
        with self._lock:
            return dict(self.captures)

    def set_static_cameras(self, cams: List[CameraCfg]) -> None:
        """Recarga en caliente las cámaras RTSP (al guardar en la interfaz):
        arranca las nuevas, detiene las eliminadas y recrea las que cambiaron de
        URL/transporte. Las USB plug-and-play no se tocan."""
        with self._static_lock:
            desired = {c.id: c for c in cams}
            # Bajas o cambios: eliminar lo que ya no está o cambió de url/transport.
            for cid, old in list(self._static_by_id.items()):
                new = desired.get(cid)
                if new is None or new.url != old.url or new.transport != old.transport:
                    self._static_by_id.pop(cid, None)
                    self._remove_capture(cid)
            # Altas (o recreadas tras un cambio).
            for cid, cam in desired.items():
                if cid not in self._static_by_id:
                    self._static_by_id[cid] = cam
                    self._add_capture(cam)

    def start(self) -> None:
        for cam in self._static_cams:
            self._static_by_id[cam.id] = cam
            self._add_capture(cam)
        if self._cfg.usb_autodetect:
            self._supervisor = threading.Thread(
                target=self._scan_loop, name="cam-supervisor", daemon=True)
            self._supervisor.start()
            log.info("Auto-detección USB activa (escaneo cada %.1fs).",
                     self._cfg.usb_scan_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._supervisor:
            self._supervisor.join(timeout=5)
        for cap in self.snapshot().values():
            cap.stop()

    # ── Altas/bajas de capturas ──────────────────────────────────────────────
    def _add_capture(self, cam: CameraCfg) -> None:
        cap = CameraCapture(
            cam=cam,
            model_name=self._model_name,
            ring_seconds=self._ring_seconds,
            on_clip=self._on_clip,
            on_frame=self._on_frame,
            on_status=self._on_status,
            rotate_provider=self._rotate_provider,
            proc_short_side=self._cfg.proc_short_side,
            clip_cooldown_s=self._cfg.clip_cooldown_s,
        )
        with self._lock:
            self.captures[cam.id] = cap
        cap.start()
        log.info("Cámara '%s' iniciada (%s).", cam.id, cam.url)

    def _remove_capture(self, camera_id: str) -> None:
        with self._lock:
            cap = self.captures.pop(camera_id, None)
        if cap is not None:
            cap.stop()
            log.info("Cámara '%s' detenida (desconectada).", camera_id)
            if self._on_camera_removed:
                self._on_camera_removed(camera_id)

    # ── Supervisor plug-and-play ─────────────────────────────────────────────
    def _scan_loop(self) -> None:
        # Escaneo inmediato y luego periódico hasta que se pida parar.
        while True:
            try:
                self._scan_once()
            except Exception:  # noqa: BLE001 - el supervisor nunca debe morir
                log.exception("Fallo escaneando cámaras USB.")
            if self._stop.wait(self._cfg.usb_scan_interval_s):
                return

    def _scan_once(self) -> None:
        present = set(discovery.list_usb_devices())

        # Bajas: nodos que ya no están en /dev.
        for dev in list(self._usb_by_dev):
            if dev not in present:
                cam_id = self._usb_by_dev.pop(dev)
                self._evaluated.discard(dev)
                self._remove_capture(cam_id)

        # Altas: nodos nuevos que no hemos evaluado todavía.
        with self._lock:
            limit = self._cfg.usb_max_devices
            usb_active = len(self._usb_by_dev)
        for dev in present:
            if dev in self._evaluated:
                continue
            if usb_active >= limit:
                log.info("Límite de %d cámaras USB alcanzado; ignoro %s.", limit, dev)
                break
            self._evaluated.add(dev)
            if not discovery.probe_capture(dev):
                continue  # nodo de metadatos u ocupado: no captura
            cam_id = _usb_camera_id(dev)
            self._usb_by_dev[dev] = cam_id
            usb_active += 1
            self._add_capture(CameraCfg(id=cam_id, url=dev, enabled=True))
