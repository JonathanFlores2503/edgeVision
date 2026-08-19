"""Worker de inferencia: un único consumidor GPU para todas las cámaras.

Las cámaras producen Clips en una cola; este worker los procesa en serie
(evita saturar la VRAM) y entrega cada ClipResult vía callback.
"""
from __future__ import annotations

import logging
import queue
import threading
from typing import TYPE_CHECKING, Callable

from ..types import Clip, ClipResult

if TYPE_CHECKING:  # solo para type hints; evita importar torch en modo mock
    from .model import TransLowNet

log = logging.getLogger(__name__)


class InferenceWorker:
    def __init__(
        self,
        model: "TransLowNet",
        on_result: Callable[[ClipResult], None],
        max_queue: int = 4,
    ):
        self.model = model
        self.on_result = on_result
        self.queue: "queue.Queue[Clip]" = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="inference", daemon=True)

    def submit(self, clip: Clip) -> bool:
        """Encola un clip. Si la cola está llena, descarta (back-pressure) y avisa."""
        try:
            self.queue.put_nowait(clip)
            return True
        except queue.Full:
            log.warning("Cola de inferencia llena; se descarta clip de %s", clip.camera_id)
            return False

    def start(self):
        self._thread.start()
        log.info("InferenceWorker iniciado.")

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=10)

    def _run(self):
        while not self._stop.is_set():
            try:
                clip = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = self.model.infer(clip)
                self.on_result(result)
            except Exception:  # noqa: BLE001 - un clip no debe tumbar el worker
                log.exception("Error infiriendo clip de %s", clip.camera_id)
            finally:
                self.queue.task_done()
