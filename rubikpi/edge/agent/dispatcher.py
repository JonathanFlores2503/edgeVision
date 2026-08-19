"""Despachador del outbox: entrega al cloud lo acumulado localmente.

Recorre periódicamente la tabla `outbox` y, según el tipo:
  - 'event'  -> publica por MQTT (topic events)
  - 'clip'   -> sube el .mp4 por HTTPS (presigned URL)
Marca cada item como 'sent' o incrementa reintentos. Esto es el corazón del
comportamiento offline-first: si no hay red, todo queda pendiente y se reintenta.
"""
from __future__ import annotations

import json
import logging
import threading

from ..config import HttpCfg
from ..storage.db import Store
from .mqtt_client import MqttClient
from .uploader import ClipUploader

log = logging.getLogger(__name__)


class OutboxDispatcher:
    def __init__(
        self,
        store: Store,
        mqtt: "MqttClient | None",
        uploader: "ClipUploader | None",
        http_cfg: "HttpCfg | None",
        interval_s: float = 2.0,
    ):
        self.store = store
        self.mqtt = mqtt
        self.uploader = uploader
        self.retry_max = http_cfg.retry_max if http_cfg else 5
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dispatcher", daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(self.interval_s):
            try:
                self._drain_once()
            except Exception:  # noqa: BLE001
                log.exception("Fallo drenando outbox")

    def _drain_once(self):
        for row in self.store.pending(limit=50):
            ok = self._handle(row)
            if ok:
                self.store.mark_sent(row["id"])
            else:
                self.store.mark_attempt(row["id"], self.retry_max)

    def _handle(self, row) -> bool:
        kind = row["kind"]
        payload = json.loads(row["payload"])

        if kind in ("event", "telemetry", "health"):
            if self.mqtt is None or not self.mqtt.connected:
                return False
            suffix = {"event": "events", "telemetry": "telemetry", "health": "health"}[kind]
            return self.mqtt.publish(suffix, payload)

        if kind == "clip":
            if self.uploader is None:
                return False
            return self.uploader.upload(
                payload["event_id"], payload["object_key"], payload["clip_path"]
            )

        log.error("Tipo de outbox desconocido: %s", kind)
        return False
