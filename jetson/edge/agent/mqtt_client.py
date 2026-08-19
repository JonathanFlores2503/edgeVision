"""Cliente MQTT para telemetría, eventos y heartbeat.

Tolerante a desconexiones: paho mantiene el loop en background y reconecta.
Las publicaciones críticas (eventos) van por el outbox; la telemetría es
best-effort.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

import paho.mqtt.client as mqtt

from ..config import MqttCfg

log = logging.getLogger(__name__)


class MqttClient:
    def __init__(self, cfg: MqttCfg, device_id: str):
        self.cfg = cfg
        self.device_id = device_id
        self._client = mqtt.Client(client_id=f"edge-{device_id}", clean_session=True)
        if cfg.username:
            self._client.username_pw_set(cfg.username, cfg.password)
        if cfg.tls:
            if cfg.tls_ca:
                self._client.tls_set(ca_certs=cfg.tls_ca)
            else:
                self._client.tls_set()
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def topic(self, suffix: str) -> str:
        return f"{self.cfg.base_topic}/{self.device_id}/{suffix}"

    def start(self) -> None:
        try:
            self._client.connect_async(self.cfg.host, self.cfg.port, keepalive=30)
            self._client.loop_start()
            log.info("MQTT conectando a %s:%s", self.cfg.host, self.cfg.port)
        except Exception:  # noqa: BLE001
            log.exception("No se pudo iniciar MQTT")

    def stop(self) -> None:
        self._client.loop_stop()
        self._client.disconnect()

    def publish(self, suffix: str, payload: dict, qos: int = 1) -> bool:
        """Publica un dict como JSON. Devuelve True si se entregó al broker."""
        info = self._client.publish(self.topic(suffix), json.dumps(payload), qos=qos)
        return info.rc == mqtt.MQTT_ERR_SUCCESS

    def _on_connect(self, client, userdata, flags, rc):
        self._connected = rc == 0
        log.info("MQTT %s (rc=%s)", "conectado" if self._connected else "rechazado", rc)

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        log.warning("MQTT desconectado (rc=%s); reconectando...", rc)
