"""Subida de clips de anomalía vía HTTPS (presigned URL).

Flujo (ver shared/contracts/messages.md):
  1. POST .../devices/{id}/clips:presign  -> { upload_url, method, headers }
  2. PUT upload_url  (binario .mp4)
  3. POST .../events/{event_id}/clip-uploaded  { object_key }
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import requests

from ..config import HttpCfg

log = logging.getLogger(__name__)


class ClipUploader:
    def __init__(self, cfg: HttpCfg, device_id: str):
        self.cfg = cfg
        self.device_id = device_id
        self._session = requests.Session()
        if cfg.device_token:
            self._session.headers["Authorization"] = f"Bearer {cfg.device_token}"

    def upload(self, event_id: str, object_key: str, clip_path: str) -> bool:
        path = Path(clip_path)
        if not path.exists():
            log.error("Clip inexistente, no se puede subir: %s", clip_path)
            return False

        size = path.stat().st_size
        try:
            presign = self._session.post(
                f"{self.cfg.base_url}/devices/{self.device_id}/clips:presign",
                json={"event_id": event_id, "object_key": object_key,
                      "content_type": "video/mp4", "size_bytes": size},
                timeout=self.cfg.upload_timeout_s, verify=self.cfg.verify_tls,
            )
            presign.raise_for_status()
            info = presign.json()

            with path.open("rb") as fh:
                put = self._session.request(
                    info.get("method", "PUT"), info["upload_url"], data=fh,
                    headers=info.get("headers", {"Content-Type": "video/mp4"}),
                    timeout=self.cfg.upload_timeout_s, verify=self.cfg.verify_tls,
                )
            put.raise_for_status()

            confirm = self._session.post(
                f"{self.cfg.base_url}/events/{event_id}/clip-uploaded",
                json={"object_key": object_key},
                timeout=self.cfg.upload_timeout_s, verify=self.cfg.verify_tls,
            )
            confirm.raise_for_status()
            log.info("Clip subido para evento %s", event_id)
            return True
        except requests.RequestException as exc:
            log.warning("Fallo subiendo clip %s: %s", event_id, exc)
            return False
