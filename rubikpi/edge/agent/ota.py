"""Cliente OTA: consulta periódicamente la versión de pesos activa para este
dispositivo y descarga los artefactos nuevos.

El despliegue real (recargar el modelo en caliente) se decide en fase posterior;
aquí se deja el mecanismo de poll + descarga a un directorio local.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import requests

from ..config import HttpCfg, OtaCfg

log = logging.getLogger(__name__)


class OtaClient:
    def __init__(self, http: HttpCfg, ota: OtaCfg, device_id: str, dest_dir: str = "./data/ota"):
        self.http = http
        self.ota = ota
        self.device_id = device_id
        self.dest = Path(dest_dir)
        self.dest.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        if http.device_token:
            self._session.headers["Authorization"] = f"Bearer {http.device_token}"
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="ota", daemon=True)
        self._current_version: Optional[str] = None

    def start(self):
        if self.ota.enabled:
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.ota.poll_interval_s):
            try:
                self._check_once()
            except requests.exceptions.RequestException as e:
                # Fallos de red esperables (DNS, timeout, host caído, HTTP error):
                # una línea, sin volcar el traceback en cada poll. Se reintenta solo.
                log.warning("Poll OTA falló (reintenta en %ss): %s",
                            self.ota.poll_interval_s, e)
            except Exception:  # noqa: BLE001
                log.exception("Fallo inesperado en poll OTA")

    def _check_once(self):
        resp = self._session.get(
            f"{self.http.base_url}/devices/{self.device_id}/model",
            timeout=30, verify=self.http.verify_tls,
        )
        resp.raise_for_status()
        info = resp.json()  # { version, artifacts: [{name, url}] }
        version = info.get("version")
        if version == self._current_version:
            return
        log.info("OTA: nueva versión de modelo disponible: %s", version)
        for art in info.get("artifacts", []):
            self._download(art["url"], self.dest / art["name"])
        self._current_version = version
        log.info("OTA: artefactos descargados en %s (requiere recarga del modelo).", self.dest)

    def _download(self, url: str, dest: Path):
        with self._session.get(url, stream=True, timeout=120, verify=self.http.verify_tls) as r:
            r.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
