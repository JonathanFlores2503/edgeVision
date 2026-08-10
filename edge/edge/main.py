"""Orquestador del nodo edge.

Cablea todo el pipeline en la Jetson:

  RTSP (4 cám) → ring buffer → InferenceWorker (TransLowNet)
       → telemetría (MQTT, best-effort)
       → EventEngine (umbral+histéresis)
            → on_open  : guarda evento + encola mensaje 'event'
            → on_close : recorta clip del ring buffer + encola 'event'(closed) y 'clip'
       → OutboxDispatcher → MQTT (eventos) / HTTPS (clips)   [offline-first]

Uso:  python -m edge.main --config config.yaml

El modelo de detección es modular (ver edge/inference/registry.py): puede ser un
clip_scorer (familia VAD) que puntúa clips, o un stream_processor (familia
heurística) que procesa frame a frame. La familia/modelo activo se elige desde el
panel de admin y se aplica al reiniciar el nodo.
"""
from __future__ import annotations

import argparse
import errno
import logging
import signal
import threading
import time
from pathlib import Path

from .alerts import AlertNotifier
from .agent.dispatcher import OutboxDispatcher
from .agent.mqtt_client import MqttClient
from .agent.ota import OtaClient
from .agent.uploader import ClipUploader
from .capture.manager import CameraManager
from .config import CameraCfg, EdgeConfig, load_config
from .events.clip import ClipEncoder, object_key_for
from .events.engine import EventEngine
from .gps.neo6m import GpsReader
from .inference.worker import InferenceWorker
from .monitor.server import MonitorServer
from .monitor.settings import Settings, default_settings
from .monitor.state import MonitorState
from .monitor.thresholds import ThresholdProvider
from .monitor.rotations import RotationProvider
from .storage.db import Store
from .types import ClipResult, Event, iso, utcnow

log = logging.getLogger(__name__)


def _event_msg(ev: Event) -> dict:
    return {
        "schema_version": 1,
        "event_id": ev.event_id,
        "device_id": ev.device_id,
        "camera_id": ev.camera_id,
        "state": ev.state,
        "t_start": iso(ev.t_start),
        "t_end": iso(ev.t_end) if ev.t_end else None,
        "max_score": round(ev.max_score, 4),
        "class_id": ev.class_id,
        "class_name": ev.class_name,
        "clip_object_key": ev.clip_object_key,
        "clip_uri": ev.clip_uri,
    }


def _telemetry_msg(device_id: str, r: ClipResult) -> dict:
    return {
        "schema_version": 1,
        "device_id": device_id,
        "camera_id": r.camera_id,
        "ts": iso(r.ts),
        "t_start": iso(r.t_start),
        "t_end": iso(r.t_end),
        "score": round(r.score, 4),
        "class_id": r.class_id,
        "class_probs": [round(p, 4) for p in r.class_probs],
        "latency_ms": round(r.latency_ms, 2),
    }


class EdgeApp:
    def __init__(self, cfg: EdgeConfig, mock: bool = False, config_path: str | None = None):
        self.cfg = cfg
        self._config_path = config_path
        self._stop = threading.Event()
        self._start_ts = time.time()
        self._clip_counter = 0

        # Config editable del dashboard (cámaras, alertas, umbrales por cámara).
        # Se crea ANTES del modelo para poder aplicar la selección de pesos elegida
        # desde admin. Se ancla junto al config.yaml (carpeta `data/` al lado), no al
        # CWD, para que los cambios persistan sin importar desde dónde se lance.
        self.settings = (Settings(self._settings_path(), defaults=default_settings(cfg))
                         if cfg.monitor.enabled else None)
        self._apply_model_override()

        # Núcleo modular. La familia/modelo activo se elige desde admin (registro
        # en edge/inference/registry.py). Con mock=True forzamos el scorer simulado
        # (sin torch ni pesos): sirve para ver cámaras reales, eventos e interfaz.
        from .inference import registry
        if mock:
            log.warning("Modo MOCK: scorer simulado (sin torch/pesos).")
            selection = {"family": "vad", "key": "mock"}
        else:
            selection = self.settings.model_selection() if self.settings else {}
        self._model_label = registry.resolve(selection, cfg.model).label
        try:
            self.model, self.model_kind = registry.build(selection, cfg.model, self.settings)
        except Exception:  # noqa: BLE001 — deps/pesos del modelo elegido pueden faltar
            log.exception("No se pudo construir el modelo elegido (%s). El nodo cae a "
                          "MOCK; instala sus dependencias o cambia el modelo en admin.",
                          selection)
            from .inference.mock import MockModel
            self.model, self.model_kind = MockModel(cfg.model), "clip_scorer"
            self._model_label = f"Mock (fallback de «{self._model_label}»)"
        log.info("Modelo activo: %s (tipo=%s).", self._model_label, self.model_kind)
        self.store = Store(cfg.storage.db_path)
        self.encoder = ClipEncoder(cfg.storage, cfg.events)
        # Umbral de anomalía por cámara: thr propio -> global del dashboard ->
        # events.score_threshold de config.yaml. Lo consultan el motor y el monitor.
        self.thresholds = ThresholdProvider(self.settings, cfg.events.score_threshold)
        # Rotación de imagen por cámara (camera_id -> grados). Se aplica en la captura
        # y la consultan las cámaras RTSP y USB. Editable desde el tile del dashboard.
        self.rotations = RotationProvider(self.settings)

        # Interfaz local: app web servida por el propio nodo (Jetson headless).
        self.monitor = None
        self.monitor_server = None
        self.notifier = None
        if cfg.monitor.enabled:
            self.monitor = MonitorState(self.thresholds.for_camera)
            self.monitor.set_meta(device_id=cfg.device.id, site=cfg.device.site,
                                  model=self._model_label)
            self.notifier = AlertNotifier(self.settings)
            self.monitor_server = MonitorServer(
                self.monitor, host=cfg.monitor.host, port=cfg.monitor.port,
                settings=self.settings,
                on_cameras_changed=self._reload_cameras,
                model_cfg=cfg.model,
                model=self.model,   # editor de zonas si el modelo lo soporta
            )

        # GPS (NEO-6M) -> alimenta el mapa del dashboard
        self.gps = GpsReader(cfg.gps, self.monitor) if self.monitor else None

        # Cloud (opcional)
        self.mqtt = None
        self.uploader = None
        self.ota = None
        if cfg.cloud.enabled:
            if cfg.cloud.mqtt:
                self.mqtt = MqttClient(cfg.cloud.mqtt, cfg.device.id)
            if cfg.cloud.http:
                self.uploader = ClipUploader(cfg.cloud.http, cfg.device.id)
                self.ota = OtaClient(cfg.cloud.http, cfg.cloud.ota, cfg.device.id)
        self.dispatcher = OutboxDispatcher(
            self.store, self.mqtt, self.uploader, cfg.cloud.http
        )

        # Motor de eventos
        self.engine = EventEngine(
            device_id=cfg.device.id,
            cfg=cfg.events,
            class_name_fn=self.model.class_name,
            on_open=self._on_event_open,
            on_close=self._on_event_close,
            threshold_for=self.thresholds.for_camera,
        )

        # Inferencia + captura (RTSP de la interfaz + USB plug-and-play).
        #   • clip_scorer (VAD): un InferenceWorker consume los clips de la cola.
        #   • stream_processor (heurístico): no hay cola de clips; el modelo recibe
        #     cada frame vía `feed` y empuja resultados por sí mismo.
        from .inference.base import KIND_STREAM_PROCESSOR
        self._is_stream = self.model_kind == KIND_STREAM_PROCESSOR
        # Si el modelo streaming dibuja sus propios overlays (bboxes de personas/
        # zonas), la vista en vivo mostrará SU frame anotado en vez del crudo.
        self._stream_draws = False
        if self._is_stream and self.monitor and hasattr(self.model, "set_frame_sink"):
            self.model.set_frame_sink(self.monitor.update_frame)
            self._stream_draws = True
        self.worker = (None if self._is_stream
                       else InferenceWorker(self.model, self._on_result,
                                            max_queue=cfg.capture.infer_queue_max))
        self.manager = CameraManager(
            static_cams=self._configured_cameras(),
            model_name=cfg.model.name,
            ring_seconds=cfg.storage.ring_buffer_s,
            on_clip=(self._ignore_clip if self._is_stream else self.worker.submit),
            on_frame=self._make_frame_tap(),
            capture_cfg=cfg.capture,
            on_camera_removed=(self.monitor.remove_camera if self.monitor else None),
            on_status=(self.monitor.set_camera_status if self.monitor else None),
            rotate_provider=self.rotations.for_camera,
        )

    @staticmethod
    def _ignore_clip(clip) -> bool:
        """No-op para on_clip cuando el modelo es streaming (no usa la cola)."""
        return True

    def _make_frame_tap(self):
        """Callback por frame: alimenta la vista en vivo y, si el modelo es
        streaming, le pasa cada frame (RGB). Devuelve None si no hace falta.

        Si el modelo streaming dibuja sus propios overlays (`_stream_draws`), NO
        empujamos el frame crudo: el modelo empuja el anotado por su frame_sink
        (si no, el crudo taparía las bboxes)."""
        feed = self.model.feed if self._is_stream else None
        push_raw = not getattr(self, "_stream_draws", False)
        update = self.monitor.update_frame if (self.monitor and push_raw) else None
        if feed is None and update is None:
            return None

        def _tap(camera_id, frame_rgb):
            if update is not None:
                update(camera_id, frame_rgb)
            if feed is not None:
                feed(camera_id, frame_rgb, utcnow())

        return _tap

    def _apply_model_override(self) -> None:
        """Aplica la selección de pesos hecha desde admin (solo backend torch).
        El backend ONNX usa exportaciones de nombre fijo, así que se ignora."""
        if self.settings is None or self.cfg.model.backend != "torch":
            return
        ov = self.settings.model_override()
        if not ov:
            return
        wdir = Path(self.cfg.model.weights_dir) / self.cfg.model.name
        det, cls = ov.get("detector"), ov.get("classifier")
        if det and (wdir / det).exists():
            self.cfg.model.weights_detector = det
        if cls and (wdir / cls).exists():
            self.cfg.model.weights_classifier = cls
        log.info("Modelo (override de admin): detector=%s, clasificador=%s",
                 self.cfg.model.weights_detector, self.cfg.model.weights_classifier)

    def _settings_path(self) -> str:
        """Ruta absoluta de settings.json: `data/settings.json` junto al config.yaml
        (no relativa al CWD). Cae a ./data/settings.json si no hay config_path."""
        if self._config_path:
            return str(Path(self._config_path).resolve().parent / "data" / "settings.json")
        return str((Path("./data/settings.json")).resolve())

    def _configured_cameras(self) -> "list[CameraCfg]":
        """Cámaras RTSP a capturar, tomadas de la fuente editable de la interfaz
        (`data/settings.json`) para que lo que el usuario ve/edita en la plataforma
        sea lo que de verdad se captura. Cae a `config.yaml` si no hay monitor."""
        if self.settings is None:
            return list(self.cfg.enabled_cameras)
        out: list[CameraCfg] = []
        for c in self.settings.data().get("cameras", []):
            url = (c.get("url") or "").strip()
            cam_id = (c.get("id") or "").strip()
            if not url or not cam_id or not c.get("enabled", True):
                continue
            out.append(CameraCfg(id=cam_id, url=url, enabled=True,
                                 transport=c.get("transport", "tcp")))
        return out

    def _reload_cameras(self) -> None:
        """Aplica en caliente los cambios guardados desde la interfaz: cámaras y
        umbrales por cámara (sin reiniciar el nodo)."""
        self.manager.set_static_cameras(self._configured_cameras())
        self.thresholds.refresh()
        self.rotations.refresh()

    # ── Callbacks del pipeline ───────────────────────────────────────────────
    def _on_result(self, r: ClipResult) -> None:
        self._clip_counter += 1
        # Telemetría best-effort (no satura el outbox).
        every = self.cfg.cloud.mqtt.telemetry_every_n_clips if self.cfg.cloud.mqtt else 1
        if self.mqtt and self._clip_counter % max(every, 1) == 0:
            self.mqtt.publish("telemetry", _telemetry_msg(self.cfg.device.id, r), qos=0)
        # Monitor web en vivo.
        if self.monitor:
            self.monitor.update_result(r, self.model.class_name(r.class_id))
        # Lógica de eventos.
        self.engine.process(r)

    def _on_event_open(self, ev: Event) -> None:
        ev.clip_object_key = object_key_for(ev)
        self.store.save_event(ev, clip_path=None)
        self.store.enqueue("event", _event_msg(ev), event_id=ev.event_id)
        if self.monitor:
            self.monitor.add_event(ev)
        if self.notifier:
            gps = self.monitor.status().get("gps", {}) if self.monitor else {}
            # Snapshot del momento (frame anotado por el modelo: cajas de personas/
            # zonas) para adjuntarlo a la alerta de Telegram.
            snapshot = self.monitor.get_jpeg(ev.camera_id) if self.monitor else None
            self.notifier.notify_event(
                camera_id=ev.camera_id, class_name=ev.class_name, score=ev.max_score,
                when=iso(ev.t_start), lat=gps.get("lat"), lon=gps.get("lon"),
                image=snapshot)

    def _on_event_close(self, ev: Event) -> None:
        ev.clip_object_key = object_key_for(ev)
        cap = self.manager.get_capture(ev.camera_id)
        clip_path = None
        if cap is not None:
            out = self.encoder.encode(ev, cap.ring)
            clip_path = str(out) if out else None
        self.store.save_event(ev, clip_path=clip_path)
        self.store.enqueue("event", _event_msg(ev), event_id=ev.event_id)
        if self.monitor:
            self.monitor.add_event(ev)
        if clip_path:
            self.store.enqueue(
                "clip",
                {"event_id": ev.event_id, "object_key": ev.clip_object_key, "clip_path": clip_path},
                event_id=ev.event_id,
            )

    # ── Heartbeat ────────────────────────────────────────────────────────────
    def _heartbeat_loop(self):
        while not self._stop.wait(15.0):
            if not self.mqtt:
                continue
            caps = self.manager.snapshot()
            self.mqtt.publish("health", {
                "schema_version": 1,
                "device_id": self.cfg.device.id,
                "ts": iso(utcnow()),
                "uptime_s": int(time.time() - self._start_ts),
                "cameras": {cid: ("online" if len(c.ring) else "starting")
                            for cid, c in caps.items()},
                "model": self.cfg.model.name,
                "fps": {cid: round(c.fps, 1) for cid, c in caps.items()},
                "outbox_pending": self.store.count_pending(),
            }, qos=0)

    # ── Stats en vivo para el dashboard (FPS + costo computacional) ──────────
    def _stats_loop(self):
        """Publica cada 2 s un evento SSE 'stats' con el costo computacional para
        el overlay de las cámaras. La latencia y el FPS los añade MonitorState."""
        try:
            import psutil
        except Exception:  # noqa: BLE001 — sin psutil, solo van FPS/latencia
            psutil = None
        if psutil is not None:
            psutil.cpu_percent(None)  # 1ª llamada inicializa el muestreo
        while not self._stop.wait(2.0):
            cost = {}
            if psutil is not None:
                try:
                    vm = psutil.virtual_memory()
                    cost = {"cpu": round(psutil.cpu_percent(None), 1),
                            "mem": round(vm.percent, 1),
                            "mem_used_mb": round(vm.used / 1e6),
                            "mem_total_mb": round(vm.total / 1e6)}
                except Exception:  # noqa: BLE001
                    cost = {}
            try:
                self.monitor.publish_stats(cost)
            except Exception:  # noqa: BLE001 — el overlay no debe tumbar el nodo
                log.exception("Error publicando stats del dashboard.")

    # ── Ciclo de vida ────────────────────────────────────────────────────────
    def run(self):
        log.info("Iniciando nodo edge '%s' con %d cámara(s) RTSP configurada(s)%s.",
                 self.cfg.device.id, len(self._configured_cameras()),
                 " + auto-detección USB" if self.cfg.capture.usb_autodetect else "")
        # Captura primero: así la recarga en caliente desde la interfaz nunca
        # corre antes de que el manager tenga su estado inicial.
        self.dispatcher.start()
        if self.worker:
            self.worker.start()
        if self._is_stream:
            self.model.start(self._on_result)   # el modelo arranca sus hilos internos
        self.manager.start()
        if self.monitor_server:
            self.monitor_server.start()
            log.info("➜  Interfaz local (LAN): http://<IP-del-equipo>:%d", self.cfg.monitor.port)
        if self.gps:
            self.gps.start()
        if self.mqtt:
            self.mqtt.start()
        if self.ota:
            self.ota.start()
        threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True).start()
        if self.monitor:
            threading.Thread(target=self._stats_loop, name="stats", daemon=True).start()

        # Headless: la interfaz es la app web local servida por el nodo.
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        self._stop.wait()
        self.shutdown()

    def shutdown(self):
        log.info("Apagando nodo edge...")
        self.manager.stop()
        if self.worker:
            self.worker.stop()
        if self._is_stream:
            try:
                self.model.stop()
            except Exception:  # noqa: BLE001 — no debe impedir el apagado
                log.exception("Error deteniendo el modelo streaming.")
        self.dispatcher.stop()
        if self.ota:
            self.ota.stop()
        if self.mqtt:
            self.mqtt.stop()
        if self.gps:
            self.gps.stop()
        if self.monitor_server:
            self.monitor_server.stop()
        self.store.close()
        log.info("Nodo edge detenido.")


def main():
    parser = argparse.ArgumentParser(description="Nodo edge VAD (Jetson)")
    parser.add_argument("--config", default="config.yaml", help="ruta a config.yaml")
    parser.add_argument("--mock", action="store_true",
                        help="scorer simulado (sin torch/pesos): ver cámaras reales, eventos e interfaz")
    parser.add_argument("--force", action="store_true",
                        help="si el puerto del dashboard ya está en uso por OTRO nodo edge, "
                             "lo detiene y arranca este en su lugar (evita el choque de puerto)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    port = cfg.monitor.port if cfg.monitor.enabled else None
    if args.force and port:
        _free_port_from_stale_node(port)
    try:
        EdgeApp(cfg, mock=args.mock, config_path=args.config).run()
    except OSError as e:
        if e.errno == errno.EADDRINUSE and port:
            log.error(
                "El puerto %d ya está en uso: casi seguro hay OTRO nodo edge "
                "corriendo. No arranco un segundo encima. Para verlo y liberarlo:\n"
                "    ss -ltnp | grep :%d        # ver qué proceso lo tiene\n"
                "    pkill -f 'edge.main'        # detener el nodo anterior\n"
                "…o vuelve a lanzar con --force para que yo lo detenga por ti.",
                port, port)
            raise SystemExit(1) from None
        raise


def _free_port_from_stale_node(port: int) -> None:
    """Con --force: mata el nodo edge que tenga tomado `port` para que este arranque
    sin el clásico 'Address already in use'. Solo apunta a procesos `edge.main`
    (no toca otros servicios). Best-effort."""
    import os
    import re
    import signal as _signal
    import subprocess
    me = os.getpid()
    victims: set[int] = set()
    try:  # 1) por dueños del puerto (ss)
        out = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if f":{port} " in line or line.rstrip().endswith(f":{port}"):
                for pid in re.findall(r"pid=(\d+)", line):
                    victims.add(int(pid))
    except Exception:  # noqa: BLE001 — sin ss caemos al método por nombre
        pass
    try:  # 2) por nombre, acotado a edge.main
        out = subprocess.run(["pgrep", "-f", "edge.main"], capture_output=True, text=True).stdout
        victims.update(int(p) for p in out.split())
    except Exception:  # noqa: BLE001
        pass
    victims.discard(me)
    for pid in victims:
        try:
            os.kill(pid, _signal.SIGTERM)
            log.info("[--force] Nodo previo (pid %d) detenido para liberar el puerto %d.", pid, port)
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            log.warning("[--force] No pude detener el pid %d.", pid)
    if victims:
        time.sleep(1.0)  # dar un momento a que el SO libere el socket


if __name__ == "__main__":
    main()
