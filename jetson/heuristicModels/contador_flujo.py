# -*- coding: utf-8 -*-
"""Adaptador edgeVision del **Contador de flujo (entradas/salidas)**.

Puente entre la captura multi-cámara de edgeVision y el núcleo de conteo
`PeopleCounterV2Detector` (en `Models/ContadorFlujo/`). El framework web/back
original (Arconte `Base/`) NO se usa: aquí ese rol lo cumple la plataforma
(CameraManager + monitor). Solo reutilizamos la **lógica de conteo**
(YOLO + ByteTrack + corredor de líneas R1–R5 + guardián de luz + FaceBubble).

Contrato `stream_processor`: el nodo llama `start(on_result)` una vez, luego
`feed(cam, rgb, ts)` por cada frame (barato: encola y descarta viejos), y el
trabajo pesado corre en un hilo por cámara. Cada **cruce** (entrada/salida) se
convierte en un `ClipResult` (score 1.0, clase 'entrada'/'salida') que alimenta
el Event Engine; entre cruces se emite un latido 'normal' de score bajo.

Nota: el contador necesita que se **dibuje un corredor (box de 4 puntos) por
cámara**. El dashboard trae un **editor** («✎ Zonas» en cada cámara, solo admin):
la plataforma detecta este modelo por duck-typing (`supports_zones()`) y expone
`get_zones`/`set_zones`, que persisten en `Models/ContadorFlujo/zones_web/
<camera_id>.json` y aplican en caliente si la cámara ya corre. Sin corredor el
modelo corre pero no cuenta.
"""
from __future__ import annotations

import json
import logging
import os.path as osp
import queue
import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import cv2  # noqa: E402  (dep declarada en META)

log = logging.getLogger("heuristic.contador_flujo")

# ── Rutas: el núcleo del modelo vive en Models/ContadorFlujo/ ─────────────────
_EDGE_DIR = Path(__file__).resolve().parents[1]          # .../edge
_CF_DIR = _EDGE_DIR / "Models" / "ContadorFlujo"
_BASE_DIR = _EDGE_DIR / "Base"
_ZONES_DIR = _CF_DIR / "zones_web"                       # corredor por cámara: <key>.json
for _p in (str(_CF_DIR), str(_BASE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# Cola de `eventos.jsonl` que se lee para la galería de testigos. Cada línea ronda
# los 400 bytes, así que 512 KB son más de mil cruces: de sobra para lo que la
# interfaz muestra, y acotado para que el fichero pueda crecer sin límite.
_EVIDENCE_TAIL_BYTES = 512 * 1024


def _safe_key(camera_id) -> str:
    """`camera_id` -> nombre de archivo seguro. La MISMA transformación se usa
    para `det.source_key` y para el `.json` del corredor, de modo que lo que el
    editor guarda es exactamente lo que el detector lee al arrancar."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(camera_id)) or "fuente"


# =============================================================================
#  META — descriptor para el registro modular (leído por AST, sin importar).
# =============================================================================
META = {
    "key": "contador_flujo",
    "label": "Contador de flujo (entradas/salidas)",
    "family": "heuristic",
    "kind": "stream_processor",
    "classes": ["normal", "entrada", "salida"],
    "requires": ["ultralytics", "lap", "cv2", "numpy"],
    "weights": ["yolo11n.pt"],
    "yolo_model": "yolo11n.pt",   # nano: mucho más rápido en CPU/edge (opciones: yolo11n/s/m/l/x)
    "mem_budget_mb": 4000,
    "detail": "Cuenta personas que cruzan una puerta/corredor (YOLO+ByteTrack, "
              "reglas anti-oclusión R1-R5). Requiere dibujar el corredor por cámara.",
    "entry": "build",
}


def _resolve_yolo(name: str) -> str:
    """Ubica los pesos YOLO: edge/<name> → Models/ContadorFlujo/<name> → nombre pelado
    (ultralytics lo descarga solo)."""
    for cand in (_EDGE_DIR / name, _CF_DIR / name):
        if cand.is_file():
            return str(cand)
    return name


class _CamStream:
    """Pipeline por cámara: su propio contador, cola y worker. Convierte cada
    cruce (entrada/salida) en `ClipResult` y lo empuja al Event Engine."""

    QUEUE_MAX = 2            # back-pressure: en tiempo real se descartan frames viejos
    EMIT_INTERVAL_S = 1.0    # latido 'normal' cuando no hay cruce (mantiene telemetría viva)

    def __init__(self, camera_id, detector, on_result, stop_event, frame_sink=None):
        self.camera_id = camera_id
        self._det = detector
        self._on_result = on_result
        self._stop = stop_event
        self._frame_sink = frame_sink
        self._q = queue.Queue(maxsize=self.QUEUE_MAX)
        self._thread = threading.Thread(target=self._run, name=f"contador-{camera_id}", daemon=True)
        self._idx = 0
        self._last_emit = 0.0

    @property
    def detector(self):
        """El detector vivo (para editar el corredor en caliente desde el dashboard)."""
        return self._det

    def start(self):
        self._thread.start()

    def submit(self, frame_bgr, ts):
        try:
            self._q.put_nowait((frame_bgr, ts))
        except queue.Full:                       # descarta el más viejo (tiempo real)
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((frame_bgr, ts))
            except queue.Full:
                pass

    def join(self, timeout=5):
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            frame_bgr, ts = item
            t0 = time.perf_counter()
            in0, out0 = self._det.count_in, self._det.count_out
            try:
                result = self._det.process_frame(frame_bgr, self._idx)
            except Exception:  # noqa: BLE001 — un frame no debe tumbar el worker
                log.exception("process_frame del contador [%s] falló", self.camera_id)
                continue
            self._idx += 1
            latency_ms = (time.perf_counter() - t0) * 1000.0
            din = self._det.count_in - in0
            dout = self._det.count_out - out0

            # Vista en vivo con el corredor, tracks y marcador ENTRADAS/SALIDAS/DENTRO.
            if self._frame_sink is not None:
                try:
                    annotated = self._det.annotate(frame_bgr, result, {})
                    self._frame_sink(self.camera_id,
                                     cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                except Exception:  # noqa: BLE001 — el overlay no debe tumbar el worker
                    log.exception("annotate del contador [%s] falló", self.camera_id)

            # Cruces → eventos; si no hubo, latido 'normal' cada EMIT_INTERVAL_S.
            if din > 0:
                self._emit(ts, latency_ms, class_id=1, score=1.0)   # entrada
            if dout > 0:
                self._emit(ts, latency_ms, class_id=2, score=1.0)   # salida
            now = time.monotonic()
            if din == 0 and dout == 0 and (now - self._last_emit) >= self.EMIT_INTERVAL_S:
                self._emit(ts, latency_ms, class_id=0, score=0.0)   # normal (latido)

    def _emit(self, ts, latency_ms, class_id, score):
        from edge.types import ClipResult
        probs = [0.0, 0.0, 0.0]
        probs[class_id] = 1.0
        self._last_emit = time.monotonic()
        self._on_result(ClipResult(
            camera_id=self.camera_id, t_start=ts, t_end=ts,
            score=score, class_id=class_id, class_probs=probs,
            latency_ms=latency_ms,
        ))


class Model:
    """Adaptador `stream_processor` multi-cámara: un contador por cámara, creado
    perezosamente al llegar el 1er frame (en un hilo aparte, para no bloquear la
    captura mientras carga YOLO)."""

    EST_PER_CAM_MB = 500           # reserva estimada por detector nuevo (YOLO)

    def __init__(self, cfg=None):
        self._cfg = cfg
        self._on_result = None
        self._frame_sink = None
        self._cams: Dict[str, _CamStream] = {}
        self._loading: set = set()
        self._denied: set = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._budget_mb = int(META.get("mem_budget_mb", 4000))
        self._yolo = _resolve_yolo(META.get("yolo_model", "yolo11n.pt"))

    # ── contrato StreamModel ─────────────────────────────────────────────────
    def set_frame_sink(self, fn):
        """El nodo pasa aquí su `monitor.update_frame` para la vista en vivo anotada."""
        self._frame_sink = fn

    def start(self, on_result):
        self._on_result = on_result
        log.info("[CONTADOR] Multi-cámara listo. YOLO=%s presupuesto RAM=%dMB.",
                 osp.basename(self._yolo), self._budget_mb)

    def feed(self, camera_id, frame_rgb, ts):
        cam = self._cams.get(camera_id)
        if cam is None:
            self._ensure_cam(camera_id)        # dispara carga en bg; se dropea este frame
            return
        cam.submit(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), ts)   # RGB->BGR (cv2/YOLO)

    def stop(self):
        self._stop.set()
        with self._lock:
            cams = list(self._cams.values())
        for c in cams:
            c.join(timeout=5)

    def class_name(self, class_id):
        names = META["classes"]
        return names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"

    # ── editor de corredor por cámara (lo consume el dashboard) ───────────────
    # La plataforma es modelo-agnóstica: detecta este protocolo por duck-typing
    # (`supports_zones()`), muestra el botón «✎ Zonas» y delega get/set aquí. El
    # corredor es un box de 4 puntos en píxeles del frame que ve el modelo.
    def supports_zones(self) -> bool:
        return True

    def zone_schema(self) -> dict:
        """Metadatos para el editor: qué figura y opciones espera este modelo."""
        return {
            "shape": "box4",                # exactamente 1 polígono de 4 puntos
            "coords": "pixels",             # px del frame que procesa el modelo
            "options": {
                "n_lines": {"type": "int", "min": 1, "max": 9, "default": 5,
                            "label": "Líneas de la escalera (anti-oclusión)"},
                "invert": {"type": "bool", "default": False,
                           "label": "Invertir entrada/salida"},
                "axis_rot": {"type": "bool", "default": False,
                             "label": "Rotar eje 90° (orientación de líneas)"},
            },
            "help": "Dibuja un box de 4 puntos sobre la puerta/corredor. Los 2 "
                    "primeros puntos definen el lado AFUERA; el eje va de ese lado "
                    "al opuesto (ADENTRO).",
        }

    # ── conteos en vivo (los consume el dashboard: entraron/salieron/dentro) ──
    # También duck-typed: la plataforma detecta `supports_counts()` y muestra el
    # marcador por cámara. Cada detector mantiene sus totales persistentes.
    def supports_counts(self) -> bool:
        return True

    def get_counts(self) -> dict:
        """{camera_id: {in, out, inside}} de cada cámara cargada."""
        with self._lock:
            cams = dict(self._cams)
        out = {}
        for cid, cam in cams.items():
            try:
                c = cam.detector.get_counts()   # {inside, in, out, offset}
                out[cid] = {"in": int(c.get("in", 0)),
                            "out": int(c.get("out", 0)),
                            "inside": int(c.get("inside", 0))}
            except Exception:  # noqa: BLE001 — una cámara sin datos no rompe el resto
                pass
        return out

    @staticmethod
    def _counts_path(key: str) -> Path:
        return _CF_DIR / "counters" / f"{key}.json"

    def set_counts(self, camera_id, inside) -> dict:
        """Fija cuántas personas hay ADENTRO de una cámara (reinicia entradas/
        salidas y usa `inside` como base). Aplica en caliente si la cámara ya
        corre; si no, persiste en `counters/<key>.json` para que el detector lo
        tome al arrancar. Enviar 0 equivale a reiniciar el contador."""
        try:
            n = max(0, int(inside))
        except (TypeError, ValueError):
            return {"ok": False, "error": "inside debe ser un entero >= 0"}
        cam = self._cams.get(camera_id)
        if cam is not None:                    # detector vivo: fija en caliente
            try:
                cam.detector.set_inside(n)     # reinicia in/out, offset=n, persiste
                return {"ok": True, "applied_live": True, "inside": n}
            except Exception as e:  # noqa: BLE001
                log.exception("[CONTADOR] set_inside en vivo de '%s' falló", camera_id)
                return {"ok": False, "error": f"no se pudo aplicar: {e}"}
        # Cámara aún no cargada: escribe el mismo json que lee el detector al
        # arrancar ({in, out, offset}); se aplicará cuando la cámara arranque.
        key = _safe_key(camera_id)
        p = self._counts_path(key)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps({"in": 0, "out": 0, "offset": n}),
                         encoding="utf-8")
        except OSError as e:
            return {"ok": False, "error": f"no se pudo guardar: {e}"}
        log.info("[CONTADOR] Conteo de '%s' fijado a %d dentro (cámara no activa "
                 "aún; se aplicará al arrancar).", camera_id, n)
        return {"ok": True, "applied_live": False, "inside": n}

    # ── testigos: la evidencia fotográfica de cada cruce ──────────────────────
    # Tercer protocolo duck-typed, igual que zonas y conteos: la plataforma
    # detecta `supports_evidence()` y muestra la galería en el panel de admin.
    #
    # El núcleo ya guardaba todo esto desde el principio en
    # `EVENTOS_CONTADOR/<key>/` (foto del fotograma con la caja dibujada, recorte
    # de la persona, rostro si FaceBubble lo pilló, y una línea en `eventos.jsonl`);
    # lo que faltaba era poder verlo sin entrar por SSH. Aquí solo se LEE.
    def supports_evidence(self) -> bool:
        return True

    @staticmethod
    def _evidence_dir(key: str) -> Path:
        return _CF_DIR / "EVENTOS_CONTADOR" / key

    def evidence_cameras(self) -> list:
        """Cámaras con evidencia en disco. No depende de que estén corriendo:
        los testigos de ayer se listan igual que los de hace un minuto."""
        root = _CF_DIR / "EVENTOS_CONTADOR"
        if not root.is_dir():
            return []
        return sorted(d.name for d in root.iterdir()
                      if d.is_dir() and (d / "eventos.jsonl").is_file())

    def list_evidence(self, camera_id: str = "", limit: int = 60) -> dict:
        """Últimos cruces con foto, del más reciente al más antiguo.

        Lee la cola de `eventos.jsonl` en vez del fichero entero: crece una línea
        por cruce y en una puerta con tráfico llega a decenas de miles."""
        cams = self.evidence_cameras()
        key = _safe_key(camera_id) if camera_id else (cams[0] if cams else "")
        out = {"supported": True, "cameras": cams, "camera": key, "items": []}
        if not key:
            return out

        path = self._evidence_dir(key) / "eventos.jsonl"
        if not path.is_file():
            return out

        try:
            with path.open("rb") as fh:
                fh.seek(0, 2)
                size = fh.tell()
                back = min(size, _EVIDENCE_TAIL_BYTES)
                fh.seek(size - back)
                raw = fh.read().decode("utf-8", "replace")
        except OSError as e:
            log.warning("[CONTADOR] No se pudo leer la evidencia de '%s': %s", key, e)
            return out

        lines = raw.splitlines()
        if back < size and lines:
            lines = lines[1:]      # la primera puede venir cortada por la mitad

        try:
            n = max(1, min(int(limit), 400))
        except (TypeError, ValueError):
            n = 60

        d = self._evidence_dir(key)
        items = []
        for line in reversed(lines):
            if len(items) >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                meta = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(meta, dict):
                continue
            # Solo se anuncian los ficheros que de verdad están: la evidencia se
            # borra a mano de vez en cuando y el jsonl no se entera.
            fotos = {}
            for campo in ("foto", "crop", "rostro"):
                nombre = meta.get(campo)
                if nombre and (d / nombre).is_file():
                    fotos[campo] = nombre
            items.append({
                "timestamp": meta.get("timestamp", ""),
                "evento": meta.get("evento", ""),
                "track_id": meta.get("track_id"),
                "regla": meta.get("regla", ""),
                "lineas": meta.get("lineas", ""),
                "dentro": meta.get("dentro"),
                "entradas": meta.get("entradas"),
                "salidas": meta.get("salidas"),
                "rostro_nombre": meta.get("rostro_nombre"),
                "fotos": fotos,
            })
        out["items"] = items
        return out

    def evidence_file(self, camera_id: str, filename: str):
        """Ruta absoluta de una foto de testigo, o `None` si no vale.

        El dashboard sirve estos bytes al navegador, así que el nombre llega
        desde fuera: se rechaza cualquiera con separadores, y además se
        comprueba que la ruta ya resuelta siga dentro de la carpeta de esa
        cámara. Lo primero para el caso obvio, lo segundo por si un enlace
        simbólico apunta a otro sitio."""
        nombre = (filename or "").strip()
        if (not nombre or "/" in nombre or "\\" in nombre or nombre.startswith(".")
                or not nombre.lower().endswith(".jpg")):
            return None
        d = self._evidence_dir(_safe_key(camera_id))
        try:
            p = (d / nombre).resolve()
            if p.parent != d.resolve():
                return None
        except OSError:
            return None
        return p if p.is_file() else None

    @staticmethod
    def _zone_path(key: str) -> Path:
        return _ZONES_DIR / f"{key}.json"

    def get_zones(self, camera_id) -> dict:
        """Corredor actual de una cámara. Si está cargada, la verdad viva del
        detector (incluye ediciones en caliente); si no, lo guardado en disco."""
        key = _safe_key(camera_id)
        cam = self._cams.get(camera_id)
        if cam is not None:
            det = cam.detector
            opts = det.corridor_opts()
            return {"supported": True, "loaded": True,
                    "regions": det.get_zones() or [],
                    "n_lines": int(opts.get("n_lines", 5)),
                    "invert": bool(opts.get("invert", False)),
                    "axis_rot": bool(opts.get("axis_rot", False))}
        data = {}
        p = self._zone_path(key)
        if p.is_file():
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — un json corrupto no debe tumbar la API
                log.exception("[CONTADOR] json de corredor ilegible: %s", p)
        return {"supported": True, "loaded": False,
                "regions": data.get("regions", []),
                "n_lines": int(data.get("n_lines", 5)),
                "invert": bool(data.get("invert", False)),
                "axis_rot": bool(data.get("axis_rot", False))}

    def set_zones(self, camera_id, regions, opts=None) -> dict:
        """Guarda el corredor en disco y lo aplica en caliente si la cámara ya
        corre (sin reiniciar el nodo). Valida: exactamente 1 box de 4 puntos."""
        opts = opts or {}
        try:
            regions = [[[int(round(float(p[0]))), int(round(float(p[1])))]
                        for p in r] for r in (regions or [])]
        except (TypeError, ValueError, IndexError):
            return {"ok": False, "error": "regions inválidas (lista de puntos [x,y])"}
        if len(regions) != 1 or len(regions[0]) != 4:
            return {"ok": False, "error": "dibuja un SOLO box de 4 puntos"}
        n_lines = max(1, min(9, int(opts.get("n_lines", 5))))
        invert = bool(opts.get("invert", False))
        axis_rot = bool(opts.get("axis_rot", False))
        key = _safe_key(camera_id)
        payload = {"regions": regions, "n_lines": n_lines,
                   "invert": invert, "axis_rot": axis_rot, "source": key}
        try:
            _ZONES_DIR.mkdir(parents=True, exist_ok=True)
            self._zone_path(key).write_text(json.dumps(payload, indent=2),
                                            encoding="utf-8")
        except OSError as e:
            return {"ok": False, "error": f"no se pudo guardar: {e}"}
        applied = False
        cam = self._cams.get(camera_id)
        if cam is not None:                    # hot-reload del detector vivo
            try:
                cam.detector.set_zones(regions, opts={
                    "n_lines": n_lines, "invert": invert, "axis_rot": axis_rot})
                applied = True
            except Exception:  # noqa: BLE001 — el guardado en disco ya persistió
                log.exception("[CONTADOR] hot-reload del corredor de '%s' falló", camera_id)
        log.info("[CONTADOR] Corredor de '%s' guardado (%s líneas, invert=%s, "
                 "axis_rot=%s%s).", camera_id, n_lines, invert, axis_rot,
                 ", aplicado en vivo" if applied else "")
        return {"ok": True, "applied_live": applied}

    # ── gestión de detectores por cámara ─────────────────────────────────────
    @staticmethod
    def _rss_mb() -> float:
        try:
            import os
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:  # noqa: BLE001 — sin psutil no aplicamos presupuesto
            return 0.0

    def _ensure_cam(self, camera_id):
        with self._lock:
            if camera_id in self._cams or camera_id in self._loading \
                    or camera_id in self._denied:
                return
            rss = self._rss_mb()
            if self._cams and rss and (rss + self.EST_PER_CAM_MB) > self._budget_mb:
                self._denied.add(camera_id)
                log.warning("[CONTADOR] Sin presupuesto de RAM para '%s' (%.0fMB + ~%dMB "
                            "> tope %dMB); se omite.", camera_id, rss,
                            self.EST_PER_CAM_MB, self._budget_mb)
                return
            self._loading.add(camera_id)
        threading.Thread(target=self._load_cam, args=(camera_id,),
                         name=f"contador-load-{camera_id}", daemon=True).start()

    def _load_cam(self, camera_id):
        try:
            log.info("[CONTADOR] Cargando contador para '%s' (YOLO=%s)...",
                     camera_id, osp.basename(self._yolo))
            from PeopleCounter_V2_Web import PeopleCounterV2Detector  # import pesado (YOLO en setup)
            det = PeopleCounterV2Detector()
            args = SimpleNamespace(model=self._yolo, invert=False,
                                   full_frame=False, zone_expand=0)
            det.source_key = _safe_key(camera_id)  # carga zones_web/<key>.json si existe
            det.configure(args)
            det.setup()
            cam = _CamStream(camera_id, det, self._on_result, self._stop,
                             frame_sink=self._frame_sink)
            cam.start()
        except Exception:  # noqa: BLE001
            log.exception("[CONTADOR] No se pudo cargar el contador de '%s'", camera_id)
            with self._lock:
                self._loading.discard(camera_id)
            return
        with self._lock:
            self._cams[camera_id] = cam
            self._loading.discard(camera_id)
        log.info("[CONTADOR] '%s' activa (RAM del proceso %.0fMB).",
                 camera_id, self._rss_mb())


def build(cfg=None):
    """Factory que usa el registro modular para construir el modelo."""
    return Model(cfg)
