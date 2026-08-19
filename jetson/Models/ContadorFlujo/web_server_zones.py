# -*- coding: utf-8 -*-
"""
web_server_zones.py
===================
Panel web multi-fuente (basado en Base/web_server.py) + EDITOR DE ZONAS en el
navegador (version web de AreaRestriction_ZoneCreator.py).

Cada tarjeta tiene un boton "Zonas" que abre un editor sobre un snapshot del
stream:
  - Click izquierdo: agregar punto (minimo 3 por zona, hasta 6 zonas)
  - Click derecho sobre una zona: borrarla
  - Botones: Cerrar zona / Deshacer punto / Limpiar / Guardar / Cancelar

Las zonas se guardan POR FUENTE en zones_web/<fuente>.json (mismo formato que
el ZoneCreator: {"regions": [...], "created": ..., "source": ...}) y se
recargan EN CALIENTE en el detector de esa fuente (sin reiniciar).

Endpoints extra sobre los de Base/web_server.py:
  GET  /api/sources/<id>/frame   → snapshot JPEG actual (para el editor)
  GET  /api/sources/<id>/zones   → {"regions": [...]} zonas actuales
  POST /api/sources/<id>/zones   → guarda + hot-reload {"regions": [...]}

El detector debe implementar (opcional):
  .set_zones(regions)  → hot-reload de zonas
  .get_zones()         → zonas actuales
y recibe .source_key (str) antes de setup() para cargar sus zonas por fuente.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "Base"))
sys.path.insert(0, BASE_DIR)

from flask import Flask, Response, jsonify, request, send_from_directory  # noqa: E402
from werkzeug.utils import secure_filename  # noqa: E402

from web_server import SourceWorker, _parse_resize  # noqa: E402  (Base/)

ZONES_WEB_DIR  = os.path.join(SCRIPT_DIR, "zones_web")
PARAMS_WEB_DIR = os.path.join(SCRIPT_DIR, "params_web")
MAX_ZONES  = 6
MIN_POINTS = 2   # 2 puntos = LINEA (contador); 3+ = poligono (area)


def source_key(source) -> str:
    """Nombre de archivo estable por fuente (rtsp url / path / webcam)."""
    key = secure_filename(str(source))
    return key or "fuente"


# ====================================================================== #
#  LOGS en la plataforma: tee de stdout/stderr a un ring-buffer en memoria
# ====================================================================== #
_log_lock = threading.Lock()
_log_buf  = deque(maxlen=600)
_log_seq  = 0


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _log_append(text: str):
    global _log_seq
    text = _ANSI_RE.sub("", text).rstrip()
    if not text:
        return
    # filtrar access-log de werkzeug (GET /api/... cada 2s ensuciaria todo)
    if " - - [" in text and "HTTP/1." in text:
        return
    with _log_lock:
        _log_seq += 1
        _log_buf.append({"n": _log_seq,
                         "t": datetime.now().strftime("%H:%M:%S"),
                         "line": text})


class _Tee:
    """Escribe al stream original Y al buffer de logs (linea por linea)."""
    def __init__(self, orig):
        self._orig = orig
        self._buf = ""

    def write(self, s):
        try:
            self._orig.write(s)
        except Exception:
            pass
        self._buf += s
        while True:
            idx = min((i for i in (self._buf.find("\n"), self._buf.find("\r"))
                       if i >= 0), default=-1)
            if idx < 0:
                break
            line, self._buf = self._buf[:idx], self._buf[idx + 1:]
            _log_append(line)
        return len(s)

    def flush(self):
        try:
            self._orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False


def install_log_capture():
    if not isinstance(sys.stdout, _Tee):
        sys.stdout = _Tee(sys.stdout)
    if not isinstance(sys.stderr, _Tee):
        sys.stderr = _Tee(sys.stderr)


# ====================================================================== #
#  Costo computacional (VRAM / GPU / RAM / CPU) — para mostrar en la UI
# ====================================================================== #
_sys_cache = {"t": 0.0, "data": {}}
_sys_lock = threading.Lock()


def _read_system_stats() -> dict:
    stats = {"gpu_name": "", "vram_used_mb": None, "vram_total_mb": None,
             "gpu_util": None, "vram_proc_mb": None,
             "ram_proc_mb": None, "cpu_load": None, "cpu_cores": os.cpu_count()}

    # GPU global + VRAM de ESTE proceso (nvidia-smi)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip().splitlines()
        if out:
            name, used, total, util = [v.strip() for v in out[0].split(",")]
            stats.update(gpu_name=name, vram_used_mb=int(used),
                         vram_total_mb=int(total), gpu_util=int(util))
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout.strip().splitlines()
        pid = os.getpid()
        for line in out:
            parts = [v.strip() for v in line.split(",")]
            if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
                stats["vram_proc_mb"] = int(parts[1])
                break
    except Exception:
        pass

    # RAM RSS de este proceso
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    stats["ram_proc_mb"] = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass

    # Carga de CPU (load average 1 min)
    try:
        stats["cpu_load"] = round(os.getloadavg()[0], 1)
    except Exception:
        pass

    return stats


def get_system_stats() -> dict:
    """Con cache de 2 s para no martillar nvidia-smi con cada refresh."""
    with _sys_lock:
        now = time.time()
        if now - _sys_cache["t"] > 2.0:
            _sys_cache["data"] = _read_system_stats()
            _sys_cache["t"] = now
        return _sys_cache["data"]


# ====================================================================== #
#  App Flask (Base/web_server.create_app + endpoints/UI de zonas)
# ====================================================================== #
def _normalize_detectors(detector_cls) -> dict:
    """Acepta una clase o un dict {NOMBRE: clase} → dict ordenado."""
    if isinstance(detector_cls, dict):
        return dict(detector_cls)
    return {detector_cls.NAME: detector_cls}


def create_app(detector_cls, args):
    detectors = _normalize_detectors(detector_cls)
    default_type = next(iter(detectors))
    title = " + ".join(detectors)
    # Clase que aporta la galeria ReID (si alguna la tiene)
    gallery_cls = next((c for c in detectors.values()
                        if callable(getattr(c, "gallery_info", None))), None)

    app = Flask(title)
    workers = {}
    lock = threading.Lock()
    os.makedirs(args.upload_dir, exist_ok=True)
    os.makedirs(ZONES_WEB_DIR, exist_ok=True)
    os.makedirs(PARAMS_WEB_DIR, exist_ok=True)

    # ------------------------------------------------------------------ #
    def _persist():
        try:
            with lock:
                data = [{"source": str(w.source), "name": w.name,
                         "type": w.detector.NAME}
                        for w in workers.values()]
            with open(args.config, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[web] No se pudo guardar {args.config}: {e}")

    def _add_source(source, name="", dtype=None):
        cls = detectors.get(dtype or "") or detectors[default_type]
        det = cls()
        det.configure(args)
        # zonas por fuente, con namespace por modelo (mismo rtsp puede tener
        # zonas distintas en area restringida y en el contador)
        det.source_key = f"{det.NAME}__{source_key(source)}"
        # parametros guardados desde el panel → re-aplicar (opcional)
        if callable(getattr(det, "set_params", None)):
            ppath = os.path.join(PARAMS_WEB_DIR, f"{det.source_key}.json")
            if os.path.isfile(ppath):
                try:
                    with open(ppath) as f:
                        saved = json.load(f)
                    det.set_params(saved)
                    print(f"[params] Aplicados {len(saved)} parametro(s) "
                          f"guardados: {ppath}")
                except Exception as e:
                    print(f"[params] No se pudo aplicar {ppath}: {e}")
        sid = uuid.uuid4().hex[:8]
        w = SourceWorker(sid, source, det, args, name=name).start()
        with lock:
            workers[sid] = w
        print(f"[web] Fuente agregada [{sid}] ({det.NAME}): {source}")
        return sid

    def _load_config():
        if not os.path.isfile(args.config):
            return
        try:
            with open(args.config) as f:
                for item in json.load(f):
                    _add_source(item["source"], item.get("name", ""),
                                item.get("type"))
        except Exception as e:
            print(f"[web] No se pudo cargar {args.config}: {e}")

    def _get_worker(sid):
        with lock:
            return workers.get(sid)

    # ------------------------------------------------------------------ #
    @app.route("/")
    def index():
        opts = "".join(f'<option value="{n}">{n}</option>' for n in detectors)
        html = (_HTML.replace("%%NAME%%", title)
                     .replace("%%TYPE_OPTIONS%%", opts))
        return Response(html, mimetype="text/html")

    @app.route("/api/system")
    def system_stats():
        return jsonify(get_system_stats())

    @app.route("/api/logs")
    def get_logs():
        try:
            since = int(request.args.get("since", 0))
        except ValueError:
            since = 0
        with _log_lock:
            lines = [e for e in _log_buf if e["n"] > since]
            last = _log_seq
        return jsonify({"lines": lines, "last": last})

    @app.route("/api/sources", methods=["GET"])
    def list_sources():
        with lock:
            items = [{"id": sid, "name": w.name, "type": w.detector.NAME,
                      "counter": callable(getattr(w.detector, "set_inside",
                                                  None)),
                      "params": callable(getattr(w.detector, "get_params",
                                                 None)),
                      **w.stats}
                     for sid, w in workers.items()]
        return jsonify(items)

    @app.route("/api/sources/<sid>/counter", methods=["GET", "POST"])
    def counter(sid):
        """Contador de personas: GET = estado; POST {"inside": N} = fijar
        cuantas personas hay dentro ahora (in/out arrancan de cero)."""
        w = _get_worker(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        det = w.detector
        if not callable(getattr(det, "set_inside", None)):
            return jsonify({"error": "Esta fuente no es un contador"}), 400
        if request.method == "POST":
            data = request.get_json(force=True, silent=True) or {}
            try:
                n = int(data.get("inside"))
            except (TypeError, ValueError):
                return jsonify({"error": "'inside' debe ser entero"}), 400
            if n < 0:
                return jsonify({"error": "'inside' debe ser >= 0"}), 400
            det.set_inside(n)
        return jsonify(det.get_counts())

    @app.route("/api/sources", methods=["POST"])
    def add_source():
        data = request.get_json(force=True, silent=True) or {}
        source = (data.get("source") or "").strip()
        if not source:
            return jsonify({"error": "Falta 'source'"}), 400
        sid = _add_source(source, data.get("name", ""), data.get("type"))
        _persist()
        return jsonify({"id": sid})

    @app.route("/api/sources/<sid>/type", methods=["POST"])
    def change_type(sid):
        """Cambia el modelo de una fuente: para el worker y lo relanza
        con el detector nuevo (las zonas de cada modelo se conservan)."""
        data = request.get_json(force=True, silent=True) or {}
        dtype = data.get("type")
        if dtype not in detectors:
            return jsonify({"error": f"Modelo desconocido: {dtype}"}), 400
        with lock:
            w = workers.get(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        if w.detector.NAME == dtype:
            return jsonify({"id": sid, "unchanged": True})
        with lock:
            workers.pop(sid, None)
        print(f"[web] Cambiando modelo de '{w.name}': "
              f"{w.detector.NAME} → {dtype}")
        w.stop()
        new_sid = _add_source(w.source, w.name, dtype)
        _persist()
        return jsonify({"id": new_sid})

    @app.route("/api/sources/<sid>", methods=["DELETE"])
    def remove_source(sid):
        with lock:
            w = workers.pop(sid, None)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        w.stop()
        _persist()
        print(f"[web] Fuente eliminada [{sid}]")
        return jsonify({"ok": True})

    @app.route("/api/upload", methods=["POST"])
    def upload():
        f = request.files.get("file")
        if f is None or not f.filename:
            return jsonify({"error": "Falta el archivo"}), 400
        fname = secure_filename(f.filename)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.abspath(os.path.join(args.upload_dir, f"{ts}_{fname}"))
        f.save(path)
        print(f"[web] Video subido: {path}")
        sid = _add_source(path, name=fname, dtype=request.form.get("type"))
        _persist()
        return jsonify({"id": sid, "path": path})

    @app.route("/api/sources/<sid>/record", methods=["POST"])
    def record(sid):
        w = _get_worker(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        data = request.get_json(force=True, silent=True) or {}
        if data.get("action") == "stop":
            path = w.stop_recording()
            print(f"[web] Grabacion detenida [{sid}]: {path}")
            return jsonify({"ok": True, "path": path})
        path = w.start_recording(data.get("name", ""))
        print(f"[web] Grabando [{sid}] -> {path}")
        return jsonify({"ok": True, "path": path})

    @app.route("/api/sources/<sid>/mjpeg")
    def mjpeg(sid):
        def gen():
            last = -1
            while True:
                w = _get_worker(sid)
                if w is None:
                    break
                jpg, version = w.holder.get()
                if jpg is None or version == last:
                    time.sleep(0.03)
                    continue
                last = version
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       + f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                       + jpg + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ------------------------------------------------------------------ #
    #  ZONAS (nuevo vs Base): snapshot + get/set con hot-reload
    # ------------------------------------------------------------------ #
    @app.route("/api/sources/<sid>/frame")
    def frame(sid):
        w = _get_worker(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        jpg, _ = w.holder.get()
        if jpg is None:
            return jsonify({"error": "Sin frame aun (fuente conectando)"}), 503
        return Response(jpg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/api/sources/<sid>/zones", methods=["GET"])
    def get_zones(sid):
        w = _get_worker(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        getter = getattr(w.detector, "get_zones", None)
        regions = getter() if callable(getter) else []
        out = {
            "regions": regions,
            "zone_expand": getattr(w.detector, "zone_expand", None),
            "full_frame": bool(getattr(w.detector, "full_frame", True)),
        }
        # Opciones del corredor (CONTADOR_V2): n_lines / invert / axis_rot
        copts = getattr(w.detector, "corridor_opts", None)
        if callable(copts):
            out.update(copts())
        return jsonify(out)

    @app.route("/api/sources/<sid>/zones", methods=["POST"])
    def set_zones(sid):
        w = _get_worker(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        data = request.get_json(force=True, silent=True) or {}
        raw = data.get("regions")
        if not isinstance(raw, list):
            return jsonify({"error": "Falta 'regions' (lista)"}), 400
        if len(raw) > MAX_ZONES:
            return jsonify({"error": f"Maximo {MAX_ZONES} zonas"}), 400
        regions = []
        for ri, r in enumerate(raw):
            if not isinstance(r, list) or len(r) < MIN_POINTS:
                return jsonify({"error": f"Zona {ri + 1}: minimo "
                                         f"{MIN_POINTS} puntos"}), 400
            regions.append([[int(p[0]), int(p[1])] for p in r])

        zone_expand = data.get("zone_expand")
        if zone_expand is not None:
            try:
                zone_expand = max(0, min(300, int(zone_expand)))
            except (TypeError, ValueError):
                return jsonify({"error": "'zone_expand' debe ser entero"}), 400

        # Opciones del corredor (CONTADOR_V2), si vienen
        opts = {}
        if data.get("n_lines") is not None:
            try:
                opts["n_lines"] = max(1, min(9, int(data["n_lines"])))
            except (TypeError, ValueError):
                return jsonify({"error": "'n_lines' debe ser entero"}), 400
        for k in ("invert", "axis_rot"):
            if data.get(k) is not None:
                opts[k] = bool(data[k])

        key = (getattr(w.detector, "source_key", "")
               or f"{w.detector.NAME}__{source_key(w.source)}")
        path = os.path.join(ZONES_WEB_DIR, f"{key}.json")
        payload = {
            "regions": regions,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source":  str(w.source),
        }
        if zone_expand is not None:
            payload["zone_expand"] = zone_expand
        payload.update(opts)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        setter = getattr(w.detector, "set_zones", None)
        if callable(setter):
            try:
                setter(regions, zone_expand, opts)
            except TypeError:      # detectores sin parametro opts
                setter(regions, zone_expand)
        print(f"[zones] [{sid}] {len(regions)} zona(s), "
              f"expand={zone_expand}, opts={opts} → {path}")
        return jsonify({"ok": True, "path": path, "n": len(regions),
                        "zone_expand": zone_expand, **opts})

    # ------------------------------------------------------------------ #
    #  PARAMETROS del modelo: ajustables en caliente desde el panel
    #  (opcional: el detector implementa get_params/set_params — ver
    #   params_mixin.ParamsMixin). Se persisten en params_web/<fuente>.json
    # ------------------------------------------------------------------ #
    @app.route("/api/sources/<sid>/params", methods=["GET"])
    def get_params(sid):
        w = _get_worker(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        getter = getattr(w.detector, "get_params", None)
        if not callable(getter):
            return jsonify({"error": "Este modelo no expone parametros"}), 400
        return jsonify({"type": w.detector.NAME, "params": getter()})

    @app.route("/api/sources/<sid>/params", methods=["POST"])
    def set_params(sid):
        w = _get_worker(sid)
        if w is None:
            return jsonify({"error": "No existe"}), 404
        setter = getattr(w.detector, "set_params", None)
        if not callable(setter):
            return jsonify({"error": "Este modelo no expone parametros"}), 400
        data = request.get_json(force=True, silent=True) or {}
        res = setter(data)
        applied = res.get("applied", {})
        if applied:
            key = (getattr(w.detector, "source_key", "")
                   or f"{w.detector.NAME}__{source_key(w.source)}")
            ppath = os.path.join(PARAMS_WEB_DIR, f"{key}.json")
            try:
                saved = {}
                if os.path.isfile(ppath):
                    with open(ppath) as f:
                        saved = json.load(f)
                saved.update(applied)
                with open(ppath, "w") as f:
                    json.dump(saved, f, indent=2)
            except Exception as e:
                print(f"[params] No se pudo guardar {ppath}: {e}")
            print(f"[params] [{sid}] {applied}")
        return jsonify(res)

    # ------------------------------------------------------------------ #
    #  GALERIA ReID (personas autorizadas) — quien SI pasa y quien NO
    # ------------------------------------------------------------------ #
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    _rebuild_lock = threading.Lock()

    def _personas_root():
        d = getattr(args, "personas_dir", None)
        if not d:
            return None
        os.makedirs(d, exist_ok=True)
        return os.path.realpath(d)

    def _persona_dir(name, must_exist=False):
        """Path seguro de la carpeta de una persona (evita path traversal)."""
        root = _personas_root()
        name = (name or "").strip()
        if root is None or not name or name.startswith("."):
            return None
        path = os.path.realpath(os.path.join(root, name))
        if path == root or not path.startswith(root + os.sep):
            return None
        if must_exist and not os.path.isdir(path):
            return None
        return path

    @app.route("/api/gallery", methods=["GET"])
    def gallery_info():
        hook = getattr(gallery_cls, "gallery_info", None) if gallery_cls else None
        if not callable(hook):
            return jsonify({"error": "Este detector no tiene galeria"}), 404
        return jsonify(hook(args))

    @app.route("/api/gallery/person", methods=["POST"])
    def gallery_add_person():
        data = request.get_json(force=True, silent=True) or {}
        pdir = _persona_dir(data.get("name"))
        if pdir is None:
            return jsonify({"error": "Nombre invalido"}), 400
        os.makedirs(pdir, exist_ok=True)
        print(f"[gallery] Persona creada: {pdir}")
        return jsonify({"ok": True, "name": os.path.basename(pdir)})

    @app.route("/api/gallery/person/<path:name>", methods=["DELETE"])
    def gallery_del_person(name):
        pdir = _persona_dir(name, must_exist=True)
        if pdir is None:
            return jsonify({"error": "No existe"}), 404
        shutil.rmtree(pdir)
        print(f"[gallery] Persona eliminada: {pdir}")
        return jsonify({"ok": True})

    @app.route("/api/gallery/person/<path:name>/photos", methods=["POST"])
    def gallery_upload(name):
        pdir = _persona_dir(name)
        if pdir is None:
            return jsonify({"error": "Nombre invalido"}), 400
        os.makedirs(pdir, exist_ok=True)
        files = request.files.getlist("files") or request.files.getlist("file")
        saved, skipped = [], []
        for f in files:
            fn = secure_filename(f.filename or "")
            if not fn or os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                skipped.append(f.filename)
                continue
            f.save(os.path.join(pdir, fn))
            saved.append(fn)
        if not saved and skipped:
            return jsonify({"error": f"Formato no soportado: {skipped}"}), 400
        print(f"[gallery] {name}: +{len(saved)} foto(s)")
        return jsonify({"ok": True, "saved": saved, "skipped": skipped})

    @app.route("/api/gallery/person/<path:name>/photos/<path:fname>",
               methods=["GET", "DELETE"])
    def gallery_photo(name, fname):
        pdir = _persona_dir(name, must_exist=True)
        if pdir is None:
            return jsonify({"error": "No existe"}), 404
        fpath = os.path.realpath(os.path.join(pdir, fname))
        if not fpath.startswith(pdir + os.sep) or not os.path.isfile(fpath):
            return jsonify({"error": "No existe"}), 404
        if request.method == "DELETE":
            os.remove(fpath)
            print(f"[gallery] {name}: foto eliminada {fname}")
            return jsonify({"ok": True})
        return send_from_directory(pdir, os.path.basename(fpath))

    @app.route("/api/gallery/rebuild", methods=["POST"])
    def gallery_rebuild():
        hook = getattr(gallery_cls, "gallery_rebuild", None) if gallery_cls else None
        if not callable(hook):
            return jsonify({"error": "Este detector no tiene galeria"}), 404
        if not _rebuild_lock.acquire(blocking=False):
            return jsonify({"error": "Ya hay una reconstruccion en curso"}), 409
        try:
            t0 = time.time()
            res = hook(args)
            res["seconds"] = round(time.time() - t0, 1)
            return jsonify(res)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            _rebuild_lock.release()

    _load_config()
    return app


# ====================================================================== #
#  CLI (mismas flags que Base/web_server.main_web)
# ====================================================================== #
def main_web_zones(detector_cls, argv=None):
    """detector_cls: una clase BaseDetector o un dict {NOMBRE: clase} para
    correr varios modelos en el mismo panel (se elige el modelo por fuente)."""
    detectors = _normalize_detectors(detector_cls)
    title = " + ".join(detectors)
    p = argparse.ArgumentParser(description=f"Panel web + zonas: {title}",
                                conflict_handler="resolve")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--upload-dir", default="uploads",
                   help="Carpeta donde se guardan los videos subidos")
    p.add_argument("--config", default="sources.json",
                   help="Persistencia de fuentes (se recargan al reiniciar)")
    p.add_argument("--resize", type=_parse_resize, default=None,
                   help="Redimensionar frames, formato WxH")
    p.add_argument("--frame-skip", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=80)
    p.add_argument("--record-dir", default="grabaciones",
                   help="Carpeta donde se guardan las grabaciones del panel")
    p.add_argument("--alert-dir", default="",
                   help="Carpeta raiz de alertas (default ALERTAS_<NAME>)")
    p.add_argument("--alert-cooldown", type=float, default=5.0)
    p.add_argument("--no-loop-files", dest="loop_files", action="store_false",
                   help="No repetir los videos en bucle al terminar")
    for cls in detectors.values():
        cls().add_arguments(p)
    args = p.parse_args(argv)

    install_log_capture()   # logs visibles tambien en la plataforma
    app = create_app(detectors, args)
    print(f"[web] Panel {title} (+editor de zonas): "
          f"http://<IP_SERVIDOR>:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


# ====================================================================== #
#  UI — HTML de Base/web_server.py + editor de zonas
# ====================================================================== #
_HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>%%NAME%% — Panel</title>
<style>
  * { box-sizing: border-box; }
  body { margin:0; background:#14161a; color:#e8e8e8;
         font-family: system-ui, sans-serif; }
  header { display:flex; align-items:center; gap:16px; flex-wrap:wrap;
           padding:12px 20px; background:#1d2026; border-bottom:1px solid #2c313a; }
  h1 { font-size:18px; margin:0; color:#7fd0ff; }
  header form { display:flex; gap:8px; align-items:center; }
  input[type=text] { background:#0f1114; border:1px solid #3a4150; color:#eee;
                     padding:8px 10px; border-radius:6px; width:340px; }
  select { background:#0f1114; border:1px solid #3a4150; color:#eee;
           padding:8px 10px; border-radius:6px; font-size:13px; }
  .typsel { padding:2px 4px; border-radius:8px; font-size:10px; font-weight:600;
            background:#312e81; color:#c7d2fe; border:1px solid #4338ca;
            margin-right:8px; cursor:pointer; }
  button { background:#2563eb; border:0; color:#fff; padding:8px 14px;
           border-radius:6px; cursor:pointer; font-size:13px; }
  button:hover { background:#3b82f6; }
  button.rm { background:#7f1d1d; padding:4px 10px; }
  button.rm:hover { background:#b91c1c; }
  #grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(480px,1fr));
          gap:14px; padding:16px 20px; }
  .card { background:#1d2026; border:1px solid #2c313a; border-radius:10px;
          overflow:hidden; }
  .card .head { display:flex; justify-content:space-between; align-items:center;
                padding:8px 12px; font-size:13px; }
  .card .view { overflow:hidden; position:relative; background:#000;
                aspect-ratio:16/9; }
  .card .view img { width:100%; height:100%; object-fit:contain; display:block;
                    transform-origin:center center; user-select:none; }
  .card .view.zoomed img { cursor:grab; }
  .card .view .zhint { position:absolute; right:8px; bottom:6px; font-size:11px;
                       color:#9aa4b2; background:rgba(0,0,0,.5);
                       padding:2px 6px; border-radius:6px; pointer-events:none; }
  .card .tools { display:flex; gap:8px; align-items:center; padding:8px 12px;
                 border-top:1px solid #2c313a; }
  .card .tools input { flex:1; width:auto; background:#0f1114;
                       border:1px solid #3a4150; color:#eee; padding:6px 8px;
                       border-radius:6px; font-size:12px; }
  button.rec { background:#166534; }
  button.rec.on { background:#b91c1c; animation: blink 1.2s infinite; }
  button.zed { background:#7c3aed; }
  button.zed:hover { background:#8b5cf6; }
  @keyframes blink { 50% { opacity:.55; } }
  .card .stat { display:flex; gap:12px; padding:8px 12px; font-size:12px;
                color:#9aa4b2; flex-wrap:wrap; }
  .stat .ri { color:#f87171; font-weight:600; }
  .stat .zn { color:#a78bfa; }
  .badge { padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .b-ONLINE { background:#14532d; color:#86efac; }
  .b-CONECTANDO, .b-RECONECTANDO { background:#713f12; color:#fde047; }
  .b-ERROR, .b-OFFLINE { background:#7f1d1d; color:#fca5a5; }
  .b-FIN { background:#374151; color:#d1d5db; }
  .alerts-hot { color:#f87171; font-weight:700; }
  #empty { padding:60px 20px; text-align:center; color:#6b7280; }

  /* ---------- Costo computacional ---------- */
  #sysbar { display:flex; gap:18px; flex-wrap:wrap; align-items:center;
            padding:6px 20px; background:#181b20; font-size:12px;
            color:#9aa4b2; border-bottom:1px solid #2c313a; }
  #sysbar b { color:#e8e8e8; font-weight:600; }
  #sysbar .gname { color:#7fd0ff; }
  .meter { position:relative; width:130px; height:10px; background:#0f1114;
           border:1px solid #3a4150; border-radius:5px; overflow:hidden;
           display:inline-block; vertical-align:middle; }
  .meter i { position:absolute; inset:0; width:0%; background:#2563eb;
             transition:width .5s; }
  .meter i.warn { background:#d97706; }
  .meter i.crit { background:#b91c1c; }

  /* ---------- Registro (logs) ---------- */
  #logsec { background:#15171b; border-bottom:1px solid #2c313a; }
  #loghead { display:flex; gap:12px; align-items:center; padding:6px 20px;
             font-size:12px; color:#9aa4b2; }
  #loghead .t { color:#7fd0ff; font-weight:600; }
  #lognew { color:#f87171; font-weight:600; }
  #loghead button { padding:3px 10px; font-size:11px; background:#374151; }
  #loghead button:hover { background:#4b5563; }
  #logbody { max-height:190px; overflow-y:auto; padding:4px 20px 10px;
             font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size:11.5px; line-height:1.5; white-space:pre-wrap;
             word-break:break-all; color:#9aa4b2; }
  #logbody.hidden { display:none; }
  #logbody .l-alert { color:#f87171; font-weight:600; }
  #logbody .l-ev    { color:#fbbf24; }
  #logbody .l-ok    { color:#86efac; }
  #logbody .l-info  { color:#7fd0ff; }

  /* ---------- Galeria de autorizados ---------- */
  button.gal { background:#0e7490; }
  button.gal:hover { background:#0891b2; }
  #gmodal { display:none; position:fixed; inset:0; z-index:60;
            background:rgba(0,0,0,.82); }
  #gmodal.open { display:flex; align-items:center; justify-content:center; }
  #gwrap { display:flex; flex-direction:column; gap:10px; width:min(860px,94vw);
           max-height:92vh; background:#1d2026; border:1px solid #3a4150;
           border-radius:10px; padding:14px; }
  #gtop { display:flex; justify-content:space-between; gap:16px;
          font-size:14px; flex-wrap:wrap; }
  #gtop .t { color:#7fd0ff; font-weight:600; }
  #gstatus { color:#9aa4b2; font-size:12px; }
  #glist { overflow-y:auto; display:flex; flex-direction:column; gap:10px; }
  .gperson { background:#14161a; border:1px solid #2c313a; border-radius:8px;
             padding:10px; }
  .gperson .gh { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
                 margin-bottom:8px; }
  .gperson .gh b { font-size:14px; }
  .gbadge { padding:2px 8px; border-radius:10px; font-size:11px; font-weight:600; }
  .g-on  { background:#14532d; color:#86efac; }
  .g-off { background:#713f12; color:#fde047; }
  .gphotos { display:flex; gap:8px; flex-wrap:wrap; }
  .gthumb { position:relative; }
  .gthumb img { height:72px; border-radius:6px; display:block;
                border:1px solid #2c313a; }
  .gthumb button { position:absolute; top:-6px; right:-6px; padding:0 6px;
                   font-size:11px; border-radius:10px; background:#7f1d1d; }
  .gperson .gup { margin-left:auto; display:flex; gap:6px; align-items:center; }
  .gperson .gup input[type=file] { color:#9aa4b2; font-size:11px; max-width:220px; }
  #gnew { display:flex; gap:8px; align-items:center; flex-wrap:wrap;
          border-top:1px solid #2c313a; padding-top:10px; }
  #gnew input[type=text] { width:220px; }
  #gnew input[type=file] { color:#9aa4b2; font-size:12px; }
  #gbtns { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #gpend { color:#fbbf24; font-size:12px; flex:1; }
  #gbtns .save { background:#166534; }
  #gbtns .save:hover { background:#15803d; }

  /* ---------- Editor de zonas ---------- */
  #zmodal { display:none; position:fixed; inset:0; z-index:50;
            background:rgba(0,0,0,.82); }
  #zmodal.open { display:flex; align-items:center; justify-content:center; }
  #zwrap { display:flex; flex-direction:column; gap:8px; max-width:96vw;
           max-height:96vh; background:#1d2026; border:1px solid #3a4150;
           border-radius:10px; padding:12px; }
  #ztop { display:flex; justify-content:space-between; gap:16px;
          font-size:13px; color:#9aa4b2; flex-wrap:wrap; }
  #zinfo { color:#a78bfa; font-weight:600; }
  #zview { overflow:auto; max-width:92vw; max-height:78vh; background:#000;
           border-radius:6px; }
  #zcanvas { display:block; max-width:90vw; max-height:76vh; cursor:crosshair; }
  #zbtns { display:flex; gap:8px; flex-wrap:wrap; }
  #zbtns .save { background:#166534; }
  #zbtns .save:hover { background:#15803d; }
  #zexp, #zcorr { display:flex; gap:10px; align-items:center; flex-wrap:wrap;
          font-size:13px; color:#9aa4b2; }
  #zexp input[type=range], #zcorr input[type=range] { width:220px; accent-color:#7c3aed; }
  #zexp b, #zcorr b { color:#a78bfa; min-width:32px; }
  #zexp .hint, #zcorr .hint { font-size:11px; color:#6b7280; }
  #zcorr label { display:flex; gap:5px; align-items:center; cursor:pointer;
                 color:#e8e8e8; }
  #zcorr input[type=checkbox] { accent-color:#7c3aed; }

  /* ---------- Modal de parametros del modelo ---------- */
  #pmodal { display:none; position:fixed; inset:0; z-index:55;
            background:rgba(0,0,0,.82); }
  #pmodal.open { display:flex; align-items:center; justify-content:center; }
  #pwrap { display:flex; flex-direction:column; gap:10px; width:min(560px,94vw);
           max-height:92vh; background:#1d2026; border:1px solid #3a4150;
           border-radius:10px; padding:14px; }
  #ptop { display:flex; justify-content:space-between; gap:12px;
          font-size:13px; color:#9aa4b2; }
  #ptitle { color:#7fd0ff; font-weight:600; }
  #pform { display:flex; flex-direction:column; gap:10px; overflow:auto; }
  .prow { display:grid; grid-template-columns: 180px 1fr 78px; gap:10px;
          align-items:center; font-size:13px; }
  .prow .plabel { color:#e8e8e8; }
  .prow .phelp { grid-column: 1 / -1; font-size:11px; color:#6b7280;
                 margin-top:-6px; }
  .prow input[type=range] { width:100%; accent-color:#7c3aed; }
  .prow input[type=number] { background:#0f1114; border:1px solid #3a4150;
          color:#eee; padding:5px 6px; border-radius:6px; width:100%; }
  .prow select { width:100%; }
  .prow input[type=checkbox] { accent-color:#7c3aed; transform:scale(1.2); }
  #pbtns { display:flex; gap:8px; justify-content:flex-end; }
  #pbtns .save { background:#166534; }
  #pbtns .save:hover { background:#15803d; }
  #pmsg { font-size:12px; color:#facc15; min-height:14px; }
</style>
</head>
<body>
<header>
  <h1>%%NAME%%</h1>
  <form id="addForm">
    <input type="text" id="srcInput" placeholder="rtsp://usuario:pass@ip/stream  |  /path/al/video.mp4  |  0 (webcam)">
    <select id="typeSel" title="Modelo que corre esta fuente">%%TYPE_OPTIONS%%</select>
    <button type="submit">+ Agregar fuente</button>
  </form>
  <form id="upForm">
    <input type="file" id="fileInput" accept="video/*" style="color:#9aa4b2;font-size:12px;">
    <button type="submit">Subir video</button>
  </form>
  <button type="button" id="galBtn" class="gal">&#128100; Autorizados</button>
</header>
<div id="sysbar">
  <span class="gname" id="sysGpuName">GPU: —</span>
  <span>VRAM panel: <b id="sysVramProc">—</b></span>
  <span>VRAM GPU: <b id="sysVram">—</b>
    <span class="meter"><i id="sysVramBar"></i></span></span>
  <span>GPU util: <b id="sysUtil">—</b>
    <span class="meter"><i id="sysUtilBar"></i></span></span>
  <span>RAM panel: <b id="sysRam">—</b></span>
  <span>CPU load: <b id="sysCpu">—</b></span>
</div>
<div id="logsec">
  <div id="loghead">
    <span class="t">&#128220; Registro (logs)</span>
    <span id="lognew"></span>
    <span style="flex:1"></span>
    <button type="button" id="logclear">Limpiar</button>
    <button type="button" id="logtoggle">&#9650; Ocultar</button>
  </div>
  <div id="logbody"></div>
</div>
<div id="grid"></div>
<div id="empty">Sin fuentes. Agrega una URL RTSP, un path de video o sube un archivo.</div>

<!-- ---------- Galeria de autorizados ---------- -->
<div id="gmodal">
  <div id="gwrap">
    <div id="gtop">
      <span class="t">Personas autorizadas (galeria ReID) — quien SI puede entrar a las zonas</span>
      <span id="gstatus"></span>
    </div>
    <div id="glist"></div>
    <div id="gnew">
      <input type="text" id="gname" placeholder="nombre de la persona nueva">
      <input type="file" id="gfiles" accept="image/*" multiple>
      <button type="button" id="gadd">+ Agregar persona</button>
    </div>
    <div id="gbtns">
      <span id="gpend"></span>
      <button type="button" id="grebuild" class="save">&#10227; Reconstruir galeria</button>
      <button type="button" id="gclose" class="rm">Cerrar</button>
    </div>
  </div>
</div>

<!-- ---------- Editor de zonas ---------- -->
<div id="zmodal">
  <div id="zwrap">
    <div id="ztop">
      <span>click: agregar punto &nbsp;·&nbsp; 2 puntos = LINEA (contador) &nbsp;·&nbsp; 3+ = poligono (area) &nbsp;·&nbsp; click derecho: borrar figura</span>
      <span id="zinfo"></span>
    </div>
    <div id="zview"><canvas id="zcanvas"></canvas></div>
    <div id="zexp">
      <span>Zona de deteccion extra (segura):</span>
      <input type="range" id="zexpand" min="0" max="200" step="5" value="40">
      <b id="zexpval">40 px</b>
      <span class="hint">— linea punteada: hasta donde se detecta (el poligono visual no cambia)</span>
    </div>
    <div id="zcorr" style="display:none">
      <span>Lineas internas:</span>
      <input type="range" id="znlines" min="1" max="9" step="1" value="5">
      <b id="znlinesval">5</b>
      <label><input type="checkbox" id="zaxisrot"> &#8635; Rotar lineas 90&deg;</label>
      <label><input type="checkbox" id="zinvert"> &#8646; Voltear AFUERA/ADENTRO</label>
      <span class="hint">— las lineas deben ATRAVESAR el camino de la gente</span>
    </div>
    <div id="zbtns">
      <button id="zclosezone">Cerrar zona</button>
      <button id="zundo">Deshacer punto</button>
      <button id="zclear">Limpiar todo</button>
      <button id="zsave" class="save">&#128190; Guardar zonas</button>
      <button id="zcancel" class="rm">Cancelar</button>
    </div>
  </div>
</div>

<!-- ---------- Parametros del modelo (por fuente) ---------- -->
<div id="pmodal">
  <div id="pwrap">
    <div id="ptop">
      <span id="ptitle">Parametros</span>
      <span>se aplican EN CALIENTE y quedan guardados para esta fuente</span>
    </div>
    <div id="pform"></div>
    <div id="pmsg"></div>
    <div id="pbtns">
      <button id="psave" class="save">&#128190; Aplicar</button>
      <button id="pcancel" class="rm">Cerrar</button>
    </div>
  </div>
</div>

<script>
const grid = document.getElementById('grid');
const empty = document.getElementById('empty');
const cards = {};   // id -> elemento

// Zoom digital: rueda = acercar/alejar, arrastrar = mover, doble clic = reset
function enableZoom(view, img) {
  let scale = 1, tx = 0, ty = 0, dragging = false, sx = 0, sy = 0;
  const apply = () => {
    img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    view.classList.toggle('zoomed', scale > 1);
  };
  view.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = view.getBoundingClientRect();
    const mx = e.clientX - r.left - r.width / 2;
    const my = e.clientY - r.top - r.height / 2;
    const old = scale;
    scale = Math.min(8, Math.max(1, scale * (e.deltaY < 0 ? 1.25 : 0.8)));
    tx = mx + (tx - mx) * (scale / old);
    ty = my + (ty - my) * (scale / old);
    if (scale === 1) { tx = 0; ty = 0; }
    apply();
  }, {passive: false});
  view.addEventListener('mousedown', (e) => {
    if (scale > 1) { dragging = true; sx = e.clientX - tx; sy = e.clientY - ty; e.preventDefault(); }
  });
  window.addEventListener('mousemove', (e) => {
    if (dragging) { tx = e.clientX - sx; ty = e.clientY - sy; apply(); }
  });
  window.addEventListener('mouseup', () => dragging = false);
  view.addEventListener('dblclick', () => { scale = 1; tx = 0; ty = 0; apply(); });
}

function makeCard(s) {
  const d = document.createElement('div');
  d.className = 'card';
  d.innerHTML = `
    <div class="head">
      <span><select class="typsel" title="Modelo de esta fuente"></select><span class="name"></span></span>
      <span><span class="badge"></span>
      <button class="rm" title="Quitar fuente">✕</button></span>
    </div>
    <div class="view">
      <img src="/api/sources/${s.id}/mjpeg" alt="stream" draggable="false">
      <span class="zhint">rueda: zoom · doble clic: reset</span>
    </div>
    <div class="tools">
      <input class="recname" type="text" placeholder="nombre del video (ej. prueba_1)">
      <button class="rec">● Grabar</button>
      <button class="zed" title="Editar zonas restringidas">✎ Zonas</button>
      <button class="prm" title="Parametros del modelo (en caliente)" style="display:none">⚙ Ajustes</button>
    </div>
    <div class="tools ctr" style="display:none">
      <span style="color:#9aa4b2;font-size:12px;">Hay</span>
      <input class="ctrn" type="number" min="0" step="1" placeholder="N" style="flex:0 0 70px;width:70px;min-width:70px;">
      <span style="color:#9aa4b2;font-size:12px;">persona(s) dentro ahora</span>
      <button class="ctrbtn" title="Fija el punto de partida; entradas/salidas arrancan de cero">Fijar y contar</button>
    </div>
    <div class="stat">
      <span class="fps"></span><span class="frame"></span><span class="al"></span><span class="ri"></span>
    </div>`;
  d.querySelector('.rm').onclick = async () => {
    if (!confirm('¿Quitar esta fuente?')) return;
    await fetch('/api/sources/' + s.id, {method:'DELETE'});
    refresh();
  };
  const recBtn = d.querySelector('.rec');
  recBtn.onclick = async () => {
    if (recBtn.dataset.on === '1') {
      const r = await (await fetch(`/api/sources/${s.id}/record`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action:'stop'})})).json();
      if (r.path) alert('Video guardado en:\\n' + r.path);
    } else {
      const name = d.querySelector('.recname').value.trim();
      await fetch(`/api/sources/${s.id}/record`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({action:'start', name})});
    }
    refresh();
  };
  d.querySelector('.zed').onclick = () => openZoneEditor(s.id);
  d.querySelector('.prm').onclick = () => openParamsEditor(s.id);
  d.querySelector('.ctrbtn').onclick = async () => {
    const v = d.querySelector('.ctrn').value.trim();
    const n = parseInt(v);
    if (v === '' || isNaN(n) || n < 0) { alert('Escribe cuantas personas hay dentro (0 o mas)'); return; }
    if (!confirm(`¿Fijar ${n} persona(s) dentro?\\nEntradas/salidas arrancan de cero desde este momento.`)) return;
    const r = await fetch(`/api/sources/${s.id}/counter`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({inside: n})});
    const j = await r.json();
    if (!r.ok) alert('Error: ' + j.error);
    d.querySelector('.ctrn').value = '';
  };
  // Selector de modelo por tarjeta (cambia el detector de ESTA fuente)
  const tsel = d.querySelector('.typsel');
  for (const o of document.getElementById('typeSel').options) {
    tsel.appendChild(new Option(o.value, o.value));
  }
  tsel.value = s.type || '';
  tsel.onchange = async () => {
    const nuevo = tsel.value;
    if (!confirm(`¿Cambiar "${s.name}" al modelo ${nuevo}?\\nLa fuente se reinicia con el modelo nuevo (sus zonas de cada modelo se conservan).`)) {
      tsel.value = s.type;
      return;
    }
    const r = await fetch(`/api/sources/${s.id}/type`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({type: nuevo})});
    if (!r.ok) alert('Error: ' + (await r.json()).error);
    refresh();
  };
  enableZoom(d.querySelector('.view'), d.querySelector('.view img'));
  return d;
}

function updateCard(el, s) {
  const tsel = el.querySelector('.typsel');
  if (document.activeElement !== tsel) tsel.value = s.type || '';
  el.querySelector('.ctr').style.display = s.counter ? 'flex' : 'none';
  el.querySelector('.prm').style.display = s.params ? '' : 'none';
  el.querySelector('.name').textContent = s.name;
  const b = el.querySelector('.badge');
  b.textContent = s.status;
  b.className = 'badge b-' + s.status;
  el.querySelector('.fps').textContent = 'FPS: ' + s.fps.toFixed(1);
  el.querySelector('.frame').textContent = 'frame: ' + s.frame;
  const al = el.querySelector('.al');
  al.textContent = 'alertas: ' + s.alerts;
  al.className = 'al' + (s.alerts > 0 ? ' alerts-hot' : '');
  const rb = el.querySelector('.rec');
  rb.dataset.on = s.recording ? '1' : '0';
  rb.textContent = s.recording ? '■ Detener' : '● Grabar';
  rb.classList.toggle('on', !!s.recording);
  el.querySelector('.ri').textContent = s.rec_file ? ('REC: ' + s.rec_file) : '';
}

function fmtGB(mb) { return mb == null ? '—' : (mb / 1024).toFixed(1) + ' GB'; }

function setMeter(bar, pct) {
  if (pct == null) { bar.style.width = '0%'; return; }
  bar.style.width = Math.min(100, pct) + '%';
  bar.className = pct >= 90 ? 'crit' : (pct >= 70 ? 'warn' : '');
}

async function refreshSystem() {
  try {
    const s = await (await fetch('/api/system')).json();
    document.getElementById('sysGpuName').textContent = 'GPU: ' + (s.gpu_name || 'N/D');
    document.getElementById('sysVramProc').textContent =
      s.vram_proc_mb != null ? fmtGB(s.vram_proc_mb) : '—';
    const vt = s.vram_total_mb, vu = s.vram_used_mb;
    document.getElementById('sysVram').textContent =
      (vu != null && vt != null) ? `${fmtGB(vu)} / ${fmtGB(vt)}` : '—';
    setMeter(document.getElementById('sysVramBar'),
             (vu != null && vt) ? 100 * vu / vt : null);
    document.getElementById('sysUtil').textContent =
      s.gpu_util != null ? s.gpu_util + '%' : '—';
    setMeter(document.getElementById('sysUtilBar'), s.gpu_util);
    document.getElementById('sysRam').textContent = fmtGB(s.ram_proc_mb);
    document.getElementById('sysCpu').textContent =
      s.cpu_load != null ? `${s.cpu_load} / ${s.cpu_cores} cores` : '—';
  } catch (e) {}
}

// ------------------- Registro (logs) en la plataforma ------------------- //
const logbody = document.getElementById('logbody');
let logLast = 0, logHidden = false;

function logClass(line) {
  if (/NO AUTORIZADO|ALERTA|ERROR|Error|Traceback/.test(line)) return 'l-alert';
  if (/ENTER|EXIT/.test(line)) return 'l-ev';
  if (/AUTORIZADO/.test(line)) return 'l-ok';
  if (/\\[GALLERY\\]|\\[zones\\]|\\[REID\\]|\\[web\\]/.test(line)) return 'l-info';
  return '';
}

async function refreshLogs() {
  try {
    const d = await (await fetch('/api/logs?since=' + logLast)).json();
    logLast = d.last;
    if (!d.lines.length) return;
    const atBottom =
      logbody.scrollTop + logbody.clientHeight >= logbody.scrollHeight - 40;
    for (const e of d.lines) {
      const div = document.createElement('div');
      div.textContent = `[${e.t}] ${e.line}`;
      const c = logClass(e.line);
      if (c) div.className = c;
      logbody.appendChild(div);
    }
    while (logbody.children.length > 600) logbody.removeChild(logbody.firstChild);
    if (logHidden) {
      const n = parseInt(document.getElementById('lognew').dataset.n || 0) + d.lines.length;
      document.getElementById('lognew').dataset.n = n;
      document.getElementById('lognew').textContent = `${n} nueva(s)`;
    } else if (atBottom) {
      logbody.scrollTop = logbody.scrollHeight;
    }
  } catch (e) {}
}

document.getElementById('logclear').onclick = () => { logbody.innerHTML = ''; };
document.getElementById('logtoggle').onclick = () => {
  logHidden = !logHidden;
  logbody.classList.toggle('hidden', logHidden);
  document.getElementById('logtoggle').innerHTML =
    logHidden ? '&#9660; Mostrar' : '&#9650; Ocultar';
  if (!logHidden) {
    document.getElementById('lognew').textContent = '';
    document.getElementById('lognew').dataset.n = 0;
    logbody.scrollTop = logbody.scrollHeight;
  }
};

async function refresh() {
  refreshSystem();
  refreshLogs();
  const list = await (await fetch('/api/sources')).json();
  const alive = new Set(list.map(s => s.id));
  for (const id of Object.keys(cards)) {
    if (!alive.has(id)) { cards[id].remove(); delete cards[id]; }
  }
  for (const s of list) {
    if (!cards[s.id]) { cards[s.id] = makeCard(s); grid.appendChild(cards[s.id]); }
    updateCard(cards[s.id], s);
  }
  empty.style.display = list.length ? 'none' : 'block';
}

document.getElementById('addForm').onsubmit = async (e) => {
  e.preventDefault();
  const v = document.getElementById('srcInput').value.trim();
  if (!v) return;
  const r = await fetch('/api/sources', {method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({source: v,
                          type: document.getElementById('typeSel').value})});
  if (!r.ok) alert('Error: ' + (await r.json()).error);
  document.getElementById('srcInput').value = '';
  refresh();
};

document.getElementById('upForm').onsubmit = async (e) => {
  e.preventDefault();
  const f = document.getElementById('fileInput').files[0];
  if (!f) return alert('Selecciona un video primero');
  const fd = new FormData();
  fd.append('file', f);
  fd.append('type', document.getElementById('typeSel').value);
  const btn = e.target.querySelector('button');
  btn.disabled = true; btn.textContent = 'Subiendo...';
  const r = await fetch('/api/upload', {method:'POST', body: fd});
  btn.disabled = false; btn.textContent = 'Subir video';
  if (!r.ok) alert('Error subiendo: ' + (await r.json()).error);
  document.getElementById('fileInput').value = '';
  refresh();
};

// ==================================================================== //
//  Galeria de autorizados (PersonasPermitidas/) — quien SI pasa
// ==================================================================== //
const gmodal = document.getElementById('gmodal');
const glist  = document.getElementById('glist');
let gdirty = false;

function gmark() {
  gdirty = true;
  document.getElementById('gpend').textContent =
    'Cambios pendientes — presiona "Reconstruir galeria" para aplicarlos';
}

async function loadGallery() {
  const r = await fetch('/api/gallery');
  if (!r.ok) { alert('Este detector no tiene galeria ReID'); return; }
  const d = await r.json();
  const loaded = new Set(d.loaded_names || []);
  const faces = d.faces || {};
  const facesTxt = faces.available
    ? ` · rostros: ${faces.n_faces} cara(s)`
    : ' · rostros: NO cargados (faltan insightface/onnxruntime o aun no arranca una fuente)';
  document.getElementById('gstatus').textContent =
    `${(d.loaded_names || []).length} persona(s) activas · ${d.n_feats} features cuerpo${facesTxt}`;
  glist.innerHTML = '';
  if (!d.persons.length) {
    glist.innerHTML = '<div style="color:#6b7280;padding:14px;">Sin personas. ' +
      'Agrega una abajo con sus fotos (cuerpo completo, varias poses).</div>';
  }
  for (const p of d.persons) {
    const el = document.createElement('div');
    el.className = 'gperson';
    const enc = encodeURIComponent(p.name);
    const isOn = loaded.has(p.name);
    const nFaces = (faces.per_person || {})[p.name] || 0;
    const faceBadge = !faces.available ? '' :
      (nFaces > 0
        ? `<span class="gbadge g-on">&#128100; ROSTRO: ${nFaces} cara(s)</span>`
        : `<span class="gbadge g-off">&#9888; SIN ROSTRO — sube una foto donde se le vea la cara</span>`);
    el.innerHTML = `
      <div class="gh">
        <b></b>
        <span class="gbadge ${isOn ? 'g-on' : 'g-off'}">${isOn ? 'ACTIVA' : 'PENDIENTE DE RECONSTRUIR'}</span>
        ${faceBadge}
        <span style="color:#9aa4b2;font-size:12px;">${p.photos.length} foto(s)</span>
        <span class="gup">
          <input type="file" accept="image/*" multiple>
          <button type="button" class="gupbtn">Subir fotos</button>
          <button type="button" class="rm gdel">Borrar persona</button>
        </span>
      </div>
      <div class="gphotos"></div>`;
    el.querySelector('b').textContent = p.name;
    const ph = el.querySelector('.gphotos');
    for (const f of p.photos) {
      const t = document.createElement('div');
      t.className = 'gthumb';
      t.innerHTML = `<img loading="lazy"><button type="button" title="Borrar foto">✕</button>`;
      t.querySelector('img').src = `/api/gallery/person/${enc}/photos/${encodeURIComponent(f)}`;
      t.querySelector('button').onclick = async () => {
        if (!confirm(`¿Borrar esta foto de ${p.name}?`)) return;
        await fetch(`/api/gallery/person/${enc}/photos/${encodeURIComponent(f)}`,
                    {method: 'DELETE'});
        gmark(); loadGallery();
      };
      ph.appendChild(t);
    }
    el.querySelector('.gupbtn').onclick = async () => {
      const inp = el.querySelector('input[type=file]');
      if (!inp.files.length) { alert('Selecciona fotos primero'); return; }
      const fd = new FormData();
      for (const f of inp.files) fd.append('files', f);
      const r2 = await fetch(`/api/gallery/person/${enc}/photos`,
                             {method: 'POST', body: fd});
      if (!r2.ok) alert('Error: ' + (await r2.json()).error);
      else gmark();
      loadGallery();
    };
    el.querySelector('.gdel').onclick = async () => {
      if (!confirm(`¿Borrar a "${p.name}" y TODAS sus fotos?\\nDejara de estar autorizado tras reconstruir.`)) return;
      await fetch(`/api/gallery/person/${enc}`, {method: 'DELETE'});
      gmark(); loadGallery();
    };
    glist.appendChild(el);
  }
}

document.getElementById('galBtn').onclick = () => {
  gmodal.classList.add('open');
  loadGallery();
};
document.getElementById('gclose').onclick = () => gmodal.classList.remove('open');

document.getElementById('gadd').onclick = async () => {
  const name = document.getElementById('gname').value.trim();
  if (!name) { alert('Escribe el nombre de la persona'); return; }
  const r = await fetch('/api/gallery/person', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({name})});
  if (!r.ok) { alert('Error: ' + (await r.json()).error); return; }
  const files = document.getElementById('gfiles').files;
  if (files.length) {
    const fd = new FormData();
    for (const f of files) fd.append('files', f);
    await fetch(`/api/gallery/person/${encodeURIComponent(name)}/photos`,
                {method: 'POST', body: fd});
  }
  document.getElementById('gname').value = '';
  document.getElementById('gfiles').value = '';
  gmark(); loadGallery();
};

document.getElementById('grebuild').onclick = async () => {
  const btn = document.getElementById('grebuild');
  btn.disabled = true; btn.textContent = 'Reconstruyendo...';
  try {
    const r = await fetch('/api/gallery/rebuild', {method: 'POST'});
    const j = await r.json();
    if (r.ok) {
      gdirty = false;
      document.getElementById('gpend').textContent = '';
      alert(`Galeria aplicada EN VIVO:\\n${j.n_persons} persona(s) · cuerpo: ${j.n_feats} features · rostros: ${j.n_faces || 0} cara(s) (${j.seconds}s)\\n${(j.names || []).join(', ')}`);
    } else {
      alert('Error: ' + j.error);
    }
  } finally {
    btn.disabled = false; btn.textContent = '⟳ Reconstruir galeria';
    loadGallery();
  }
};

// ==================================================================== //
//  Editor de zonas (version web del AreaRestriction_ZoneCreator)
// ==================================================================== //
const MAX_ZONES = 6, MIN_POINTS = 2;  // 2 puntos = linea (contador)
// ZONE_COLORS del detector (BGR → CSS)
const ZCOLORS = ['#dc0000','#00c800','#0000c8','#dcdc00','#dc00dc','#00dcdc'];

const zmodal  = document.getElementById('zmodal');
const zcanvas = document.getElementById('zcanvas');
const zctx    = zcanvas.getContext('2d');
const zinfo   = document.getElementById('zinfo');
let zsid = null, zimg = new Image(), zregions = [], zcurrent = [];
let zfull = true;   // full_frame (outdoor): expand se aplica en espacio proc 640px
let zcorridor = false;   // la fuente es un CONTADOR_V2 (corredor)
const PROC_W = 640;
const zexpand  = document.getElementById('zexpand');
const zexpval  = document.getElementById('zexpval');
const znlines  = document.getElementById('znlines');
const zaxisrot = document.getElementById('zaxisrot');
const zinvert  = document.getElementById('zinvert');

// Misma matematica que _expand_polygon del detector (desde el centroide)
function expandPoly(r, px) {
  if (px <= 0) return r;
  const cx = r.reduce((a, p) => a + p[0], 0) / r.length;
  const cy = r.reduce((a, p) => a + p[1], 0) / r.length;
  return r.map(([x, y]) => {
    const dx = x - cx, dy = y - cy;
    const d = Math.max(Math.hypot(dx, dy), 1e-6);
    return [x + dx / d * px, y + dy / d * px];
  });
}

// Vista previa del corredor (CONTADOR_V2): escalera + AFUERA/ADENTRO
function drawCorridorPreview(r, lw, fs) {
  if (r.length !== 4) return;
  const n = parseInt(znlines.value);
  document.getElementById('znlinesval').textContent = n;
  const rot = zaxisrot.checked;
  // rieles: entre el lado de entrada y el opuesto (segun rotacion)
  const railA = rot ? [r[1], r[0]] : [r[0], r[3]];
  const railB = rot ? [r[2], r[3]] : [r[1], r[2]];
  zctx.setLineDash([10, 8]);
  zctx.strokeStyle = '#ff8c30'; zctx.lineWidth = Math.max(2, lw - 1);
  for (let i = 1; i <= n; i++) {
    const t = i / (n + 1);
    const ax = railA[0][0] + (railA[1][0] - railA[0][0]) * t;
    const ay = railA[0][1] + (railA[1][1] - railA[0][1]) * t;
    const bx = railB[0][0] + (railB[1][0] - railB[0][0]) * t;
    const by = railB[0][1] + (railB[1][1] - railB[0][1]) * t;
    zctx.beginPath(); zctx.moveTo(ax, ay); zctx.lineTo(bx, by); zctx.stroke();
  }
  zctx.setLineDash([]);
  // etiquetas AFUERA / ADENTRO
  const e1 = rot ? [r[1], r[2]] : [r[0], r[1]];
  const e2 = rot ? [r[3], r[0]] : [r[2], r[3]];
  let lbl = ['AFUERA', 'ADENTRO'];
  if (zinvert.checked) lbl = ['ADENTRO', 'AFUERA'];
  const colFor = t => t === 'ADENTRO' ? '#5ce07a' : '#ffc23d';
  zctx.font = `bold ${fs}px sans-serif`;
  zctx.fillStyle = colFor(lbl[0]);
  zctx.fillText(lbl[0], (e1[0][0]+e1[1][0])/2 - fs*2, (e1[0][1]+e1[1][1])/2 - 8);
  zctx.fillStyle = colFor(lbl[1]);
  zctx.fillText(lbl[1], (e2[0][0]+e2[1][0])/2 - fs*2, (e2[0][1]+e2[1][1])/2 - 8);
}

function zdraw() {
  if (!zimg.naturalWidth) return;
  zctx.drawImage(zimg, 0, 0);
  const lw = Math.max(2, zcanvas.width / 500);
  const pr = Math.max(5, zcanvas.width / 220);
  const fs = Math.max(16, zcanvas.width / 55);

  // expand esta en px de espacio proc (outdoor) → escalar a px nativos
  const expPx = parseInt(zexpand.value) * (zfull ? zcanvas.width / PROC_W : 1);
  zexpval.textContent = zexpand.value + ' px';

  zregions.forEach((r, i) => {
    const c = ZCOLORS[i % ZCOLORS.length];
    zctx.beginPath();
    r.forEach((p, j) => j ? zctx.lineTo(p[0], p[1]) : zctx.moveTo(p[0], p[1]));
    zctx.closePath();
    zctx.fillStyle = c + '40';
    zctx.fill();
    zctx.strokeStyle = c; zctx.lineWidth = lw; zctx.stroke();
    // vista previa del area de deteccion expandida (punteada)
    if (expPx > 0) {
      const ex = expandPoly(r, expPx);
      zctx.beginPath();
      ex.forEach((p, j) => j ? zctx.lineTo(p[0], p[1]) : zctx.moveTo(p[0], p[1]));
      zctx.closePath();
      zctx.setLineDash([10, 8]);
      zctx.strokeStyle = c; zctx.lineWidth = Math.max(1, lw - 1); zctx.stroke();
      zctx.setLineDash([]);
    }
    r.forEach(p => { zctx.beginPath(); zctx.arc(p[0], p[1], pr, 0, 7);
                     zctx.fillStyle = c; zctx.fill(); });
    const cx = r.reduce((a, p) => a + p[0], 0) / r.length;
    const cy = r.reduce((a, p) => a + p[1], 0) / r.length;
    zctx.fillStyle = '#fff'; zctx.font = `bold ${fs}px sans-serif`;
    zctx.fillText('Zona ' + (i + 1), cx - fs, cy);
    if (zcorridor && i === 0) drawCorridorPreview(r, lw, fs);
  });

  // puntos en curso (blanco)
  if (zcurrent.length > 1) {
    zctx.beginPath();
    zcurrent.forEach((p, j) => j ? zctx.lineTo(p[0], p[1]) : zctx.moveTo(p[0], p[1]));
    zctx.strokeStyle = '#fff'; zctx.lineWidth = lw; zctx.stroke();
  }
  zcurrent.forEach(p => { zctx.beginPath(); zctx.arc(p[0], p[1], pr, 0, 7);
                          zctx.fillStyle = '#fff'; zctx.fill(); });

  zinfo.textContent = `Zonas: ${zregions.length}/${MAX_ZONES} · puntos en curso: ${zcurrent.length} (min ${MIN_POINTS})`;
}

function zpos(e) {
  const r = zcanvas.getBoundingClientRect();
  return [Math.round((e.clientX - r.left) * zcanvas.width  / r.width),
          Math.round((e.clientY - r.top)  * zcanvas.height / r.height)];
}

function pointInPoly(x, y, poly) {
  let ins = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi))
      ins = !ins;
  }
  return ins;
}

zcanvas.onclick = (e) => {
  if (zregions.length >= MAX_ZONES) { alert(`Maximo ${MAX_ZONES} zonas`); return; }
  zcurrent.push(zpos(e));
  zdraw();
};

zcanvas.oncontextmenu = (e) => {
  e.preventDefault();
  const [x, y] = zpos(e);
  for (let i = zregions.length - 1; i >= 0; i--) {
    if (pointInPoly(x, y, zregions[i])) {
      if (confirm(`¿Borrar Zona ${i + 1}?`)) { zregions.splice(i, 1); zdraw(); }
      return;
    }
  }
};

document.getElementById('zclosezone').onclick = () => {
  if (zcurrent.length < MIN_POINTS) { alert(`Minimo ${MIN_POINTS} puntos`); return; }
  zregions.push(zcurrent.slice());
  zcurrent = [];
  zdraw();
};
document.getElementById('zundo').onclick  = () => { zcurrent.pop(); zdraw(); };
zexpand.oninput = zdraw;
znlines.oninput = zdraw;
zaxisrot.onchange = zdraw;
zinvert.onchange = zdraw;
document.getElementById('zclear').onclick = () => { zregions = []; zcurrent = []; zdraw(); };
document.getElementById('zcancel').onclick = () => zmodal.classList.remove('open');

document.getElementById('zsave').onclick = async () => {
  if (zcurrent.length >= MIN_POINTS) { zregions.push(zcurrent.slice()); zcurrent = []; }
  const body = {regions: zregions, zone_expand: parseInt(zexpand.value)};
  if (zcorridor) {
    body.n_lines = parseInt(znlines.value);
    body.invert = zinvert.checked;
    body.axis_rot = zaxisrot.checked;
  }
  const r = await fetch(`/api/sources/${zsid}/zones`, {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)});
  const j = await r.json();
  if (r.ok) {
    zmodal.classList.remove('open');
  } else {
    alert('Error: ' + j.error);
  }
};

async function openZoneEditor(sid) {
  zsid = sid; zregions = []; zcurrent = [];
  try {
    const d = await (await fetch(`/api/sources/${sid}/zones`)).json();
    zregions = d.regions || [];
    zfull = d.full_frame !== false;
    if (d.zone_expand != null) zexpand.value = d.zone_expand;
    zcorridor = !!d.corridor;
    document.getElementById('zcorr').style.display = zcorridor ? 'flex' : 'none';
    if (zcorridor) {
      if (d.n_lines != null) znlines.value = d.n_lines;
      zaxisrot.checked = !!d.axis_rot;
      zinvert.checked = !!d.invert;
    }
  } catch (e) {}
  zimg = new Image();
  zimg.onload = () => {
    zcanvas.width  = zimg.naturalWidth;
    zcanvas.height = zimg.naturalHeight;
    zdraw();
  };
  zimg.onerror = () => {
    zmodal.classList.remove('open');
    alert('Sin frame disponible aun — espera a que la fuente este ONLINE');
  };
  zimg.src = `/api/sources/${sid}/frame?t=` + Date.now();
  zmodal.classList.add('open');
}

// ------------------------- Parametros del modelo -------------------------
const pmodal = document.getElementById('pmodal');
const pform  = document.getElementById('pform');
const pmsg   = document.getElementById('pmsg');
let   psid   = null;

function paramRow(p) {
  const row = document.createElement('div');
  row.className = 'prow';
  row.dataset.key = p.key;
  row.dataset.type = p.type;
  const label = `<span class="plabel" title="${p.key}">${p.label || p.key}</span>`;
  if (p.type === 'bool') {
    row.innerHTML = label +
      `<span></span><input type="checkbox" class="pval" ${p.value ? 'checked' : ''}>`;
  } else if (p.type === 'choice') {
    const opts = (p.choices || []).map(c =>
      `<option value="${c}" ${c === p.value ? 'selected' : ''}>${c}</option>`).join('');
    row.innerHTML = label + `<select class="pval">${opts}</select><span></span>`;
  } else {
    const step = p.step != null ? p.step : (p.type === 'int' ? 1 : 0.01);
    const rng = (p.min != null && p.max != null)
      ? `<input type="range" class="prng" min="${p.min}" max="${p.max}" step="${step}" value="${p.value}">`
      : `<span></span>`;
    row.innerHTML = label + rng +
      `<input type="number" class="pval" step="${step}"` +
      (p.min != null ? ` min="${p.min}"` : '') +
      (p.max != null ? ` max="${p.max}"` : '') +
      ` value="${p.value}">`;
    const rngEl = row.querySelector('.prng');
    const numEl = row.querySelector('.pval');
    if (rngEl) {
      rngEl.oninput = () => { numEl.value = rngEl.value; };
      numEl.oninput = () => { rngEl.value = numEl.value; };
    }
  }
  if (p.help) row.innerHTML += `<span class="phelp">${p.help}</span>`;
  // valor inicial → al guardar solo se mandan los campos EDITADOS
  row.dataset.init = p.type === 'bool' ? (p.value ? '1' : '0') : String(p.value);
  return row;
}

async function openParamsEditor(sid) {
  psid = sid; pform.innerHTML = ''; pmsg.textContent = '';
  let d;
  try {
    const r = await fetch(`/api/sources/${sid}/params`);
    d = await r.json();
    if (!r.ok) { alert('Error: ' + d.error); return; }
  } catch (e) { alert('Error consultando parametros'); return; }
  document.getElementById('ptitle').textContent = 'Parametros — ' + d.type;
  for (const p of d.params) pform.appendChild(paramRow(p));
  pmodal.classList.add('open');
}

document.getElementById('pcancel').onclick = () => pmodal.classList.remove('open');

document.getElementById('psave').onclick = async () => {
  if (!psid) return;
  const data = {};
  for (const row of pform.querySelectorAll('.prow')) {
    const el = row.querySelector('.pval');
    const t = row.dataset.type;
    const cur = t === 'bool' ? (el.checked ? '1' : '0') : String(el.value);
    const dirty = (t === 'int' || t === 'float')
      ? Number(cur) !== Number(row.dataset.init)
      : cur !== row.dataset.init;
    if (!dirty) continue;               // solo mandar lo que el usuario EDITO
    data[row.dataset.key] = t === 'bool' ? el.checked : el.value;
  }
  if (!Object.keys(data).length) {
    pmsg.textContent = 'Sin cambios';
    pmsg.style.color = '#9ca3af';
    return;
  }
  const btn = document.getElementById('psave');
  btn.disabled = true;
  try {
    const r = await fetch(`/api/sources/${psid}/params`, {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(data)});
    const j = await r.json();
    if (!r.ok) { pmsg.textContent = 'Error: ' + j.error; return; }
    const errs = Object.entries(j.errors || {});
    pmsg.textContent = errs.length
      ? 'Con errores: ' + errs.map(([k, v]) => `${k}: ${v}`).join(' · ')
      : 'Aplicado ✓ (guardado para esta fuente)';
    pmsg.style.color = errs.length ? '#f87171' : '#4ade80';
    // re-pintar el form con los valores REALES del detector (un cambio de
    // escena, por ejemplo, re-ajusta conf/NMS/grid/modelo en cadena)
    try {
      const r2 = await fetch(`/api/sources/${psid}/params`);
      const d2 = await r2.json();
      if (r2.ok) {
        pform.innerHTML = '';
        for (const p of d2.params) pform.appendChild(paramRow(p));
      }
    } catch (e) {}
  } catch (e) {
    pmsg.textContent = 'Error aplicando parametros';
  } finally {
    btn.disabled = false;
  }
};

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>
"""
