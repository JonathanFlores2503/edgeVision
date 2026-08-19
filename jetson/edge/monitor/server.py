"""Servidor web del monitor edge — solo librería estándar (sin FastAPI/uvicorn).

Sirve el dashboard y la API. Acceso protegido por **sesión** (cookie): hay que
iniciar sesión (admin o usuario). El admin puede editar la Configuración; el
usuario solo puede ver.

  GET  /                 -> página HTML (incluye la pantalla de login)
  GET  /api/me           -> sesión actual (rol, si debe cambiar contraseña)
  POST /api/login        -> inicia sesión (set-cookie)
  POST /api/logout       -> cierra sesión
  POST /api/password     -> cambia contraseña (propia; admin puede cambiar otras)
  GET  /api/users        -> lista de cuentas (solo admin)
  GET  /api/status|cameras|events|config   -> datos (requiere sesión)
  POST /api/config       -> guarda config (solo admin)
  GET  /api/videos       -> biblioteca de vídeos subidos (requiere sesión)
  GET  /api/faces        -> escena en vivo de rostros/personas (grid bajo el vídeo)
  GET  /api/faces/gallery-> galería de rostros registrados (solo admin)
  GET  /rostro/<persona>/<archivo> -> foto de la galería (solo admin)
  POST /api/faces/photo?person=…&name=… -> sube una foto (binario, solo admin)
  POST /api/faces/delete -> borra una foto o una persona (solo admin)
  POST /api/faces/rebuild-> re-embebe la galería en caliente (solo admin)
  POST /api/videos/upload?name=… -> sube un vídeo (cuerpo binario, solo admin)
  POST /api/videos/delete        -> borra un vídeo (solo admin)
  GET  /video/<nombre>   -> reproduce un vídeo subido (con Range, requiere sesión)
  GET  /stream/<cam>     -> vídeo MJPEG anotado (requiere sesión)
  GET  /events           -> SSE en vivo (requiere sesión)
"""
from __future__ import annotations

import json
import logging
import queue
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from .. import branding as _branding
from ..alerts import AlertNotifier
from ..capture import video_file
from .rotations import norm_deg
from .settings import Settings
from .state import MonitorState

log = logging.getLogger(__name__)

_DIR = Path(__file__).parent
# El HTML se lee FRESCO en cada carga de la página (no se cachea en memoria), para
# que editar la interfaz no obligue a reiniciar el nodo. Es una sola página liviana.

# Archivos estáticos servidos tal cual (PWA + iconos).
_STATIC = {
    "/manifest.webmanifest": ("manifest.webmanifest", "application/manifest+json"),
    "/sw.js": ("sw.js", "application/javascript"),
    "/icon-192.png": ("icon-192.png", "image/png"),
    "/icon-512.png": ("icon-512.png", "image/png"),
    # Mapa offline: contorno de los estados de México (servido localmente, sin internet).
    "/assets/mexico_states.json": ("assets/mexico_states.json", "application/json"),
}

_COOKIE = "tln_sess"


class _Sessions:
    """Sesiones en memoria: token -> {username, role, must_change, exp}."""

    def __init__(self, ttl_s: int = 12 * 3600):
        self._d: dict = {}
        self._lock = threading.Lock()
        self.ttl = ttl_s

    def create(self, username: str, role: str, must_change: bool) -> str:
        tok = secrets.token_urlsafe(24)
        with self._lock:
            self._d[tok] = {"username": username, "role": role,
                            "must_change": must_change, "exp": time.time() + self.ttl}
        return tok

    def get(self, tok: str | None) -> dict | None:
        if not tok:
            return None
        with self._lock:
            s = self._d.get(tok)
            if not s:
                return None
            if s["exp"] < time.time():
                self._d.pop(tok, None)
                return None
            return dict(s)

    def update(self, tok: str, **kw) -> None:
        with self._lock:
            if tok in self._d:
                self._d[tok].update(kw)

    def destroy(self, tok: str | None) -> None:
        if tok:
            with self._lock:
                self._d.pop(tok, None)


def _cameras_using(settings: Settings, video_name: str) -> list:
    """Ids de las cámaras configuradas que reproducen ese vídeo."""
    target = video_file.safe_name(video_name)
    if not target:
        return []
    out = []
    for c in (settings.data().get("cameras") or []):
        if video_file.source_name(c.get("url") or "") == target:
            out.append(c.get("id") or "")
    return [c for c in out if c]


def _zone_capable(model) -> bool:
    """¿El modelo activo edita zonas por cámara? (duck-typing, modelo-agnóstico)."""
    fn = getattr(model, "supports_zones", None)
    try:
        return bool(callable(fn) and fn())
    except Exception:  # noqa: BLE001
        return False


def _evidence_capable(model) -> bool:
    """¿El modelo activo guarda testigos (fotos por evento)? (duck-typing)."""
    fn = getattr(model, "supports_evidence", None)
    try:
        return bool(callable(fn) and fn())
    except Exception:  # noqa: BLE001
        return False


def _faces_capable(model) -> bool:
    """¿El modelo activo publica la escena de rostros/personas? (duck-typing)."""
    fn = getattr(model, "supports_faces", None)
    try:
        return bool(callable(fn) and fn())
    except Exception:  # noqa: BLE001
        return False


def _gallery_capable(model) -> bool:
    """¿El modelo activo tiene galería de rostros gestionable? (duck-typing)."""
    fn = getattr(model, "supports_gallery", None)
    try:
        return bool(callable(fn) and fn())
    except Exception:  # noqa: BLE001
        return False


def _model_info(model_cfg, settings: Settings, model=None, running: str = "") -> dict:
    """Resumen del modelo en uso + pesos VAD disponibles + catálogo de modelos
    (familias VAD y heurística) para el panel de admin.

    `running` es la etiqueta del modelo que el nodo cargó AL ARRANCAR. Se compara
    con la selección guardada para avisar de lo que más despista del panel: haber
    guardado un modelo distinto del que está corriendo y no haber reiniciado."""
    from ..inference import registry  # import diferido (evita arrastrar torch aquí)

    families = registry.list_models(model_cfg)
    active = (settings.model_selection() if settings else {}) or {}
    if not active:  # sin selección guardada -> el VAD configurado por defecto
        active = {"family": "vad", "key": "vad"}
    try:
        active_label = registry.resolve(active, model_cfg).label
    except Exception:  # noqa: BLE001 — el panel no depende de esto
        active_label = ""
    restart = bool(running and active_label and running != active_label)

    # Editor de zonas: solo si el modelo VIVO lo soporta (p. ej. Contador de flujo).
    zones = {"enabled": _zone_capable(model)}
    if zones["enabled"] and callable(getattr(model, "zone_schema", None)):
        try:
            zones["schema"] = model.zone_schema()
        except Exception:  # noqa: BLE001
            pass

    live = {"active_label": active_label, "running": running,
            "restart_pending": restart}
    if model_cfg is None:
        return {"available": False, "families": families, "active": active,
                "zones": zones, **live}

    name = getattr(model_cfg, "name", "")
    ov = settings.model_override() if settings else {}
    wdir = Path(model_cfg.weights_dir) / name
    files = sorted(p.name for p in wdir.glob("*.pkl")) if wdir.exists() else []
    det = ov.get("detector") or model_cfg.weights_detector
    cls = ov.get("classifier") or model_cfg.weights_classifier
    avail_det = avail_cls = files
    switchable = True
    directory = str(wdir)
    return {
        "available": True, "name": name, "backend": "torch",
        "device": getattr(model_cfg, "device", ""),
        "detector": det, "classifier": cls,
        "available_detectors": avail_det, "available_classifiers": avail_cls,
        "switchable": switchable, "dir": directory, "overridden": bool(ov),
        # Catálogo modular:
        "families": families, "active": active, "zones": zones,
        # Qué corre AHORA vs qué está guardado:
        **live,
    }


def _make_handler(state: MonitorState, settings: Settings, sessions: _Sessions,
                  on_cameras_changed=None, model_cfg=None, model=None):
    def _apply_live():
        """Aplica en caliente lo recién guardado (cámaras + umbrales). No revienta
        el guardado si algo falla."""
        if on_cameras_changed is None:
            return
        try:
            on_cameras_changed()
        except Exception:  # noqa: BLE001
            log.exception("Error aplicando cambios en caliente tras guardar.")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # silenciar logs por request
            pass

        def handle_one_request(self):
            # El navegador cierra/refresca pestañas de stream MJPEG constantemente;
            # el socket se resetea mientras se lee la siguiente request. Es benigno:
            # lo tragamos para no ensuciar el log con tracebacks de socketserver.
            try:
                super().handle_one_request()
            except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
                self.close_connection = True

        # ── helpers de respuesta ─────────────────────────────────────────────
        def _headers(self, code=200, ctype="text/html; charset=utf-8", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _json(self, obj, code=200, extra=None):
            body = json.dumps(obj).encode()
            hdr = {"Content-Length": str(len(body))}
            hdr.update(extra or {})
            self._headers(code, "application/json", hdr)
            self.wfile.write(body)

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            if not n:
                return {}
            try:
                obj = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                return obj if isinstance(obj, dict) else {}
            except json.JSONDecodeError:
                return {}

        # ── sesión ───────────────────────────────────────────────────────────
        def _token(self) -> str | None:
            raw = self.headers.get("Cookie", "")
            for part in raw.split(";"):
                k, _, v = part.strip().partition("=")
                if k == _COOKIE:
                    return v
            return None

        def _session(self) -> dict | None:
            return sessions.get(self._token())

        # ── selección de modelo (admin) ──────────────────────────────────────
        def _handle_model_post(self, body: dict, model_cfg, settings: Settings) -> dict:
            """Procesa POST /api/model: selección de familia/modelo (modular) y/o
            cambio de pesos del VAD torch. Todo se aplica al reiniciar el nodo."""
            from ..inference import registry
            out = {"ok": True, "restart_required": False}

            # 1) Selección de familia/modelo (VAD vs heurístico).
            family = (body.get("family") or "").strip()
            key = (body.get("key") or "").strip()
            if family and key:
                spec = registry._all_specs(model_cfg).get((family, key))
                if spec is None:
                    return {"ok": False, "error": f"modelo desconocido: {family}/{key}"}
                settings.set_model_selection(family, key)
                out["restart_required"] = True
                # Se permite pre-seleccionar un modelo no disponible en ESTE equipo
                # (p. ej. para desplegar en la Jetson). Se avisa; si al reiniciar no
                # puede construirse, el nodo cae a MOCK.
                if not spec.available:
                    out["warning"] = (f"«{spec.label}» no está disponible aquí "
                                      f"({spec.unavailable_reason}). Correrá donde estén "
                                      f"sus dependencias; si no, el nodo usará MOCK.")

            # 2) Cambio de pesos del VAD torch (detector/clasificador), como antes.
            det = (body.get("detector") or "").strip()
            cls = (body.get("classifier") or "").strip()
            if det or cls:
                if model_cfg is None:
                    return {"ok": False, "error": "no hay modelo configurado"}
                wdir = Path(model_cfg.weights_dir) / model_cfg.name
                if not (wdir / det).exists():
                    return {"ok": False, "error": "detector inválido"}
                if not (wdir / cls).exists():
                    return {"ok": False, "error": "clasificador inválido"}
                settings.set_model_override(det, cls)
                out["restart_required"] = True

            if not out["restart_required"]:
                return {"ok": False, "error": "nada que guardar"}
            return out

        # ── GET ──────────────────────────────────────────────────────────────
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            # Públicos (sin sesión): página, estáticos y /api/me.
            if path == "/":
                html = (_DIR / "index.html").read_text(encoding="utf-8")
                for tok, val in _branding.tokens().items():  # marca white-label
                    html = html.replace(tok, val)
                html = html.encode("utf-8")
                self._headers(extra={"Content-Length": str(len(html)),
                                     "Cache-Control": "no-cache"})
                self.wfile.write(html)
                return
            if path in _STATIC:
                fname, ctype = _STATIC[path]
                if fname == "manifest.webmanifest":  # marca white-label en la PWA
                    txt = (_DIR / fname).read_text(encoding="utf-8")
                    for tok, val in _branding.tokens().items():
                        txt = txt.replace(tok, val)
                    body = txt.encode("utf-8")
                else:
                    body = (_DIR / fname).read_bytes()
                self._headers(ctype=ctype, extra={"Content-Length": str(len(body)),
                                                  "Cache-Control": "no-cache"})
                self.wfile.write(body)
                return
            if path == "/api/me":
                s = self._session()
                if s:
                    self._json({"authenticated": True, "username": s["username"],
                                "role": s["role"], "must_change": s["must_change"]})
                else:
                    self._json({"authenticated": False})
                return

            # A partir de aquí, requiere sesión.
            s = self._session()
            if not s:
                self._json({"error": "no autenticado"}, code=401)
                return

            if path == "/api/status":
                self._json(state.status())
            elif path == "/api/cameras":
                self._json(state.cameras())
            elif path == "/api/events":
                self._json(state.recent_events())
            elif path == "/api/config":
                if s["role"] != "admin":
                    self._json({"error": "solo admin"}, code=403)
                else:
                    self._json(settings.data())
            elif path == "/api/users":
                if s["role"] != "admin":
                    self._json({"error": "solo admin"}, code=403)
                else:
                    self._json(settings.list_users())
            elif path == "/api/model":
                if s["role"] != "admin":
                    self._json({"error": "solo admin"}, code=403)
                else:
                    # `state.status()["model"]` es la etiqueta con la que arrancó el
                    # nodo: es la única verdad sobre qué modelo está corriendo.
                    self._json(_model_info(model_cfg, settings, model,
                                           running=state.status().get("model", "")))
            elif path == "/api/zones":
                if s["role"] != "admin":
                    self._json({"error": "solo admin"}, code=403)
                elif not _zone_capable(model):
                    self._json({"supported": False})
                else:
                    from urllib.parse import parse_qs, urlparse
                    q = parse_qs(urlparse(self.path).query)
                    cam = (q.get("camera", [""])[0] or "").strip()
                    if not cam:
                        self._json({"supported": True, "error": "falta camera"}, code=400)
                    else:
                        self._json(model.get_zones(cam))
            elif path == "/api/counts":
                # Conteos entradas/salidas/dentro por cámara (si el modelo cuenta).
                fn = getattr(model, "get_counts", None)
                if callable(getattr(model, "supports_counts", None)) \
                        and model.supports_counts() and callable(fn):
                    try:
                        self._json({"supported": True, "cameras": fn()})
                    except Exception:  # noqa: BLE001
                        self._json({"supported": True, "cameras": {}})
                else:
                    self._json({"supported": False, "cameras": {}})
            elif path == "/api/faces":
                # Escena en vivo del modelo de rostros: personas, rostros con nombre y
                # confianza, y los parámetros del detector. Es lo que el dashboard pinta
                # en el grid DEBAJO del vídeo (antes iba quemado en el frame).
                if not _faces_capable(model):
                    self._json({"supported": False, "cameras": {}})
                else:
                    try:
                        d = model.get_faces()
                        d["supported"] = True
                        self._json(d)
                    except Exception:  # noqa: BLE001 — el grid no tumba el panel
                        log.exception("Error leyendo la escena de rostros.")
                        self._json({"supported": True, "cameras": {},
                                    "error": "no se pudo leer la escena"}, code=500)
            elif path == "/api/faces/gallery":
                # Solo admin: son fotos de personas identificables con su nombre, el
                # dato más sensible del nodo.
                if s["role"] != "admin":
                    self._json({"error": "solo admin"}, code=403)
                elif not _gallery_capable(model):
                    self._json({"supported": False, "personas": []})
                else:
                    try:
                        d = model.get_gallery()
                        d["supported"] = True
                        self._json(d)
                    except Exception:  # noqa: BLE001
                        log.exception("Error listando la galería de rostros.")
                        self._json({"supported": True, "personas": [],
                                    "error": "no se pudo leer la galería"}, code=500)
            elif path.startswith("/rostro/"):
                self._rostro(s, unquote(path[len("/rostro/"):]))
            elif path == "/api/testigos":
                # Galería de evidencia del modelo activo. Solo admin: son fotos
                # de personas identificables, no un dato operativo más.
                if s["role"] != "admin":
                    self._json({"error": "solo admin"}, code=403)
                elif not _evidence_capable(model):
                    self._json({"supported": False, "cameras": [], "items": []})
                else:
                    from urllib.parse import parse_qs, urlparse
                    q = parse_qs(urlparse(self.path).query)
                    cam = (q.get("camera", [""])[0] or "").strip()
                    try:
                        lim = int(q.get("limit", ["60"])[0])
                    except (TypeError, ValueError):
                        lim = 60
                    try:
                        self._json(model.list_evidence(cam, lim))
                    except Exception:  # noqa: BLE001 — la galería no tumba el panel
                        log.exception("Error listando testigos de '%s'.", cam)
                        self._json({"supported": True, "cameras": [], "items": [],
                                    "error": "no se pudo leer la evidencia"}, code=500)
            elif path.startswith("/testigo/"):
                self._testigo(s, unquote(path[len("/testigo/"):]))
            elif path.startswith("/snapshot/"):
                self._snapshot(unquote(path[len("/snapshot/"):]))
            elif path == "/api/thresholds":
                d = settings.data()
                self._json({"global": (d.get("alerts") or {}).get("score_threshold"),
                            "cameras": d.get("camera_thresholds", {})})
            elif path == "/api/rotations":
                self._json({"cameras": settings.data().get("camera_rotations", {})})
            elif path == "/api/videos":
                # Biblioteca de vídeos subidos: la ve cualquier sesión (para saber
                # qué se está reproduciendo); subir/borrar es solo de admin.
                self._json({"dir": str(video_file.videos_dir()),
                            "max_mb": video_file.max_bytes() // (1024 * 1024),
                            "formats": sorted(e[1:] for e in video_file.VIDEO_EXTS),
                            "items": video_file.list_videos()})
            elif path.startswith("/video/"):
                self._video(unquote(path[len("/video/"):]))
            elif path.startswith("/stream/"):
                self._stream(unquote(path[len("/stream/"):]))
            elif path == "/events":
                self._sse()
            else:
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")

        # ── POST ─────────────────────────────────────────────────────────────
        def do_POST(self):
            path = self.path.split("?", 1)[0]

            # Login: sin sesión previa.
            if path == "/api/login":
                body = self._read_json()
                u = (body.get("username") or "").strip()
                p = body.get("password") or ""
                info = settings.verify_user(u, p)
                if not info:
                    self._json({"ok": False, "error": "Usuario o contraseña incorrectos"},
                               code=401)
                    return
                tok = sessions.create(info["username"], info["role"], info["must_change"])
                cookie = (f"{_COOKIE}={tok}; Path=/; HttpOnly; SameSite=Lax; "
                          f"Max-Age={sessions.ttl}")
                self._json({"ok": True, "role": info["role"], "username": info["username"],
                            "must_change": info["must_change"]},
                           extra={"Set-Cookie": cookie})
                return

            if path == "/api/logout":
                tok = self._token()
                sessions.destroy(tok)
                self._json({"ok": True},
                           extra={"Set-Cookie": f"{_COOKIE}=; Path=/; Max-Age=0"})
                return

            # Resto: requiere sesión.
            s = self._session()
            if not s:
                self._json({"ok": False, "error": "no autenticado"}, code=401)
                return

            if path == "/api/password":
                body = self._read_json()
                target = (body.get("username") or s["username"]).strip()
                newpw = body.get("password") or ""
                # Un usuario normal solo puede cambiar su propia contraseña.
                if s["role"] != "admin" and target != s["username"]:
                    self._json({"ok": False, "error": "solo admin"}, code=403)
                    return
                if len(newpw) < 4:
                    self._json({"ok": False, "error": "mínimo 4 caracteres"})
                    return
                ok = settings.set_password(target, newpw)
                if ok and target == s["username"]:
                    sessions.update(self._token(), must_change=False)
                self._json({"ok": ok, "error": "" if ok else "usuario no existe"})
                return

            # Acciones de admin.
            if path in ("/api/config", "/api/alert/test", "/api/telegram/detect",
                        "/api/camera_threshold", "/api/camera_rotation",
                        "/api/users", "/api/users/delete", "/api/model", "/api/zones",
                        "/api/counts", "/api/videos/upload", "/api/videos/delete",
                        "/api/faces/photo", "/api/faces/delete",
                        "/api/faces/rebuild"):
                if s["role"] != "admin":
                    # Una subida rechazada deja el cuerpo (cientos de MB) sin leer:
                    # cerramos la conexión en vez de intentar drenarlo.
                    if path in ("/api/videos/upload", "/api/faces/photo"):
                        self.close_connection = True
                    self._json({"ok": False, "error": "solo admin"}, code=403)
                    return

            if path == "/api/config":
                patch = self._read_json()
                saved = settings.update(patch)
                # Aplica en caliente (sin reiniciar): recarga cámaras y umbrales.
                _apply_live()
                self._json({"ok": True, "settings": saved})
            elif path == "/api/camera_threshold":
                body = self._read_json()
                cam = (body.get("camera_id") or "").strip()
                if not cam:
                    self._json({"ok": False, "error": "falta camera_id"})
                    return
                raw = body.get("thr")
                # thr vacío/None -> quitar override (la cámara vuelve al umbral global).
                thr = None
                if raw not in (None, ""):
                    try:
                        thr = max(0.0, min(1.0, float(raw)))
                    except (TypeError, ValueError):
                        self._json({"ok": False, "error": "thr inválido (0–1)"})
                        return
                settings.set_camera_threshold(cam, thr)
                _apply_live()  # refresca el umbral en motor y monitor al instante
                self._json({"ok": True, "camera_id": cam, "thr": thr})
            elif path == "/api/camera_rotation":
                body = self._read_json()
                cam = (body.get("camera_id") or "").strip()
                if not cam:
                    self._json({"ok": False, "error": "falta camera_id"})
                    return
                deg = norm_deg(body.get("rotate"), -1)
                if deg < 0:
                    self._json({"ok": False, "error": "rotate inválido (0/90/180/270)"})
                    return
                settings.set_camera_rotation(cam, deg)
                _apply_live()  # la captura toma la nueva rotación al instante
                self._json({"ok": True, "camera_id": cam, "rotate": deg})
            elif path == "/api/users":
                body = self._read_json()
                res = settings.create_user(body.get("username") or "",
                                           body.get("password") or "",
                                           body.get("role") or "user")
                self._json(res, code=200 if res.get("ok") else 400)
            elif path == "/api/users/delete":
                body = self._read_json()
                res = settings.delete_user(body.get("username") or "", s["username"])
                self._json(res, code=200 if res.get("ok") else 400)
            elif path == "/api/model":
                body = self._read_json()
                self._json(self._handle_model_post(body, model_cfg, settings))
            elif path == "/api/zones":
                if not _zone_capable(model):
                    self._json({"ok": False, "error": "el modelo activo no edita zonas"})
                    return
                body = self._read_json()
                cam = (body.get("camera_id") or "").strip()
                if not cam:
                    self._json({"ok": False, "error": "falta camera_id"})
                    return
                opts = {"n_lines": body.get("n_lines", 5),
                        "invert": body.get("invert", False),
                        "axis_rot": body.get("axis_rot", False)}
                self._json(model.set_zones(cam, body.get("regions"), opts))
            elif path == "/api/counts":
                # Fija cuántas personas hay ADENTRO de una cámara (reinicia el
                # contador y cuenta desde ahí). Solo modelos que cuentan.
                fn = getattr(model, "set_counts", None)
                if not (callable(getattr(model, "supports_counts", None))
                        and model.supports_counts() and callable(fn)):
                    self._json({"ok": False, "error": "el modelo activo no cuenta"})
                    return
                body = self._read_json()
                cam = (body.get("camera_id") or "").strip()
                if not cam:
                    self._json({"ok": False, "error": "falta camera_id"})
                    return
                self._json(fn(cam, body.get("inside")))
            elif path == "/api/faces/photo":
                self._upload_rostro()
            elif path in ("/api/faces/delete", "/api/faces/rebuild"):
                if not _gallery_capable(model):
                    self._json({"ok": False,
                                "error": "el modelo activo no tiene galería de rostros"},
                               code=400)
                    return
                if path == "/api/faces/rebuild":
                    try:
                        self._json(model.gallery_rebuild())
                    except Exception:  # noqa: BLE001
                        log.exception("Error reconstruyendo la galería de rostros.")
                        self._json({"ok": False,
                                    "error": "no se pudo reconstruir"}, code=500)
                    return
                body = self._read_json()
                try:
                    self._json(model.gallery_delete(
                        (body.get("persona") or "").strip(),
                        (body.get("foto") or "").strip()))
                except Exception:  # noqa: BLE001
                    log.exception("Error borrando de la galería de rostros.")
                    self._json({"ok": False, "error": "no se pudo borrar"}, code=500)
            elif path == "/api/videos/upload":
                self._upload_video()
            elif path == "/api/videos/delete":
                body = self._read_json()
                name = (body.get("name") or "").strip()
                ok = video_file.delete(name)
                # Si alguna cámara reproducía ese vídeo, se quedará "sin señal":
                # se avisa en la respuesta para que el panel lo diga en claro.
                self._json({"ok": ok,
                            "error": "" if ok else "no existe ese vídeo",
                            "in_use": _cameras_using(settings, name)})
            elif path == "/api/alert/test":
                self._read_json()  # consume el cuerpo si lo hay
                # Adjunta el frame de la 1ª cámara viva (si hay) para probar el
                # envío de FOTO por Telegram, no solo el texto.
                snap = None
                for cam in state.cameras():
                    snap = state.get_jpeg(cam)
                    if snap:
                        break
                res = AlertNotifier(settings).send_test(image=snap)
                self._json({"ok": res["sent"] > 0, "sent": res["sent"],
                            "errors": res["errors"], "preview": res["text"],
                            "with_photo": bool(snap)})
            elif path == "/api/telegram/detect":
                self._read_json()
                from ..alerts.telegram import get_chat_ids
                token = (settings.data().get("alerts", {}) or {}).get("telegram_bot_token", "")
                if not (token or "").strip():
                    self._json({"ok": False, "error": "Guarda primero el token del bot."})
                else:
                    ok, chats, detail = get_chat_ids(token)
                    self._json({"ok": ok, "chats": chats, "error": detail})
            else:
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")

        # ── vídeos subidos ───────────────────────────────────────────────────
        def _upload_video(self):
            """Recibe un vídeo (cuerpo binario crudo, `?name=…`) y lo guarda.

            Sin multipart a propósito: el navegador manda el File tal cual y aquí
            se escribe a disco por trozos, sin cargar cientos de MB en la RAM del
            nodo (que es justo el recurso escaso de la placa)."""
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            name = (q.get("name", [""])[0] or "").strip()
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                length = 0
            try:
                info = video_file.save_stream(name, self.rfile, length)
            except video_file.UploadError as e:
                # El cuerpo queda a medio leer: esta conexión ya no es reutilizable.
                self.close_connection = True
                self._json({"ok": False, "error": str(e)}, code=400)
                return
            except Exception:  # noqa: BLE001 — una subida rara no tumba el panel
                log.exception("Error guardando el vídeo subido '%s'.", name)
                self.close_connection = True
                self._json({"ok": False, "error": "no se pudo guardar el vídeo"},
                           code=500)
                return
            self._json({"ok": True, "video": info})

        def _upload_rostro(self):
            """Recibe UNA foto de la galería: `?person=<Nombre>&name=<archivo>.jpg`,
            cuerpo binario crudo (igual que los vídeos, sin multipart).

            A diferencia del vídeo esta sí se lee entera en memoria: son fotos de pocos
            MB y el modelo necesita los bytes, no un fichero a medias. El tope lo pone
            el propio modelo (`MAX_FOTO_MB`), y aquí solo se respeta el Content-Length
            para no leer un cuerpo enorme antes de rechazarlo."""
            from urllib.parse import parse_qs, urlparse
            if not _gallery_capable(model):
                self.close_connection = True
                self._json({"ok": False,
                            "error": "el modelo activo no tiene galería de rostros"},
                           code=400)
                return
            q = parse_qs(urlparse(self.path).query)
            persona = (q.get("person", [""])[0] or "").strip()
            nombre = (q.get("name", [""])[0] or "").strip()
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except ValueError:
                length = 0
            tope = int(getattr(model, "MAX_FOTO_MB", 12)) * 1024 * 1024
            if length <= 0:
                self.close_connection = True
                self._json({"ok": False, "error": "cuerpo vacío"}, code=400)
                return
            if length > tope:
                # No se lee: rechazar y cerrar es más barato que tragarse los bytes.
                self.close_connection = True
                self._json({"ok": False,
                            "error": f"la foto pasa de {tope // (1024 * 1024)} MB"},
                           code=400)
                return
            try:
                datos = self.rfile.read(length)
            except OSError:
                self.close_connection = True
                self._json({"ok": False, "error": "subida interrumpida"}, code=400)
                return
            try:
                self._json(model.gallery_add_photo(persona, nombre, datos))
            except Exception:  # noqa: BLE001 — una subida rara no tumba el panel
                log.exception("Error guardando la foto '%s' de '%s'.", nombre, persona)
                self._json({"ok": False, "error": "no se pudo guardar la foto"},
                           code=500)

        def _rostro(self, s: dict, rest: str):
            """Sirve una foto de la galería: `/rostro/<persona>/<archivo>`.

            El nombre lo valida el modelo (`gallery_photo_path`), que es quien sabe
            dónde vive su galería; aquí solo se comprueba el rol y se entregan bytes."""
            if s["role"] != "admin":
                self._headers(403, "text/plain", {"Content-Length": "10"})
                self.wfile.write(b"solo admin")
                return
            persona, _, fname = rest.partition("/")
            path_foto = None
            if _gallery_capable(model) and persona and fname:
                try:
                    path_foto = model.gallery_photo_path(persona, fname)
                except Exception:  # noqa: BLE001
                    log.exception("Error resolviendo la foto '%s/%s'.", persona, fname)
            if path_foto is None:
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")
                return
            try:
                body = path_foto.read_bytes()
            except OSError:
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")
                return
            ext = path_foto.suffix.lower()
            ctype = {".png": "image/png", ".webp": "image/webp",
                     ".bmp": "image/bmp"}.get(ext, "image/jpeg")
            # A diferencia de los testigos, estas fotos SÍ cambian (se borran y se
            # vuelven a subir con el mismo nombre), así que no se cachean.
            self._headers(ctype=ctype,
                          extra={"Content-Length": str(len(body)),
                                 "Cache-Control": "no-store"})
            self.wfile.write(body)

        def _video(self, name: str):
            """Sirve un vídeo de la biblioteca, con soporte de Range para que el
            navegador pueda previsualizarlo y saltar dentro de él."""
            path = video_file.resolve(name)
            if path is None or path.parent != video_file.videos_dir():
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")
                return
            size = path.stat().st_size
            start, end, code = 0, size - 1, 200
            rng = (self.headers.get("Range") or "").strip()
            if rng.startswith("bytes="):
                first, _, last = rng[6:].partition("-")
                try:
                    if first:
                        start = int(first)
                        end = int(last) if last else size - 1
                    elif last:                    # sufijo: últimos N bytes
                        start = max(size - int(last), 0)
                except ValueError:
                    start, end = 0, size - 1
                else:
                    end = min(end, size - 1)
                    if start > end:
                        self._headers(416, "text/plain",
                                      {"Content-Range": f"bytes */{size}",
                                       "Content-Length": "0"})
                        return
                    code = 206
            n = end - start + 1
            hdr = {"Content-Length": str(n), "Accept-Ranges": "bytes",
                   "Cache-Control": "private, max-age=60"}
            if code == 206:
                hdr["Content-Range"] = f"bytes {start}-{end}/{size}"
            self._headers(code, video_file.content_type(path), hdr)
            try:
                with path.open("rb") as fh:
                    fh.seek(start)
                    while n > 0:
                        buf = fh.read(min(1024 * 256, n))
                        if not buf:
                            break
                        self.wfile.write(buf)
                        n -= len(buf)
            except OSError:
                self.close_connection = True   # el navegador cortó (saltó/cerró)

        # ── streaming ────────────────────────────────────────────────────────
        def _stream(self, camera_id: str):
            boundary = "frame"
            self.send_response(200)
            self.send_header("Content-Type",
                             f"multipart/x-mixed-replace; boundary={boundary}")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                while True:
                    jpeg = state.get_jpeg(camera_id)
                    if jpeg:
                        self.wfile.write(f"--{boundary}\r\n".encode())
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.07)  # ~14 fps de visualización
            except OSError:
                pass  # el cliente cerró el stream (broken pipe / abort / reset)

        def _snapshot(self, camera_id: str):
            """Un solo JPEG del frame actual (fondo del editor de zonas)."""
            jpeg = state.get_raw_jpeg(camera_id)
            if not jpeg:
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")
                return
            self._headers(ctype="image/jpeg",
                          extra={"Content-Length": str(len(jpeg)),
                                 "Cache-Control": "no-store"})
            self.wfile.write(jpeg)

        def _testigo(self, s: dict, rest: str):
            """Sirve una foto de testigo: `/testigo/<camara>/<fichero>.jpg`.

            La validación del nombre la hace el modelo (`evidence_file`), que es
            quien sabe dónde vive su evidencia; aquí solo se comprueba el rol y
            se entregan los bytes."""
            if s["role"] != "admin":
                self._headers(403, "text/plain", {"Content-Length": "10"})
                self.wfile.write(b"solo admin")
                return
            cam, _, fname = rest.partition("/")
            path = None
            if _evidence_capable(model) and cam and fname:
                try:
                    path = model.evidence_file(cam, fname)
                except Exception:  # noqa: BLE001
                    log.exception("Error resolviendo el testigo '%s/%s'.", cam, fname)
            if path is None:
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")
                return
            try:
                body = path.read_bytes()
            except OSError:
                self._headers(404, "text/plain", {"Content-Length": "9"})
                self.wfile.write(b"not found")
                return
            # Las fotos son inmutables (el nombre lleva fecha y hora), así que se
            # pueden cachear: la galería carga decenas de miniaturas de golpe.
            self._headers(ctype="image/jpeg",
                          extra={"Content-Length": str(len(body)),
                                 "Cache-Control": "private, max-age=3600"})
            self.wfile.write(body)

        def _sse(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q = state.subscribe()
            try:
                while True:
                    try:
                        data = q.get(timeout=15)
                    except queue.Empty:  # timeout -> ping para mantener viva la conexión
                        self.wfile.write(b": ping\n\n")
                    else:
                        self.wfile.write(f"data: {data}\n\n".encode())
            except OSError:
                pass  # el cliente cerró la conexión SSE
            finally:
                state.unsubscribe(q)

    return Handler


class _Httpd(ThreadingHTTPServer):
    """Servidor del dashboard endurecido contra la falta de RAM.

    La placa va justa de memoria y el modelo se lleva la mayor parte. Cuando se agota,
    `socketserver` no puede crear el hilo de la petición y la excepción sube hasta el
    bucle de `accept`: el hilo del monitor moría, el nodo seguía vivo, el puerto seguía
    en LISTEN y el dashboard dejaba de responder **sin una sola línea en el log**.
    """

    daemon_threads = True
    #: Cola de accept. El defecto de socketserver es 5: con el nodo cargado se llenaba
    #: y el navegador veía "Error de conexión" en vez de esperar su turno.
    request_queue_size = 64

    def handle_error(self, request, client_address):
        # El defecto imprime el traceback pelado en stderr, fuera del logging del nodo.
        log.exception("Error atendiendo una petición de %s.",
                      client_address[0] if client_address else "?")


class MonitorServer:
    #: Espera entre reintentos del bucle de accept, si se cae.
    RETRY_MIN_S = 1.0
    RETRY_MAX_S = 30.0

    def __init__(self, state: MonitorState, host: str = "0.0.0.0", port: int = 8080,
                 settings: Settings | None = None, on_cameras_changed=None,
                 model_cfg=None, model=None):
        self.state = state
        self.settings = settings or Settings()
        self.sessions = _Sessions()
        self._httpd = _Httpd(
            (host, port),
            _make_handler(state, self.settings, self.sessions, on_cameras_changed,
                          model_cfg, model))
        self._stopping = threading.Event()
        self._serving = threading.Event()
        self._thread = threading.Thread(target=self._serve_supervised,
                                        name="monitor", daemon=True)
        self.port = port

    def _serve_supervised(self):
        """`serve_forever` vigilado: si el bucle de accept se cae, se avisa y se vuelve.

        Un dashboard que no responde es malo; uno que no responde y no lo dice cuesta
        horas de diagnóstico equivocado.
        """
        wait_s = self.RETRY_MIN_S
        while not self._stopping.is_set():
            try:
                self._serving.set()
                # poll_interval corto para que stop() no tenga que esperar medio segundo.
                self._httpd.serve_forever(poll_interval=0.2)
                return                      # salida limpia: alguien llamó a stop()
            except Exception:               # noqa: BLE001 — este bucle no puede morir callado
                if self._stopping.is_set():
                    return
                log.exception("El bucle del dashboard se cayó; reintento en %.0f s. "
                              "La causa habitual es falta de RAM: mira `free -m`.", wait_s)
                self._stopping.wait(wait_s)
                wait_s = min(wait_s * 2, self.RETRY_MAX_S)
            finally:
                self._serving.clear()

    def start(self):
        self._thread.start()
        log.info("Monitor web en http://localhost:%d", self.port)

    def stop(self):
        self._stopping.set()
        # Solo se le pide parar si el bucle está corriendo: `shutdown()` espera a que
        # `serve_forever` salga, y si no está dentro (p. ej. esperando un reintento)
        # esa espera no termina nunca.
        if self._serving.is_set():
            try:
                self._httpd.shutdown()
            except Exception:  # noqa: BLE001 — el apagado no se detiene por esto
                log.exception("Error parando el bucle del dashboard.")
        self._thread.join(timeout=5.0)
        try:
            self._httpd.server_close()      # libera el puerto para el siguiente arranque
        except Exception:  # noqa: BLE001
            log.exception("Error cerrando el socket del dashboard.")
