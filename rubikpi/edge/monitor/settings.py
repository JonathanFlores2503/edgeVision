"""Configuración editable desde la web (independiente de config.yaml).

El usuario edita en la pestaña *Configuración* del monitor: datos del vehículo,
cámaras y contactos de alerta (SMS, correo, WhatsApp, Telegram). Esto se persiste
en un JSON aparte (``data/settings.json``) para NO reescribir el ``config.yaml``
del nodo (que carga/valida el pipeline). Local-first: vive en el disco del nodo.

También guarda las **cuentas** de acceso (admin/usuario) con la contraseña
*hasheada* (PBKDF2-HMAC-SHA256, nunca en texto plano).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# Claves leídas/escritas en disco.
_PERSIST_KEYS = ("vehicle", "cameras", "alerts", "users", "camera_thresholds",
                 "camera_rotations", "model_override", "model_selection")
# Claves que la pestaña Configuración (/api/config) puede modificar.
# `users` y `camera_thresholds` quedan FUERA a propósito: se gestionan por
# endpoints dedicados (auth y umbral por cámara), no por el guardado masivo.
_WEB_KEYS = ("vehicle", "cameras", "alerts")

_ITER = 120_000  # iteraciones PBKDF2


def _hash_pw(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                               bytes.fromhex(salt), _ITER).hex()


def make_user(username: str, password: str, role: str) -> Dict[str, Any]:
    """Crea un registro de usuario con sal aleatoria y contraseña hasheada."""
    salt = os.urandom(16).hex()
    return {"username": username, "role": role, "salt": salt,
            "hash": _hash_pw(password, salt), "is_default": True}


def default_settings(cfg=None) -> Dict[str, Any]:
    """Estructura por defecto, sembrada desde el EdgeConfig si se pasa."""
    vehicle = {"id": "", "site": "", "plate": "", "route": "", "driver": ""}
    cameras: List[dict] = []
    threshold = 0.55
    if cfg is not None:
        vehicle["id"] = getattr(cfg.device, "id", "") or ""
        vehicle["site"] = getattr(cfg.device, "site", "") or ""
        cameras = [{"id": c.id, "url": c.url, "enabled": bool(c.enabled)}
                   for c in getattr(cfg, "cameras", [])]
        threshold = float(getattr(cfg.events, "score_threshold", 0.55))
    return {
        "vehicle": vehicle,
        "cameras": cameras,
        # Umbral de anomalía por cámara (camera_id -> thr). Vacío = todas usan el
        # umbral global (alerts.score_threshold). Se edita desde el tile en vivo.
        "camera_thresholds": {},
        # Rotación por cámara (camera_id -> grados 0/90/180/270). Vacío = sin
        # rotación. Se edita desde el tile en vivo (sirve para RTSP y USB).
        "camera_rotations": {},
        "alerts": {
            "enabled": False,              # interruptor maestro de envío AUTOMÁTICO (off por defecto)
            "sms": [],                     # números para SMS (link de ubicación incluido)
            "emails": [],                  # correos donde mandar la información
            "whatsapp": [],                # números de WhatsApp
            "telegram": [],                # chat ids / @usuarios de Telegram
            "telegram_bot_token": "",      # token del bot (@BotFather)
            "include_location_link": True, # adjuntar link del mapa al activarse
            "map_provider": "google",      # google | osm
            "score_threshold": threshold,  # umbral para disparar alerta
        },
        # Selección de pesos del modelo elegida desde admin (vacío = usa config.yaml).
        # Solo aplica al backend torch; se aplica al reiniciar el nodo.
        "model_override": {},
        # Cuentas por defecto (cambiar al primer ingreso → is_default=True).
        "users": [
            make_user("admin", "admin", "admin"),
            make_user("usuario", "usuario", "user"),
        ],
    }


class Settings:
    """Almacén thread-safe de la config editable, respaldado por un JSON."""

    def __init__(self, path: str | Path = "./data/settings.json",
                 defaults: Optional[Dict[str, Any]] = None):
        # Ruta ABSOLUTA: así el archivo no depende del directorio desde el que se
        # lance el nodo (un CWD distinto escribía/leía otro settings.json y parecía
        # que los cambios —p. ej. la contraseña— "no se guardaban").
        self._path = Path(path).resolve()
        self._lock = threading.Lock()
        self._data = defaults or default_settings()
        loaded = False
        if self._path.exists():
            try:
                disk = json.loads(self._path.read_text(encoding="utf-8"))
                for k in _PERSIST_KEYS:
                    if k in disk:
                        self._data[k] = disk[k]
                loaded = True
            except (json.JSONDecodeError, OSError):
                log.warning("settings.json ilegible (%s); usando valores por defecto.",
                            self._path)
        # Garantiza que siempre existan cuentas (p. ej. archivo viejo sin users).
        if not self._data.get("users"):
            self._data["users"] = default_settings()["users"]
        self._flush()
        log.info("Configuración del dashboard en %s (%s, %d usuario(s)).",
                 self._path, "cargada del disco" if loaded else "nueva",
                 len(self._data.get("users", [])))

    # ── Config pública (sin secretos) ────────────────────────────────────────
    def _public(self) -> Dict[str, Any]:
        """Copia profunda SIN las cuentas (no exponer hashes por la API)."""
        d = json.loads(json.dumps(self._data))
        d.pop("users", None)
        return d

    def data(self) -> Dict[str, Any]:
        with self._lock:
            return self._public()

    def update(self, patch: Dict[str, Any]) -> Dict[str, Any]:
        """Mezcla un patch (solo claves web; nunca `users`) y persiste."""
        with self._lock:
            for k in _WEB_KEYS:
                if k in patch and patch[k] is not None:
                    self._data[k] = patch[k]
            self._flush()
            return self._public()

    # ── Cuentas / autenticación ──────────────────────────────────────────────
    def verify_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Devuelve {username, role, must_change} si las credenciales son válidas."""
        with self._lock:
            for u in self._data.get("users", []):
                if u.get("username") == username:
                    expected = u.get("hash", "")
                    got = _hash_pw(password, u.get("salt", "")) if u.get("salt") else ""
                    if expected and hmac.compare_digest(expected, got):
                        return {"username": username, "role": u.get("role", "user"),
                                "must_change": bool(u.get("is_default"))}
                    return None
        return None

    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{"username": u["username"], "role": u.get("role", "user"),
                     "is_default": bool(u.get("is_default"))}
                    for u in self._data.get("users", [])]

    def role_of(self, username: str) -> Optional[str]:
        with self._lock:
            for u in self._data.get("users", []):
                if u.get("username") == username:
                    return u.get("role", "user")
        return None

    def create_user(self, username: str, password: str, role: str) -> Dict[str, Any]:
        """Crea una cuenta nueva. Devuelve {ok, error}. Solo lo invoca el admin."""
        username = (username or "").strip()
        role = role if role in ("admin", "user") else "user"
        if not username:
            return {"ok": False, "error": "falta el nombre de usuario"}
        if len(password or "") < 4:
            return {"ok": False, "error": "contraseña mínimo 4 caracteres"}
        with self._lock:
            if any(u.get("username") == username for u in self._data.get("users", [])):
                return {"ok": False, "error": "ese usuario ya existe"}
            rec = make_user(username, password, role)
            rec["is_default"] = False  # creada a mano, no es cuenta por defecto
            self._data.setdefault("users", []).append(rec)
            self._flush()
        return {"ok": True}

    def delete_user(self, username: str, acting_user: str) -> Dict[str, Any]:
        """Borra una cuenta. No permite borrarse a uno mismo ni dejar el sistema
        sin ningún admin. Devuelve {ok, error}."""
        username = (username or "").strip()
        if username == acting_user:
            return {"ok": False, "error": "no puedes borrar tu propia cuenta"}
        with self._lock:
            users = self._data.get("users", [])
            victim = next((u for u in users if u.get("username") == username), None)
            if victim is None:
                return {"ok": False, "error": "usuario no existe"}
            admins = [u for u in users if u.get("role") == "admin"]
            if victim.get("role") == "admin" and len(admins) <= 1:
                return {"ok": False, "error": "debe quedar al menos un admin"}
            self._data["users"] = [u for u in users if u.get("username") != username]
            self._flush()
        return {"ok": True}

    def set_password(self, username: str, new_password: str) -> bool:
        if not new_password:
            return False
        with self._lock:
            for u in self._data.get("users", []):
                if u.get("username") == username:
                    u["salt"] = os.urandom(16).hex()
                    u["hash"] = _hash_pw(new_password, u["salt"])
                    u["is_default"] = False
                    self._flush()
                    return True
        return False

    # ── Selección de modelo (pesos) ──────────────────────────────────────────
    def model_override(self) -> Dict[str, str]:
        """Pesos elegidos desde admin: {detector, classifier} o {} si no hay."""
        with self._lock:
            ov = self._data.get("model_override") or {}
            return {k: ov[k] for k in ("detector", "classifier") if ov.get(k)}

    def set_model_override(self, detector: str, classifier: str) -> Dict[str, Any]:
        """Guarda los pesos elegidos (nombres de archivo). Se aplica al reiniciar."""
        with self._lock:
            self._data["model_override"] = {"detector": detector, "classifier": classifier}
            self._flush()
        return self.model_override()

    # ── Selección de familia/modelo (VAD vs heurístico) ──────────────────────
    def model_selection(self) -> Dict[str, str]:
        """Modelo activo elegido desde admin: {family, key}, o {} si no hay (cae al
        VAD configurado). Ortogonal a `model_override` (pesos del VAD torch)."""
        with self._lock:
            sel = self._data.get("model_selection") or {}
            return {k: sel[k] for k in ("family", "key") if sel.get(k)}

    def set_model_selection(self, family: str, key: str) -> Dict[str, Any]:
        """Guarda la familia/modelo activo. Se aplica al reiniciar el nodo."""
        with self._lock:
            self._data["model_selection"] = {"family": family, "key": key}
            self._flush()
        return self.model_selection()

    # ── Umbral por cámara ─────────────────────────────────────────────────────
    def set_camera_threshold(self, camera_id: str, thr: Optional[float]) -> Dict[str, Any]:
        """Fija (o quita, si thr is None) el umbral propio de una cámara. Persiste."""
        with self._lock:
            m = self._data.setdefault("camera_thresholds", {})
            if thr is None:
                m.pop(camera_id, None)
            else:
                m[camera_id] = float(thr)
            self._flush()
            return self._public()

    # ── Rotación por cámara ────────────────────────────────────────────────────
    def set_camera_rotation(self, camera_id: str, deg: Optional[int]) -> Dict[str, Any]:
        """Fija (o quita, si deg es 0/None) la rotación propia de una cámara. Persiste."""
        with self._lock:
            m = self._data.setdefault("camera_rotations", {})
            if not deg:
                m.pop(camera_id, None)
            else:
                m[camera_id] = int(deg)
            self._flush()
            return self._public()

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self._path)  # escritura atómica
