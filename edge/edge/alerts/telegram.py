"""Enviador por Telegram (Bot API) — solo librería estándar, sin dependencias.

Requiere internet (4G en el vehículo). Crear el bot con @BotFather para obtener el
token; el chat id se obtiene escribiéndole al bot y consultando getUpdates, o con
@userinfobot. Tanto el token como los chat ids se guardan en settings.json.
"""
from __future__ import annotations

import json
import logging
import secrets
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/sendMessage"
_PHOTO = "https://api.telegram.org/bot{token}/sendPhoto"
_UPDATES = "https://api.telegram.org/bot{token}/getUpdates"

# Telegram limita el pie de foto (caption) a 1024 caracteres.
_CAPTION_MAX = 1024


def clean_token(token: str) -> str:
    """Quita espacios accidentales al pegar (los tokens no llevan espacios)."""
    return "".join((token or "").split())


def get_chat_ids(token: str, timeout: float = 10.0) -> tuple[bool, list, str]:
    """Lee getUpdates y devuelve (ok, [{id,name}], detalle).

    Encuentra los chats que YA le escribieron al bot. El usuario debe haberle
    mandado al menos un mensaje (o pulsado Start) para aparecer aquí.
    """
    token = clean_token(token)
    try:
        with urllib.request.urlopen(_UPDATES.format(token=token), timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            return False, [], body.get("description", "error")
        chats: dict = {}
        for upd in body.get("result", []):
            msg = (upd.get("message") or upd.get("edited_message")
                   or upd.get("channel_post") or {})
            chat = msg.get("chat") or {}
            if "id" in chat:
                name = (chat.get("title")
                        or " ".join(filter(None, [chat.get("first_name"), chat.get("last_name")]))
                        or chat.get("username") or "")
                chats[chat["id"]] = name
        return True, [{"id": str(k), "name": v} for k, v in chats.items()], ""
    except urllib.error.HTTPError as e:
        try:
            desc = json.loads(e.read().decode("utf-8")).get("description", str(e))
        except Exception:  # noqa: BLE001
            desc = f"HTTP {e.code}"
        return False, [], desc
    except Exception as e:  # noqa: BLE001
        return False, [], str(e)


def send_telegram(token: str, chat_id: str, text: str,
                  timeout: float = 10.0) -> tuple[bool, str]:
    """Envía un mensaje. Devuelve (ok, detalle). `detalle` trae el motivo si falla."""
    data = urllib.parse.urlencode({
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": "true",
    }).encode()
    try:
        req = urllib.request.Request(_API.format(token=token), data=data)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if not body.get("ok"):
            desc = body.get("description", "rechazado")
            log.warning("Telegram rechazó el envío a %s: %s", chat_id, desc)
            return False, desc
        return True, ""
    except urllib.error.HTTPError as e:  # 401 token malo, 400 chat malo, etc.
        try:
            desc = json.loads(e.read().decode("utf-8")).get("description", str(e))
        except Exception:  # noqa: BLE001
            desc = f"HTTP {e.code}"
        log.warning("Telegram HTTP %s a %s: %s", e.code, chat_id, desc)
        return False, desc
    except Exception as e:  # noqa: BLE001 — best-effort, nunca tumba el pipeline
        log.warning("No se pudo enviar a Telegram (%s): %s", chat_id, e)
        return False, str(e)


def _multipart(fields: dict, photo_name: str, photo_bytes: bytes) -> tuple[bytes, str]:
    """Arma un cuerpo multipart/form-data (solo stdlib) con campos de texto + 1 foto."""
    boundary = "----vad" + secrets.token_hex(16)
    crlf = b"\r\n"
    body = bytearray()
    for k, v in fields.items():
        body += b"--" + boundary.encode() + crlf
        body += f'Content-Disposition: form-data; name="{k}"'.encode() + crlf + crlf
        body += str(v).encode("utf-8") + crlf
    body += b"--" + boundary.encode() + crlf
    body += (f'Content-Disposition: form-data; name="photo"; filename="{photo_name}"'
             .encode() + crlf)
    body += b"Content-Type: image/jpeg" + crlf + crlf
    body += photo_bytes + crlf
    body += b"--" + boundary.encode() + b"--" + crlf
    return bytes(body), boundary


def send_telegram_photo(token: str, chat_id: str, photo_bytes: bytes,
                        caption: str = "", timeout: float = 20.0) -> tuple[bool, str]:
    """Envía una FOTO (JPEG) con pie de foto vía sendPhoto. Devuelve (ok, detalle)."""
    token = clean_token(token)
    fields = {"chat_id": chat_id}
    if caption:
        fields["caption"] = caption[:_CAPTION_MAX]
    body, boundary = _multipart(fields, "evento.jpg", photo_bytes)
    try:
        req = urllib.request.Request(_PHOTO.format(token=token), data=body)
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resbody = json.loads(resp.read().decode("utf-8"))
        if not resbody.get("ok"):
            desc = resbody.get("description", "rechazado")
            log.warning("Telegram (foto) rechazó a %s: %s", chat_id, desc)
            return False, desc
        return True, ""
    except urllib.error.HTTPError as e:
        try:
            desc = json.loads(e.read().decode("utf-8")).get("description", str(e))
        except Exception:  # noqa: BLE001
            desc = f"HTTP {e.code}"
        log.warning("Telegram (foto) HTTP %s a %s: %s", e.code, chat_id, desc)
        return False, desc
    except Exception as e:  # noqa: BLE001 — best-effort
        log.warning("No se pudo enviar foto a Telegram (%s): %s", chat_id, e)
        return False, str(e)
