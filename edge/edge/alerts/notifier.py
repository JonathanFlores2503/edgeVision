"""Notificador agnóstico al canal.

Lee destinatarios/opciones de settings.json en cada disparo (así toma cambios sin
reiniciar), formatea el mensaje y lo despacha a los canales configurados. Hoy:
Telegram. SMS/WhatsApp/correo se enchufan aquí mismo más adelante.

Es *best-effort*: el envío corre en un hilo aparte y nunca propaga excepciones al
pipeline. (La integración con el outbox offline-first queda como paso posterior.)
"""
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Optional

from .message import build_alert_text
from .telegram import send_telegram, send_telegram_photo

log = logging.getLogger(__name__)


class AlertNotifier:
    def __init__(self, settings):
        self.settings = settings  # edge.monitor.settings.Settings

    # ── API pública ──────────────────────────────────────────────────────────
    def notify_event(self, *, camera_id: str, class_name: str, score: float,
                     when: str, lat: Optional[float] = None,
                     lon: Optional[float] = None, image: Optional[bytes] = None) -> None:
        """Dispara el envío en segundo plano (no bloquea el pipeline).

        `image` (JPEG) opcional: si viene, la alerta de Telegram se manda como FOTO
        (el frame del momento, con las cajas del modelo) en vez de solo texto.

        Solo envía si el interruptor maestro ``alerts.enabled`` está activado;
        si no, no hace nada (las pruebas con send_test() sí envían siempre).
        """
        alerts = self.settings.data().get("alerts", {}) or {}
        if not alerts.get("enabled", False):
            log.info("Evento detectado, pero las alertas automáticas están DESACTIVADAS "
                     "(activa el switch en Config → Opciones y Guarda).")
            return  # envío automático desactivado
        threading.Thread(
            target=self._dispatch, daemon=True, name="alert",
            kwargs=dict(camera_id=camera_id, class_name=class_name, score=score,
                        when=when, lat=lat, lon=lon, image=image),
        ).start()

    def send_test(self, image: Optional[bytes] = None) -> dict:
        """Envía una alerta de prueba de forma síncrona y devuelve el resultado."""
        return self._dispatch(
            camera_id="cam-front", class_name="Prueba de alerta", score=0.99,
            when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            lat=13.6929, lon=-89.2182, image=image,
        )

    # ── Interno ──────────────────────────────────────────────────────────────
    def _dispatch(self, *, camera_id, class_name, score, when, lat, lon,
                  image: Optional[bytes] = None) -> dict:
        data = self.settings.data()
        alerts = data.get("alerts", {}) or {}
        text = build_alert_text(
            vehicle=data.get("vehicle"), camera_id=camera_id, class_name=class_name,
            score=score, when=when, lat=lat, lon=lon,
            include_location=alerts.get("include_location_link", True),
            map_provider=alerts.get("map_provider", "google"),
        )
        sent, errors = 0, []

        # ── Telegram ──
        # Los tokens no llevan espacios; quita cualquier espacio accidental al pegar.
        token = "".join((alerts.get("telegram_bot_token") or "").split())
        chats = [str(c).strip() for c in (alerts.get("telegram") or []) if str(c).strip()]
        if token and chats:
            for chat in chats:
                # Con imagen -> foto (el ratero/quienes roban); sin imagen -> texto.
                if image:
                    ok, detail = send_telegram_photo(token, chat, image, caption=text)
                    if not ok:  # si la foto falla (p. ej. imagen muy grande), cae a texto
                        ok, detail = send_telegram(token, chat, text)
                else:
                    ok, detail = send_telegram(token, chat, text)
                if ok:
                    sent += 1
                else:
                    errors.append(f"{chat}: {detail}")
        elif chats and not token:
            errors.append("telegram: falta el token del bot")

        # ── (futuro) SMS / WhatsApp / correo se agregan aquí ──

        if sent == 0 and not errors:
            errors.append("sin canal configurado (agrega token y chat id de Telegram)")

        log.info("Alerta '%s' en %s: %d enviada(s), %d error(es)",
                 class_name, camera_id, sent, len(errors))
        return {"text": text, "sent": sent, "errors": errors}
