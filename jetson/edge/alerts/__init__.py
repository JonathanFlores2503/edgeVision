"""Envío de alertas multicanal (Telegram, y luego SMS/WhatsApp/correo).

Diseño agnóstico al canal: un *formateador* de mensaje (igual para todos) y
*enviadores* enchufables. Los destinatarios y el token viven en settings.json
(editable desde la pestaña Configuración de la web).
"""
from .notifier import AlertNotifier

__all__ = ["AlertNotifier"]
