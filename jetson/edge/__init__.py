"""Plataforma edge (Jetson) para detección de anomalías en vídeo (VAD).

Ejecuta el detector TransLowNet 100% on-device sobre hasta 4 cámaras RTSP,
genera eventos de anomalía y los comunica al cloud (MQTT + subida de clips).
"""

__version__ = "0.1.0"
