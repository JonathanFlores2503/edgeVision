"""Formateo del mensaje de alerta (común a todos los canales)."""
from __future__ import annotations

from typing import Optional


def location_url(lat: Optional[float], lon: Optional[float],
                 provider: str = "google") -> Optional[str]:
    """Link a un mapa con la posición. None si no hay coordenadas."""
    if lat is None or lon is None:
        return None
    if provider == "osm":
        return (f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}"
                f"#map=17/{lat}/{lon}")
    return f"https://maps.google.com/?q={lat},{lon}"


def build_alert_text(*, vehicle: Optional[dict], camera_id: str, class_name: str,
                     score: float, when: str, lat: Optional[float] = None,
                     lon: Optional[float] = None, include_location: bool = True,
                     map_provider: str = "google") -> str:
    """Texto de la alerta. Las URLs se autoenlazan solas en Telegram/WhatsApp."""
    veh = vehicle or {}
    name = veh.get("plate") or veh.get("id") or "vehículo"
    route = veh.get("site") or ""
    head = f"Vehículo: {name}" + (f" ({route})" if route else "")
    lines = [
        "🚨 Alerta TransLowNet",
        head,
        f"Cámara: {camera_id}",
        f"Tipo: {class_name} · score {score:.2f}",
        f"Hora: {when}",
    ]
    if include_location:
        url = location_url(lat, lon, map_provider)
        lines.append(f"📍 Ubicación: {url}" if url else "📍 Ubicación: sin señal GPS")
    return "\n".join(lines)
