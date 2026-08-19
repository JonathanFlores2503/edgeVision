"""Marca del producto — ÚNICA fuente de verdad (white-label).

Cambia SOLO este archivo para rebrandizar toda la plataforma (dashboard web,
PWA, títulos, telemetría). El nodo sirve estos valores al cargar la interfaz:
`index.html` usa tokens `{{BRAND}}`, `{{BRAND_SHORT}}`, `{{BRAND_TAGLINE}}` y
`{{BRAND_VERSION}}` que `monitor/server.py` sustituye al vuelo.

Ejemplo para un cliente concreto:
    PRODUCT_NAME    = "Acme Vision"
    PRODUCT_SHORT   = "Acme"
    PRODUCT_TAGLINE = "Videovigilancia inteligente"
"""
from __future__ import annotations

# ── Identidad del producto ───────────────────────────────────────────────────
PRODUCT_NAME = "EdgeVision"                       # nombre completo (títulos, PWA)
PRODUCT_SHORT = "EdgeVision"                      # nombre corto (icono PWA, sidebar)
PRODUCT_TAGLINE = "Plataforma de detección en el borde"  # subtítulo/eslogan
PRODUCT_DESCRIPTION = "Detección en vídeo on-device, multi-cámara y offline-first"
PRODUCT_VERSION = "v0.1"

# ── Estética (dark SOC por defecto) ──────────────────────────────────────────
THEME_COLOR = "#0f1115"                            # color de barra del navegador / PWA

# ── Tokens que server.py sustituye en index.html ─────────────────────────────
def tokens() -> dict:
    """Mapa {token: valor} para inyectar en la interfaz servida."""
    return {
        "{{BRAND}}": PRODUCT_NAME,
        "{{BRAND_SHORT}}": PRODUCT_SHORT,
        "{{BRAND_TAGLINE}}": PRODUCT_TAGLINE,
        "{{BRAND_VERSION}}": PRODUCT_VERSION,
    }
