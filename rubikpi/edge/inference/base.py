"""Contratos de modelo del nodo edge.

El nodo soporta dos *familias* de modelo, unificadas por un único contrato de
salida (`ClipResult`, en `..types`), de modo que el resto del pipeline —eventos,
clips, alertas, telemetría, dashboard— es idéntico para ambas:

  • **clip_scorer** (familia VAD): recibe un `Clip` ya muestreado y devuelve un
    `ClipResult`. Lo cumplen `TransLowNet`, `TransLowNetONNX` y `MockModel`.

  • **stream_processor** (familia heurística): procesa el vídeo *frame a frame*
    con su propia máquina de estados (p. ej. YOLO+CLIP). No bloquea el hilo de
    captura: se le *empujan* frames y empuja `ClipResult` de vuelta cuando tiene
    una decisión.

`ModelSpec` describe un modelo seleccionable desde el panel de admin y sabe
construirlo bajo demanda (los imports/pesos pesados se difieren al `build`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:  # evita importar numpy/torch solo para los type hints
    import numpy as np

    from ..types import Clip, ClipResult


# ── Tipos de modelo (cómo lo despacha el runtime) ────────────────────────────
KIND_CLIP_SCORER = "clip_scorer"
KIND_STREAM_PROCESSOR = "stream_processor"

# ── Familias (cómo se agrupa en el dashboard) ────────────────────────────────
FAMILY_VAD = "vad"
FAMILY_HEURISTIC = "heuristic"


@runtime_checkable
class ClipScorer(Protocol):
    """Modelo VAD: un score de anomalía por clip muestreado."""

    def infer(self, clip: "Clip") -> "ClipResult": ...

    def class_name(self, class_id: int) -> str: ...


@runtime_checkable
class StreamModel(Protocol):
    """Modelo heurístico streaming.

    El nodo llama `start(on_result)` una vez, luego `feed(...)` por cada frame
    capturado (barato; el trabajo pesado corre en hilos internos del modelo), y
    `stop()` al apagar. El modelo invoca `on_result(ClipResult)` cuando confirma
    o cambia de estado.
    """

    def start(self, on_result: Callable[["ClipResult"], None]) -> None: ...

    def feed(self, camera_id: str, frame_rgb: "np.ndarray", ts: datetime) -> None: ...

    def stop(self) -> None: ...

    def class_name(self, class_id: int) -> str: ...


@dataclass
class ModelSpec:
    """Descriptor de un modelo seleccionable desde admin.

    `build(cfg, settings)` construye la instancia (difiriendo deps/pesos pesados).
    `available` indica si el entorno tiene lo necesario para ejecutarlo (deps +
    pesos); si es False, el dashboard lo muestra pero deshabilitado.
    """

    key: str
    label: str
    family: str                    # FAMILY_VAD | FAMILY_HEURISTIC
    kind: str                      # KIND_CLIP_SCORER | KIND_STREAM_PROCESSOR
    build: Callable[..., Any]
    classes: List[str] = field(default_factory=list)
    detail: str = ""
    available: bool = True
    unavailable_reason: str = ""
    # Extra opcional para la UI (p. ej. pesos disponibles del VAD torch).
    extra: dict = field(default_factory=dict)

    def to_public(self) -> dict:
        """Forma serializable para el panel de admin (sin el callable `build`)."""
        return {
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "kind": self.kind,
            "classes": list(self.classes),
            "detail": self.detail,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "extra": dict(self.extra),
        }
