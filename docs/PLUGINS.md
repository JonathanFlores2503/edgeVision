# Añadir un modelo (plugins)

La plataforma es **modelo-agnóstica**: todo el pipeline (captura, eventos, clips, storage,
dashboard, alertas) solo depende de un contrato de salida, **`ClipResult`**. Para añadir un
detector nuevo (caras, objetos, EPP, violencia, lo que sea) **no se toca la plataforma**: se
suelta un archivo `.py` en [`edge/heuristicModels/`](../edge/heuristicModels/) y aparece solo en
el panel de admin del dashboard, listo para seleccionar.

## El contrato de salida — `ClipResult`

Todo modelo produce, por cada decisión, un `ClipResult` (ver [`edge/edge/types.py`](../edge/edge/types.py)):

```python
ClipResult(
    camera_id="cam-1",
    t_start=..., t_end=...,      # datetime del tramo analizado
    score=0.92,                  # [0,1]; el motor de eventos dispara si score >= umbral
    class_id=1,                  # índice dentro de META["classes"]
    class_probs=[0.08, 0.92],    # probabilidad por clase
    latency_ms=12.4,
)
```

- **`score`** manda: el Event Engine abre/cierra evento con `score >= umbral` + histéresis.
  Para un detector "hay/no hay", usa `score = confianza` cuando detectas y `~0` cuando no.
- **`class_id` / `class_probs`** indexan tu lista `META["classes"]` (etiqueta mostrada en la UI).

## Dos formas de modelo (`kind`)

| `kind` | Cómo lo llama el nodo | Ideal para |
|---|---|---|
| `clip_scorer` | Le pasa un **clip ya muestreado** → devuelve **un** `ClipResult` | clasificación de vídeo, VAD, análisis por ventana |
| `stream_processor` | Le **empuja frames** uno a uno; tu modelo decide cuándo emitir | **detección de caras/objetos**, YOLO, tracking, máquinas de estado |

Para detección de caras/objetos casi siempre quieres **`stream_processor`**.

## Anatomía de un plugin

Un archivo en `edge/heuristicModels/mi_modelo.py` con **dos cosas**:

### 1) `META` — descriptor (literal puro)

El registro lo lee **por AST sin importar el módulo** (así el dashboard lista el modelo aunque
falten `torch`/`ultralytics`). Debe ser un diccionario literal, sin variables ni llamadas:

```python
META = {
    "key": "face_detector",              # id único
    "label": "Detección de caras",       # nombre en la UI
    "family": "heuristic",               # agrupación en el dashboard
    "kind": "stream_processor",          # o "clip_scorer"
    "classes": ["normal", "cara"],       # etiquetas; class_id indexa aquí
    "requires": ["cv2", "numpy"],        # deps a verificar (find_spec); si faltan -> deshabilitado
    "weights": [],                        # archivos de pesos esperados (informativo en la UI)
    "detail": "Detección facial por cámara con OpenCV.",
    "entry": "build",                     # nombre de la factory (default: "build")
}
```

### 2) `build(cfg)` — factory

Devuelve una instancia que cumple el protocolo del `kind` elegido. **Los imports pesados van
DENTRO de `build`/la clase**, nunca arriba del módulo (para que el listado sea barato).

## Ejemplo mínimo — `stream_processor` (estilo detección de caras)

```python
"""edge/heuristicModels/face_detector.py — detección de caras por cámara."""
from __future__ import annotations
import time
from datetime import datetime
from typing import Callable

META = {
    "key": "face_detector",
    "label": "Detección de caras",
    "family": "heuristic",
    "kind": "stream_processor",
    "classes": ["normal", "cara"],
    "requires": ["cv2", "numpy"],
    "weights": [],
    "detail": "Cuenta caras por frame con el clasificador Haar de OpenCV.",
    "entry": "build",
}


class FaceDetector:
    def __init__(self, cfg):
        import cv2  # import pesado DENTRO
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cv2 = cv2
        self._cascade = cv2.CascadeClassifier(path)
        self._on_result: Callable | None = None

    def start(self, on_result):
        self._on_result = on_result

    def feed(self, camera_id: str, frame_rgb, ts: datetime):
        from edge.types import ClipResult, utcnow
        t0 = time.perf_counter()
        gray = self._cv2.cvtColor(frame_rgb, self._cv2.COLOR_RGB2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.1, 5)
        hit = len(faces) > 0
        self._on_result(ClipResult(
            camera_id=camera_id, t_start=ts, t_end=utcnow(),
            score=1.0 if hit else 0.0,
            class_id=1 if hit else 0,
            class_probs=[0.0, 1.0] if hit else [1.0, 0.0],
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        ))

    def stop(self):
        pass

    def class_name(self, class_id: int) -> str:
        return META["classes"][class_id] if 0 <= class_id < len(META["classes"]) else "?"


def build(cfg):
    return FaceDetector(cfg)
```

> `feed()` debe ser **barato** (corre en el hilo de captura). Si tu modelo es pesado, encola el
> frame y procésalo en tus propios hilos; llama `on_result(...)` cuando tengas la decisión.
> Referencia completa multi-cámara con presupuesto de RAM: [`edge/heuristicModels/FightDetector_Production.py`](../edge/heuristicModels/FightDetector_Production.py).

## Ejemplo mínimo — `clip_scorer`

```python
class MiScorer:
    def __init__(self, cfg): ...
    def infer(self, clip):                 # clip.frames = (T,H,W,3) RGB uint8
        from edge.types import ClipResult
        return ClipResult(camera_id=clip.camera_id, t_start=clip.t_start,
                          t_end=clip.t_end, score=..., class_id=..., class_probs=[...],
                          latency_ms=...)
    def class_name(self, class_id): ...

def build(cfg):
    return MiScorer(cfg)
```

## Probarlo

1. Suelta tu archivo en `edge/heuristicModels/`.
2. Arranca el nodo: `cd edge && python -m edge.main --config config.yaml --mock`
   (el `--mock` evita necesitar los pesos del modelo integrado).
3. Abre el dashboard → **Config → Modelos enchufables**: tu modelo aparece.
   - Si sus `requires` no están instalados, sale **deshabilitado** con el motivo.
4. Selecciónalo y **reinicia el nodo** (el cambio de modelo se aplica al reiniciar).

## Reglas de oro

- `META` es un **literal puro** (se lee por AST). Nada de f-strings ni variables ahí.
- **Imports pesados dentro de `build`/la clase**, no en la cabecera del módulo.
- El `score` gobierna los eventos: mapea la confianza de tu modelo a `[0,1]`.
- Archivos que empiezan con `_` se ignoran (útil para helpers compartidos).
