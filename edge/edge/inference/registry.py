"""Registro de modelos: descubre y construye los modelos seleccionables.

Dos familias (ver `base.py`):
  • VAD (built-in): el modelo configurado en `config.yaml` (torch u ONNX) más el
    scorer `mock`. La selección de pesos detector/clasificador del VAD torch se
    sigue manejando aparte (`settings.model_override`), sin cambios.
  • Heurística (descubierta): cada archivo `heuristicModels/*.py` que exponga un
    diccionario `META` y una factory `build(cfg) -> StreamModel`.

El listado para el dashboard (`list_models`) NO ejecuta deps pesadas: lee `META`
(import ligero, con respaldo por AST si el import falla por falta de libs) y
calcula `available` mirando si las dependencias declaradas están instaladas.
La construcción real (`build`) sí difiere los imports/pesos pesados.
"""
from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import base
from .base import (
    FAMILY_HEURISTIC,
    FAMILY_VAD,
    KIND_CLIP_SCORER,
    KIND_STREAM_PROCESSOR,
    ModelSpec,
)

log = logging.getLogger(__name__)

# Carpeta donde el usuario sube modelos heurísticos (raíz del repo, junto a `edge/`).
HEUR_DIR = Path(__file__).resolve().parents[2] / "heuristicModels"


# ── Disponibilidad de dependencias (sin ejecutarlas) ─────────────────────────
def _missing_deps(requires: List[str]) -> List[str]:
    missing = []
    for mod in requires or []:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError, ModuleNotFoundError):
            missing.append(mod)
    return missing


# ── Familia VAD (built-in) ───────────────────────────────────────────────────
# Nota: estas funciones reciben `model_cfg` = config.ModelCfg (lo que tiene tanto
# main.py como el MonitorServer), no el EdgeConfig completo.
def _vad_specs(model_cfg=None) -> List[ModelSpec]:
    backend = getattr(model_cfg, "backend", "torch") or "torch"
    name = getattr(model_cfg, "name", "") or ""
    classes = list(getattr(model_cfg, "class_names", []) or [])

    if backend == "onnx":
        miss = _missing_deps(["onnxruntime", "numpy"])
        vad = ModelSpec(
            key="vad", label=f"Detección en vídeo · {name or 'x3d'} (ONNX)", family=FAMILY_VAD,
            kind=KIND_CLIP_SCORER, build=_build_vad, classes=classes,
            detail="Modelo de detección en vídeo exportado a ONNX (onnxruntime, QNN/CPU).",
            available=not miss,
            unavailable_reason=("faltan deps: " + ", ".join(miss)) if miss else "",
            extra={"backend": "onnx"},
        )
    else:
        miss = _missing_deps(["torch", "pytorchvideo", "numpy"])
        vad = ModelSpec(
            key="vad", label=f"Detección en vídeo · {name or 'x3d'} (torch)", family=FAMILY_VAD,
            kind=KIND_CLIP_SCORER, build=_build_vad, classes=classes,
            detail="Detección en vídeo: X3D + detector + clasificador (torch).",
            available=not miss,
            unavailable_reason=("faltan deps: " + ", ".join(miss)) if miss else "",
            extra={"backend": "torch"},
        )

    mock = ModelSpec(
        key="mock", label="Mock (scorer simulado)", family=FAMILY_VAD,
        kind=KIND_CLIP_SCORER, build=_build_mock,
        detail="Sin torch ni pesos; para ver cámaras/UI sin GPU.",
        available=True,
    )
    return [vad, mock]


def _build_vad(model_cfg, settings=None):
    if getattr(model_cfg, "backend", "torch") == "onnx":
        from .model_onnx import TransLowNetONNX
        return TransLowNetONNX(model_cfg)
    from .model import TransLowNet
    return TransLowNet(model_cfg)


def _build_mock(model_cfg, settings=None):
    from .mock import MockModel
    return MockModel(model_cfg)


# ── Familia heurística (descubierta en heuristicModels/) ─────────────────────
def _read_meta_via_ast(path: Path) -> Optional[dict]:
    """Lee el literal `META = {...}` sin ejecutar el módulo (respaldo cuando el
    import falla por deps faltantes)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, OSError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "META":
                    try:
                        return ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        return None
    return None


def _import_module_from_path(path: Path):
    mod_name = f"heuristicModels.{path.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"no se pudo cargar el spec de {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # puede fallar si faltan deps pesadas
    return module


def _spec_from_meta(path: Path, meta: dict) -> Optional[ModelSpec]:
    key = meta.get("key") or path.stem
    requires = list(meta.get("requires") or [])
    miss = _missing_deps(requires)
    detail = meta.get("detail", "")
    weights = meta.get("weights") or []
    if weights:
        detail = (detail + f"  Pesos: {', '.join(weights)}").strip()
    entry = meta.get("entry", "build")

    def _build(cfg, settings=None, _path=path, _entry=entry):
        module = _import_module_from_path(_path)
        factory = getattr(module, _entry, None)
        if factory is None:
            raise AttributeError(f"{_path.name}: falta la factory '{_entry}(cfg)'")
        return factory(cfg)

    return ModelSpec(
        key=key,
        label=meta.get("label", key),
        family=FAMILY_HEURISTIC,
        kind=meta.get("kind", KIND_STREAM_PROCESSOR),
        build=_build,
        classes=list(meta.get("classes") or []),
        detail=detail,
        available=not miss,
        unavailable_reason=("faltan deps: " + ", ".join(miss)) if miss else "",
        extra={"requires": requires, "weights": weights, "file": path.name},
    )


def discover_heuristics() -> Dict[str, ModelSpec]:
    """Escanea `heuristicModels/*.py` y devuelve {key: ModelSpec}.

    El listado NO importa los módulos (que arrastran torch/ultralytics/clip): lee
    el literal `META` por AST. El módulo solo se importa al construir (`build`).
    """
    out: Dict[str, ModelSpec] = {}
    if not HEUR_DIR.exists():
        return out
    for path in sorted(HEUR_DIR.glob("*.py")):
        if path.name.startswith("_"):
            continue
        meta = _read_meta_via_ast(path)
        if not isinstance(meta, dict) or not (meta.get("key") or meta.get("label")):
            log.debug("Heurístico ignorado (sin META literal válido): %s", path.name)
            continue
        spec = _spec_from_meta(path, meta)
        if spec is not None:
            out[spec.key] = spec
    return out


# ── API pública ──────────────────────────────────────────────────────────────
def list_models(model_cfg=None) -> Dict[str, List[dict]]:
    """Listado serializable para el panel de admin, agrupado por familia."""
    return {
        FAMILY_VAD: [s.to_public() for s in _vad_specs(model_cfg)],
        FAMILY_HEURISTIC: [s.to_public() for s in discover_heuristics().values()],
    }


def _all_specs(model_cfg=None) -> Dict[Tuple[str, str], ModelSpec]:
    specs: Dict[Tuple[str, str], ModelSpec] = {}
    for s in _vad_specs(model_cfg):
        specs[(s.family, s.key)] = s
    for s in discover_heuristics().values():
        specs[(s.family, s.key)] = s
    return specs


def resolve(selection: Optional[dict], model_cfg=None) -> ModelSpec:
    """Devuelve el ModelSpec activo a partir de la selección guardada en settings.
    Si no hay selección válida, cae al VAD configurado (primer spec VAD)."""
    vad_default = _vad_specs(model_cfg)[0]
    if not selection:
        return vad_default
    family = selection.get("family") or FAMILY_VAD
    key = selection.get("key") or vad_default.key
    spec = _all_specs(model_cfg).get((family, key))
    return spec or vad_default


def build(selection: Optional[dict], model_cfg=None, settings=None) -> Tuple[Any, str]:
    """Construye el modelo activo. Devuelve (modelo, kind)."""
    spec = resolve(selection, model_cfg)
    model = spec.build(model_cfg, settings)
    return model, spec.kind
