"""Backend de inferencia ONNX Runtime para RUBIK Pi 3 (Qualcomm).

Mismo contrato que `model.TransLowNet` (`infer(clip) -> ClipResult`,
`class_name(id)`), pero ejecuta los 3 ONNX exportados por `tools/export_onnx.py`
con onnxruntime. NO importa torch: el preprocesado es numpy puro
(`preprocess_np`) y la inferencia es onnxruntime.

Selección de Execution Provider (en orden de preferencia):
  1. QNNExecutionProvider  -> NPU Hexagon (requiere QNN SDK de Qualcomm +
     onnxruntime compilado con --use_qnn). Acelera de verdad en la placa.
  2. CPUExecutionProvider   -> fallback que funciona hoy sin SDK (Cortex-A78).

Así, esta misma clase corre ya en CPU y pasa a Hexagon en cuanto el QNN SDK
esté instalado, sin cambiar código (solo aparece el provider).
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import ModelCfg
from ..types import Clip, ClipResult
from . import preprocess_np

log = logging.getLogger(__name__)


def _build_providers() -> list:
    """Devuelve la lista de providers (con opciones) según lo disponible."""
    available = ort.get_available_providers()
    providers: list = []
    if "QNNExecutionProvider" in available:
        # backend_path -> librería del backend Hexagon (HTP). Configurable por env
        # para no hardcodear la ruta del QNN SDK.
        backend_path = os.environ.get("QNN_BACKEND_PATH", "libQnnHtp.so")
        providers.append((
            "QNNExecutionProvider",
            {"backend_path": backend_path},
        ))
        log.info("QNN EP disponible -> NPU Hexagon (backend_path=%s).", backend_path)
    else:
        log.warning(
            "QNNExecutionProvider NO disponible; usando CPU. Para Hexagon instala el "
            "QNN SDK de Qualcomm y un onnxruntime con --use_qnn."
        )
    providers.append("CPUExecutionProvider")
    return providers


class TransLowNetONNX:
    """Pipeline de inferencia TransLowNet sobre onnxruntime (CPU / QNN-Hexagon)."""

    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        onnx_dir = Path(cfg.onnx_dir) / cfg.name
        if not onnx_dir.exists():
            raise FileNotFoundError(
                f"No existe {onnx_dir}. Exporta primero con: "
                f"python -m tools.export_onnx --config <config.yaml> --out-dir {cfg.onnx_dir}"
            )

        providers = _build_providers()
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        def _session(fname: str) -> ort.InferenceSession:
            path = onnx_dir / fname
            if not path.exists():
                raise FileNotFoundError(f"Falta {path}. Re-exporta con tools.export_onnx.")
            return ort.InferenceSession(str(path), sess_options=so, providers=providers)

        log.info("Cargando ONNX desde %s ...", onnx_dir)
        self.fe = _session("x3d_features.onnx")
        self.detector = _session("detector.onnx")
        self.classifier = _session("classifier.onnx")

        self._fe_in = self.fe.get_inputs()[0].name
        self._det_in = self.detector.get_inputs()[0].name
        self._cls_in = self.classifier.get_inputs()[0].name

        log.info("TransLowNetONNX listo. Providers efectivos: %s", self.fe.get_providers())

    def extract_features(self, clip_arr: np.ndarray) -> np.ndarray:
        """X3D -> feature map 5D; promedia sobre (T',H',W') -> [1, C]."""
        feats_map = self.fe.run(None, {self._fe_in: clip_arr})[0]  # [1, C, T', H', W']
        return feats_map.mean(axis=(2, 3, 4)).astype(np.float32)   # [1, C]

    def infer(self, clip: Clip) -> ClipResult:
        t0 = time.perf_counter()
        clip_arr = preprocess_np.build_clip_array(clip.frames, self.cfg.name)
        feats = self.extract_features(clip_arr)

        score = float(np.asarray(self.detector.run(None, {self._det_in: feats})[0]).squeeze())
        probs = np.asarray(self.classifier.run(None, {self._cls_in: feats})[0]).squeeze()
        class_id = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ClipResult(
            camera_id=clip.camera_id,
            t_start=clip.t_start,
            t_end=clip.t_end,
            score=score,
            class_id=class_id,
            class_probs=[float(p) for p in np.atleast_1d(probs)],
            latency_ms=latency_ms,
        )

    def class_name(self, class_id: int) -> str:
        names = self.cfg.class_names
        return names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
