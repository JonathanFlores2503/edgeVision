"""Wrapper de inferencia de TransLowNet (X3D + detector + clasificador).

Corre íntegramente en la Jetson. Carga:
  1. Extractor X3D (torch.hub, sin el último bloque) -> features 192-d
  2. Detector_VAD / Model_V3_Connection -> score de anomalía [0,1]
  3. violenceOneCrop -> distribución de clases
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from ..config import ModelCfg
from ..types import Clip, ClipResult
from . import preprocess
from .networks import Detector_VAD, Model_V3_Connection, violenceOneCrop

log = logging.getLogger(__name__)


def _strip_module_prefix(state_dict):
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


class TransLowNet:
    """Pipeline de inferencia completo, listo para multiplexar varias cámaras."""

    def __init__(self, cfg: ModelCfg):
        self.cfg = cfg
        self.device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
        if self.device != cfg.device:
            log.warning("CUDA no disponible; usando CPU (rendimiento reducido).")

        log.info("Cargando extractor X3D '%s' desde torch.hub...", cfg.name)
        fe = torch.hub.load("facebookresearch/pytorchvideo", cfg.name, pretrained=True)
        del fe.blocks[-1]  # quitamos la cabeza de clasificación -> features
        self.feature_extractor = fe.eval().to(self.device)

        weights_dir = Path(cfg.weights_dir) / cfg.name

        log.info("Cargando detector de anomalía...")
        if cfg.name == "x3d_l":
            self.detector = Detector_VAD(cfg.n_features)
        else:
            self.detector = Model_V3_Connection(cfg.n_features)
        self._load_weights(self.detector, weights_dir / cfg.weights_detector)
        self.detector = self.detector.eval().to(self.device)

        log.info("Cargando clasificador de clases...")
        self.classifier = violenceOneCrop(cfg.n_features, cfg.n_classes)
        self._load_weights(self.classifier, weights_dir / cfg.weights_classifier)
        self.classifier = self.classifier.eval().to(self.device)

        log.info("TransLowNet listo en %s.", self.device)

    def _load_weights(self, module: torch.nn.Module, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"No se encontraron pesos: {path}")
        state = torch.load(str(path), map_location=self.device)
        module.load_state_dict(_strip_module_prefix(state))

    @torch.no_grad()
    def extract_features(self, clip_tensor: torch.Tensor) -> torch.Tensor:
        """X3D -> features [1, C]. Promedia sobre T, H, W (igual que Propuesta2)."""
        feats = self.feature_extractor(clip_tensor)   # [1, C, T, H, W]
        feats = feats.squeeze(0).mean(dim=(1, 2, 3))  # [C]
        return feats.unsqueeze(0)                      # [1, C]

    @torch.no_grad()
    def infer(self, clip: Clip) -> ClipResult:
        """Ejecuta el pipeline completo sobre un Clip y devuelve un ClipResult."""
        t0 = time.perf_counter()
        clip_tensor = preprocess.build_clip_tensor(clip.frames, self.cfg.name, self.device)
        feats = self.extract_features(clip_tensor)

        score = float(self.detector(feats).squeeze().item())
        probs = self.classifier(feats).squeeze(0).cpu().numpy()
        class_id = int(np.argmax(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return ClipResult(
            camera_id=clip.camera_id,
            t_start=clip.t_start,
            t_end=clip.t_end,
            score=score,
            class_id=class_id,
            class_probs=[float(p) for p in probs],
            latency_ms=latency_ms,
        )

    def class_name(self, class_id: int) -> str:
        names = self.cfg.class_names
        return names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"
