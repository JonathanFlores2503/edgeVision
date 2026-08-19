"""Exporta TransLowNet a ONNX para el backend ONNX/QNN (RUBIK Pi 3).

Genera 3 ficheros .onnx (uno por etapa del pipeline), que luego corre
`edge.inference.model_onnx.TransLowNetONNX` con onnxruntime (CPU EP hoy, QNN
EP / Hexagon cuando esté el SDK de Qualcomm):

  <out_dir>/<name>/
    ├── x3d_features.onnx   # X3D sin la cabeza  -> feature map 5D [1, C, T', H', W']
    ├── detector.onnx       # Detector_VAD / Model_V3_Connection -> score [1, 1]
    └── classifier.onnx     # violenceOneCrop -> probs [1, n_classes]

El mean sobre (T',H',W') que hace `TransLowNet.extract_features` NO se mete en el
grafo: lo aplica TransLowNetONNX en numpy (mantiene el ONNX del extractor simple
y portable a QNN).

Uso (one-time, en un equipo con torch — PC de dev o la propia placa en CPU):

    python -m tools.export_onnx --config config.yaml --out-dir ./data/onnx
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from edge.config import load_config
from edge.inference.model import _strip_module_prefix
from edge.inference.networks import Detector_VAD, Model_V3_Connection, violenceOneCrop
from edge.inference.params import MODEL_TRANSFORM_PARAMS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("export_onnx")

OPSET = 13


def _load_head(module: torch.nn.Module, path: Path) -> torch.nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"No se encontraron pesos: {path}")
    state = torch.load(str(path), map_location="cpu")
    module.load_state_dict(_strip_module_prefix(state))
    return module.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta TransLowNet a ONNX (X3D + cabezas).")
    ap.add_argument("--config", required=True, help="Ruta a config.yaml")
    ap.add_argument("--out-dir", default="./data/onnx", help="Carpeta de salida de los .onnx")
    args = ap.parse_args()

    cfg = load_config(args.config).model
    out_dir = Path(args.out_dir) / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    p = MODEL_TRANSFORM_PARAMS[cfg.name]
    crop, t = p["crop_size"], p["num_frames"]
    weights_dir = Path(cfg.weights_dir) / cfg.name

    # 1) Extractor X3D (torch.hub) sin la cabeza de clasificación.
    log.info("Cargando extractor X3D '%s'...", cfg.name)
    fe = torch.hub.load("facebookresearch/pytorchvideo", cfg.name, pretrained=True)
    del fe.blocks[-1]
    fe = fe.eval()

    dummy = torch.randn(1, 3, t, crop, crop)
    fe_path = out_dir / "x3d_features.onnx"
    log.info("Exportando %s (input [1,3,%d,%d,%d])...", fe_path.name, t, crop, crop)
    torch.onnx.export(
        fe, dummy, str(fe_path),
        input_names=["clip"], output_names=["features_map"],
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )

    # 2) Detector de anomalía.
    log.info("Exportando detector...")
    detector = Detector_VAD(cfg.n_features) if cfg.name == "x3d_l" \
        else Model_V3_Connection(cfg.n_features)
    detector = _load_head(detector, weights_dir / cfg.weights_detector)
    feats = torch.randn(1, cfg.n_features)
    torch.onnx.export(
        detector, feats, str(out_dir / "detector.onnx"),
        input_names=["features"], output_names=["score"],
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )

    # 3) Clasificador de clases.
    log.info("Exportando clasificador...")
    classifier = _load_head(violenceOneCrop(cfg.n_features, cfg.n_classes),
                            weights_dir / cfg.weights_classifier)
    torch.onnx.export(
        classifier, feats, str(out_dir / "classifier.onnx"),
        input_names=["features"], output_names=["probs"],
        opset_version=OPSET, do_constant_folding=True, dynamo=False,
    )

    log.info("Listo. ONNX en: %s", out_dir)
    log.info("Pon en config.yaml -> model.backend: onnx  y  model.onnx_dir: %s", out_dir)


if __name__ == "__main__":
    main()
