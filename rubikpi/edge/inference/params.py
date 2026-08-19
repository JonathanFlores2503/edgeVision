"""Parámetros de transformación por variante de modelo (sin dependencias de torch).

Separado de preprocess.py para poder calcular tamaños de clip / leer parámetros
sin importar torch (útil en tests y en el simulador en modo mock).
"""
from __future__ import annotations

# Normalización Kinetics usada por X3D (idéntica a Propuesta2).
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]

# Parámetros por variante (de Propuesta2/model_transform_params).
MODEL_TRANSFORM_PARAMS = {
    "x3d_xs": {"crop_size": 182, "num_frames": 4, "sampling_rate": 12},
    "x3d_s": {"crop_size": 182, "num_frames": 13, "sampling_rate": 6},
    "x3d_m": {"crop_size": 256, "num_frames": 16, "sampling_rate": 5},
    "x3d_l": {"crop_size": 320, "num_frames": 16, "sampling_rate": 5},
}


def clip_len_frames(model_name: str) -> int:
    """Nº de frames crudos que se deben capturar para formar un clip."""
    p = MODEL_TRANSFORM_PARAMS[model_name]
    return p["num_frames"] * p["sampling_rate"]
