"""Preprocesado de clips para X3D en NumPy puro (sin torch).

Espejo exacto de `preprocess.build_clip_tensor`, pero devuelve un np.ndarray
float32 con shape [1, 3, T, H, W]. Se usa en el backend ONNX/QNN (RUBIK Pi 3),
donde NO queremos torch en tiempo de ejecución: onnxruntime consume numpy.

Mantener sincronizado con preprocess.build_clip_tensor (misma lógica de
muestreo, resize del lado corto, central crop y normalización mean/std).
"""
from __future__ import annotations

import cv2
import numpy as np

from .params import MEAN, MODEL_TRANSFORM_PARAMS, STD


def _resize_short_side(frame: np.ndarray, size: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _central_crop(frame: np.ndarray, target: int) -> np.ndarray:
    h, w = frame.shape[:2]
    top = (h - target) // 2
    left = (w - target) // 2
    return frame[top:top + target, left:left + target]


def sample_indices(n_raw: int, num_frames: int, sampling_rate: int) -> list[int]:
    return [min(i * sampling_rate, n_raw - 1) for i in range(num_frames)]


def build_clip_array(frames_rgb: np.ndarray, model_name: str) -> np.ndarray:
    """frames RGB (T_raw, H, W, 3) uint8 -> array float32 [1, 3, T, H, W] normalizado."""
    p = MODEL_TRANSFORM_PARAMS[model_name]
    crop_size, num_frames, sampling_rate = p["crop_size"], p["num_frames"], p["sampling_rate"]

    idxs = sample_indices(len(frames_rgb), num_frames, sampling_rate)
    selected = frames_rgb[idxs]  # (num_frames, H, W, 3)

    processed = []
    for f in selected:
        f = _resize_short_side(f, crop_size)
        f = _central_crop(f, crop_size)
        processed.append(f)
    clip = np.stack(processed).astype(np.float32) / 255.0   # (T, H, W, 3)

    clip = np.transpose(clip, (3, 0, 1, 2))[np.newaxis, ...]  # (1, 3, T, H, W)

    mean = np.array(MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    std = np.array(STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    clip = (clip - mean) / std
    return np.ascontiguousarray(clip, dtype=np.float32)
