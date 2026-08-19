"""Preprocesado de clips para el extractor X3D.

Replica la lógica de `Propuesta2/inferenceFinal_Offline.py`
(muestreo, central crop y normalización mean/std) pero operando sobre frames
en memoria (ring buffer) en lugar de leer de disco.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
import torch

from .params import MEAN, MODEL_TRANSFORM_PARAMS, STD, clip_len_frames  # noqa: F401


def _resize_short_side(frame: np.ndarray, size: int) -> np.ndarray:
    """Redimensiona manteniendo aspect ratio para que el lado corto == size."""
    h, w = frame.shape[:2]
    scale = size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _central_crop(frame: np.ndarray, target: int) -> np.ndarray:
    h, w = frame.shape[:2]
    top = (h - target) // 2
    left = (w - target) // 2
    return frame[top:top + target, left:left + target]


def sample_indices(n_raw: int, num_frames: int, sampling_rate: int) -> List[int]:
    """Selecciona `num_frames` índices espaciados `sampling_rate` dentro de n_raw frames."""
    idxs = [min(i * sampling_rate, n_raw - 1) for i in range(num_frames)]
    return idxs


def build_clip_tensor(frames_rgb: np.ndarray, model_name: str, device: str) -> torch.Tensor:
    """Convierte frames RGB (T_raw, H, W, 3) uint8 en un tensor [1, 3, T, H, W] normalizado.

    Args:
        frames_rgb: frames crudos en RGB (uint8), tal como salen del ring buffer.
        model_name: x3d_l / x3d_m / x3d_s.
        device: cuda | cpu.
    """
    p = MODEL_TRANSFORM_PARAMS[model_name]
    crop_size, num_frames, sampling_rate = p["crop_size"], p["num_frames"], p["sampling_rate"]

    idxs = sample_indices(len(frames_rgb), num_frames, sampling_rate)
    selected = frames_rgb[idxs]  # (num_frames, H, W, 3)

    processed = []
    for f in selected:
        f = _resize_short_side(f, crop_size)
        f = _central_crop(f, crop_size)
        processed.append(f)
    clip = np.stack(processed)                       # (T, H, W, 3)

    clip = torch.from_numpy(clip).to(torch.float32) / 255.0
    clip = clip.permute(3, 0, 1, 2).unsqueeze(0)     # (1, 3, T, H, W)

    mean = torch.tensor(MEAN).view(1, 3, 1, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1, 1)
    clip = (clip - mean) / std
    return clip.to(device)
