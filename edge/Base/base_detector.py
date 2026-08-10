# -*- coding: utf-8 -*-
"""Shim mínimo de `BaseDetector` para edgeVision.

El framework original **Base** (de ArconteDetection) aportaba el servidor web,
la gestión de fuentes RTSP, el bucle de captura y el streaming MJPEG. En
edgeVision **ese rol ya lo cumple la plataforma** (CameraManager + monitor), así
que aquí solo dejamos lo único que los detectores realmente usan de la clase
base: guardar `self.args` en `configure()`. El resto lo define cada detector.

Un detector portado (p. ej. `PeopleCounter_V2_Web.PeopleCounterV2Detector`) se
maneja desde un adaptador `stream_processor` en `heuristicModels/` que hace de
puente entre la captura de edgeVision y `process_frame()`.
"""
from __future__ import annotations


class BaseDetector:
    """Base mínima: solo el contrato que consumen los detectores portados."""

    NAME = "detector"

    def add_arguments(self, parser):
        """Los detectores registran sus flags aquí (opcional)."""
        return None

    def configure(self, args):
        """Guarda la config de arranque. Los detectores llaman super().configure()."""
        self.args = args

    def setup(self):
        """Carga de pesos/estado (lo implementa el detector)."""
        return None

    def process_frame(self, frame, frame_idx):
        """Procesa un frame BGR y devuelve un dict-resultado (lo implementa el detector)."""
        raise NotImplementedError

    def annotate(self, frame, result, stats):
        """Devuelve el frame anotado para la vista en vivo (opcional)."""
        return frame

    def draw_infobar(self, frame, stats, extra=""):
        """Barra genérica de telemetría (FPS/modelo) que dibujaba el framework
        Base de Arconte. En edgeVision **la plataforma** ya superpone FPS/latencia/
        costo por cámara (MonitorState), así que aquí es un no-op: solo evita que
        un detector portado que la invoque reviente. No modifica el frame."""
        return frame

    def teardown(self):
        """Libera recursos (opcional)."""
        return None
