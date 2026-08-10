# -*- coding: utf-8 -*-
"""
face_bubble.py
==============
Deteccion de rostro (MediaPipe) + "nube"/burbuja con el rostro ampliado
sobre el video. Compartido por AREA_REID y CONTADOR_PERSONAS.

Uso en un detector:

    from face_bubble import FaceBubble

    # en setup():
    self.faces = FaceBubble()          # una instancia POR fuente (no es
                                       # thread-safe entre fuentes)

    # en process_frame(), por cada persona trackeada:
    self.faces.update(tid, frame, bbox, frame_idx)   # refresca el rostro

    # en annotate(), por cada persona:
    self.faces.draw_bubble(out, tid, bbox)           # dibuja la nube

    # para guardar evidencia:
    crop = self.faces.get_face(tid)                  # BGR o None

    # limpieza de tracks muertos:
    self.faces.purge(active_tids)

El rostro se busca SOLO en la parte superior del bbox de la persona (donde
esta la cabeza) para que sea rapido y sin falsos positivos.
"""

import os
import os.path as osp
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_SCRIPT_DIR = osp.dirname(osp.realpath(__file__))
# Modelo BlazeFace de la API Tasks de MediaPipe (se descarga 1 vez, ~230KB)
_MODEL_PATH = osp.join(_SCRIPT_DIR, "weights_face",
                       "blaze_face_short_range.tflite")
_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
              "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")

# ═══════════════════════════════════════════════════════════════════════════
#  VARIABLES DE PRUEBA (debug) — modifica estas para afinar el rostro
# ═══════════════════════════════════════════════════════════════════════════
# Cada cuantos frames se re-detecta el rostro de un track.
#   ↓ mas bajo = rostro mas actualizado pero mas costo CPU
FACE_REFRESH = 10

# Fraccion SUPERIOR del bbox de la persona donde se busca la cabeza.
#   ↑ subir (0.6) si la camara esta muy inclinada y la cara queda mas abajo
HEAD_FRAC = 0.45

# Margen extra alrededor del rostro detectado (fraccion del rostro).
#   ↑ subir para que el crop incluya mas contexto (orejas, pelo)
FACE_MARGIN = 0.25

# Lado de la burbuja/nube en px sobre el frame nativo.
BUBBLE_SIZE = 110

# Confianza minima de MediaPipe (0-1).
#   ↓ bajar (0.2) si no encuentra rostros lejanos/borrosos (mas falsos)
#   ↑ subir (0.5) si marca cosas que no son caras
MIN_CONF = 0.35

# Resolucion MINIMA (alto en px) a la que se analiza la cabeza: si el crop
# es mas chico, se reescala a este alto antes de BlazeFace.
#   ↑ subir (128/160) si las caras lejanas no se detectan — mas costo
HEAD_MIN_H = 96

# Log de diagnostico: cada cuantas detecciones exitosas imprimir la
# resolucion real de analisis (0 = nunca)
DEBUG_LOG_EVERY = 50
# ═══════════════════════════════════════════════════════════════════════════


class FaceBubble:
    def __init__(self, bubble_size: int = BUBBLE_SIZE,
                 personas_dir: str = None):
        self.bubble_size = bubble_size
        self._det = None            # lazy: MediaPipe se crea al primer uso
        self._failed = False
        self._n_found = 0           # rostros encontrados (para el log diag)
        # tid -> {"face": img, "name": str|None, "score": float, "frame": n}
        self._cache: Dict[int, dict] = {}
        # Reconocimiento facial (SCRFD + AdaFace del face_runner) — si las
        # deps estan instaladas. Si no, cae a MediaPipe (sin nombres).
        try:
            import face_id
            face_id.ensure_ready(personas_dir or face_id.DEFAULT_PERSONAS_DIR)
            self._face_id = face_id if face_id.available() else None
        except Exception as e:
            print(f"[FaceBubble] face_id no disponible: {e}")
            self._face_id = None

    # ------------------------------------------------------------------ #
    def _detector(self):
        """MediaPipe Tasks API (la legacy mp.solutions ya no existe en 0.10.35+)."""
        if self._det is None and not self._failed:
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_py
                from mediapipe.tasks.python import vision

                if not osp.isfile(_MODEL_PATH):
                    os.makedirs(osp.dirname(_MODEL_PATH), exist_ok=True)
                    print(f"[FaceBubble] Descargando modelo BlazeFace "
                          f"(~230KB) → {_MODEL_PATH}")
                    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)

                self._mp = mp
                self._det = vision.FaceDetector.create_from_options(
                    vision.FaceDetectorOptions(
                        base_options=mp_py.BaseOptions(
                            model_asset_path=_MODEL_PATH),
                        min_detection_confidence=MIN_CONF))
                print("[FaceBubble] MediaPipe FaceDetector (Tasks) listo")
            except Exception as e:
                self._failed = True
                print(f"[FaceBubble] Sin MediaPipe ({e}) — burbujas desactivadas. "
                      f"Agrega 'mediapipe' al --with de uv run.")
        return self._det

    # ------------------------------------------------------------------ #
    def _detect_in_head(self, frame, bbox) -> Optional[dict]:
        """Busca el rostro en la parte superior del bbox de la persona.
        Retorna {"face", "name", "score"} o None.
        Con face_id (SCRFD+AdaFace) trae NOMBRE; con MediaPipe solo la cara."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame.shape[:2]
        hy2 = y1 + max(1, int((y2 - y1) * HEAD_FRAC))
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, hy2 = min(W, x2), min(H, hy2)
        head = frame[y1c:hy2, x1c:x2c]
        if head.size == 0 or head.shape[0] < 24 or head.shape[1] < 24:
            return None

        # ── Backend 1: face_runner (SCRFD + AdaFace) → cara + NOMBRE ──
        if self._face_id is not None:
            if head.shape[0] < HEAD_MIN_H:
                s = HEAD_MIN_H / head.shape[0]
                head = cv2.resize(head, (int(head.shape[1] * s), HEAD_MIN_H))
            r = self._face_id.analyze_head(head)
            if r is None:
                return None
            self._n_found += 1
            if DEBUG_LOG_EVERY and self._n_found % DEBUG_LOG_EVERY == 1:
                print(f"[FaceBubble] analisis(SCRFD): cabeza "
                      f"{head.shape[1]}x{head.shape[0]}px | rostro "
                      f"{r['crop'].shape[1]}x{r['crop'].shape[0]}px | "
                      f"{r['name'] or 'desconocido'} ({r['score']:.2f})")
            return {"face": r["crop"], "name": r["name"],
                    "score": r["score"]}

        # ── Backend 2 (fallback): MediaPipe — solo deteccion ──────────
        det = self._detector()
        if det is None:
            return None

        # MediaPipe rinde mejor con cabezas de tamano razonable
        raw_h, raw_w = head.shape[:2]
        if head.shape[0] < HEAD_MIN_H:
            scale = HEAD_MIN_H / head.shape[0]
            head = cv2.resize(head,
                              (int(head.shape[1] * scale), HEAD_MIN_H))

        rgb = np.ascontiguousarray(cv2.cvtColor(head, cv2.COLOR_BGR2RGB))
        mp_img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                                data=rgb)
        res = det.detect(mp_img)
        if not res.detections:
            return None
        # la deteccion con mayor score
        best = max(res.detections, key=lambda d: d.categories[0].score)
        bb = best.bounding_box                # px sobre 'head'
        hh, hw = head.shape[:2]
        fx1 = float(bb.origin_x)
        fy1 = float(bb.origin_y)
        fx2 = fx1 + float(bb.width)
        fy2 = fy1 + float(bb.height)
        mx = (fx2 - fx1) * FACE_MARGIN
        my = (fy2 - fy1) * FACE_MARGIN
        fx1, fy1 = max(0, int(fx1 - mx)), max(0, int(fy1 - my))
        fx2, fy2 = min(hw, int(fx2 + mx)), min(hh, int(fy2 + my))
        if fx2 - fx1 < 12 or fy2 - fy1 < 12:
            return None

        # Diagnostico periodico: resolucion real del analisis de rostro
        self._n_found += 1
        if DEBUG_LOG_EVERY and self._n_found % DEBUG_LOG_EVERY == 1:
            print(f"[FaceBubble] analisis(MediaPipe): cabeza {raw_w}x{raw_h}px "
                  f"→ BlazeFace {hw}x{hh}px | rostro {fx2 - fx1}x{fy2 - fy1}px "
                  f"(conf>={MIN_CONF}, refresh={FACE_REFRESH}f, "
                  f"head_frac={HEAD_FRAC})")
        return {"face": head[fy1:fy2, fx1:fx2].copy(),
                "name": None, "score": 0.0}

    # ------------------------------------------------------------------ #
    def update(self, tid: int, frame, bbox, frame_idx: int):
        """Refresca el rostro del track cada FACE_REFRESH frames."""
        c = self._cache.get(tid)
        if c is not None and frame_idx - c["frame"] < FACE_REFRESH:
            return
        r = self._detect_in_head(frame, bbox)
        if r is not None:
            # conservar el mejor nombre visto para este track
            if (c is not None and c.get("name")
                    and (not r["name"] or c["score"] > r["score"])):
                r["name"], r["score"] = c["name"], c["score"]
            r["frame"] = frame_idx
            self._cache[tid] = r
        elif c is not None:
            c["frame"] = frame_idx - FACE_REFRESH // 2   # reintenta pronto

    def get_face(self, tid: int) -> Optional[np.ndarray]:
        c = self._cache.get(tid)
        return c["face"] if c else None

    def get_name(self, tid: int) -> Tuple[Optional[str], float]:
        """Nombre reconocido (AdaFace) y score del track, o (None, 0)."""
        c = self._cache.get(tid)
        if c is None:
            return None, 0.0
        return c.get("name"), float(c.get("score", 0.0))

    def purge(self, active_tids: List[int]):
        for tid in list(self._cache):
            if tid not in active_tids:
                del self._cache[tid]

    # ------------------------------------------------------------------ #
    def draw_bubble(self, frame, tid: int, bbox,
                    color: Tuple[int, int, int] = (255, 255, 255)):
        """Dibuja la 'nube' (globo estilo comic) con el rostro ampliado,
        anclada a la cabeza de la persona."""
        face = self.get_face(tid)
        if face is None:
            return
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        s = self.bubble_size
        fimg = cv2.resize(face, (s, s))

        ax = (x1 + x2) // 2                 # ancla: arriba-centro del bbox
        ay = max(0, y1)
        bx = ax - s // 2                    # esquina de la burbuja
        by = ay - s - 26
        bx = max(4, min(W - s - 4, bx))
        by = max(58, min(H - s - 4, by))    # 58: no tapar el banner

        # cola del globo (triangulo hacia la cabeza)
        tip = (ax, min(ay, by + s + 24))
        base_y = by + s + 2
        pts = np.array([[bx + s // 2 - 12, base_y],
                        [bx + s // 2 + 12, base_y],
                        [tip[0], tip[1]]], dtype=np.int32)
        cv2.fillPoly(frame, [pts], color)

        # marco "nube": borde blanco redondeado + bumps
        cv2.rectangle(frame, (bx - 4, by - 4), (bx + s + 4, by + s + 4),
                      color, -1)
        r = 10
        for cxx in range(bx, bx + s + 1, s // 3):
            cv2.circle(frame, (cxx, by - 4), r, color, -1)
            cv2.circle(frame, (cxx, by + s + 4), r, color, -1)
        for cyy in range(by, by + s + 1, s // 3):
            cv2.circle(frame, (bx - 4, cyy), r, color, -1)
            cv2.circle(frame, (bx + s + 4, cyy), r, color, -1)

        frame[by:by + s, bx:bx + s] = fimg
        name, score = self.get_name(tid)
        label = f"{name} {score:.2f}" if name else f"id{tid}"
        cv2.rectangle(frame, (bx, by + s - 18), (bx + s, by + s),
                      (0, 0, 0), -1)
        cv2.putText(frame, label, (bx + 3, by + s - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (80, 255, 80) if name else (255, 255, 255), 1)
