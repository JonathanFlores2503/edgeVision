"""
FightDetector_Production.py
===========================
Versión producción del detector de peleas.

Basado en FightDetector_Debug_v6.py — misma lógica y pipeline, display mínimo.

Display: video original + overlays:
  - Banner superior (50px): PELEA DETECTADA / NORMAL con scores de zona primaria
  - ROI rectangles por zona activa (rojo=confirmada, naranja=activa, cian=tibia)
  - Info bar inferior (30px): FPS, modelo, YOLO ms, CLIP ms, VRAM

Console: activaciones por batch CLIP (zona, ema, raw, consec).

Alertas:
  - ALERTAS_FIGHT/FIGHT_z<id>_<ts>.jpg  — frame al momento de confirmación
  - FIGHT_PROD_LIVE.jpg  — último frame procesado (para monitoreo remoto)

Uso:
  python FightDetector_Production.py <video_path>
  python FightDetector_Production.py rtsp://...
  python FightDetector_Production.py /media/pc/MainWork2/Codes/ArconteDetection_DebugTools/JonathanTools/Figthing/Videos/Pelea003.mp4 --clip-model ViT-B/32 --frame-skip 1 --save
  python FightDetector_Production.py <video_path> --frame-skip 1
  python FightDetector_Production.py <video_path> --resize 1280x720 --save
  python FightDetector_Production.py rtsp://... --no-display --no-summary
"""

import cv2
import numpy as np
from ultralytics import YOLO
import json
import time
import os
import sys
import threading
import queue
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import torch
import clip

# =============================================================================
# META — descriptor para el registro modular del nodo edge.
# Debe ser un literal puro: el registro lo lee por AST sin importar este módulo
# (que arrastra torch/ultralytics/clip). Ver edge/inference/registry.py.
# =============================================================================
META = {
    "key": "fight_detector",
    "label": "Fight Detector (YOLO + CLIP)",
    "family": "heuristic",
    "kind": "stream_processor",
    "classes": ["normal", "fight"],
    "requires": ["torch", "ultralytics", "clip"],
    "weights": ["yolo11n.pt"],
    "yolo_model": "yolo11n.pt",  # nano: mucho más rápido en CPU/edge (opciones: yolo11n/s/m/l/x .pt)
    "clip_model": "ViT-B/32",    # más ligero/rápido que ViT-L/14 (opciones: ViT-B/32, ViT-B/16, ViT-L/14)
    "mem_budget_mb": 4700,       # tope de RAM del proceso; no crea más detectores si se pasaría
    "infer_every_n": 1,          # 1 = analiza todos los frames; 2 = uno de cada dos, etc.
                                 # Súbelo para dejar GPU libre con varias cámaras.
    "detail": "Multi-cámara: un YOLO11n por cámara + UN CLIP ViT-B/32 fp16 compartido, "
              "con juez batcheado. Umbrales calibrables (heuristicModels/fight_params.json).",
    "entry": "build",
}

# =============================================================================
# CONFIG
# =============================================================================
clip_device  = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_SIZE = (960, 540)   # resolución máxima de display; fuentes más grandes se reducen

CLASS_PERSON = 0

# ── Heurística espacial ──
F_DIST_MAX_LIMIT   = 300
F_MIN_IOU_CRITICAL = 0.01
F_STICKY_FRAMES    = 25
F_PAD_PIXELS       = 100
F_PAD_CLOSE        = 160

# ── Gate de movimiento ──
F_MIN_MOVEMENT_FOR_CLIP = 2.0

# ── CLIP buffer y umbrales ──
F_BUFFER_SIZE   = 6
F_SET_SCORE_TH  = 3.8
F_CONSEC_MIN    = 3
F_SMOOTH_FACTOR = 0.70
F_FAST_TRACK_TH = 6.0

# ── KSM ──
KSM_SELECT = 6

# ── Mosaico temporal ──
CLIP_FRAME_SIZE = (128, 128)
MOSAIC_COLS     = 3
MOSAIC_ROWS     = 2
# Cómo entra el mosaico 3×2 en el cuadro cuadrado de CLIP:
#   "squash" → el mosaico completo, deformando un poco cada celda (por defecto)
#   "crop"   → como el `preprocess` original de CLIP, que recorta el centro y
#              descarta los bordes de las columnas laterales
MOSAIC_FIT      = "squash"

# ── Multitud ──
CROWD_PERSON_TH = 5

# ── Multi-ROI ──
ZONE_MAX            = 3
ZONE_IDLE_MAX       = 45
ZONE_MIN_PAIR_SCORE = 28

# ── Estabilizador de zona ──
STAB_EMA_ALPHA   = 0.25
STAB_IOU_TH      = 0.05
STAB_PENDING_MAX = 4

# ── Profiling ──
_TIMING_ALPHA = 0.15

# ── Clip size usado como input de display a CLIP (no afecta display real) ──
_CLIP_VIEW_SIZE = (640, 480)

# ── Ancho de referencia de la calibración espacial ───────────────────────────
# F_DIST_MAX_LIMIT / F_PAD_* están en píxeles ABSOLUTOS, medidos sobre una cámara
# de 1280 de ancho. En el nodo edge los frames llegan reescalados (capture.
# proc_short_side=360 → ~640 de ancho), así que 300 px cubrirían media pantalla y
# se formarían pares entre personas que no están juntas. El detector reescala
# estos tres umbrales al ancho real del frame (ver `_fit_scale`).
REF_FRAME_WIDTH = 1280

# =============================================================================
# PARÁMETROS CALIBRABLES
# =============================================================================
# Los umbrales del juez CLIP dependen del backbone: `raw` es una diferencia de
# logits, y su escala cambia con el modelo (los 3.8 / 6.0 de fábrica se midieron
# con ViT-L/14). Cambiar de backbone sin recalibrar deja el detector mudo o
# disparando de más, así que los umbrales se leen de un JSON que escribe
# `tools/calibrate_fight.py` con medidas sobre vídeos reales.
_PARAM_KEYS = (
    "F_SET_SCORE_TH", "F_FAST_TRACK_TH", "F_CONSEC_MIN", "F_SMOOTH_FACTOR",
    "F_MIN_MOVEMENT_FOR_CLIP", "ZONE_MIN_PAIR_SCORE", "F_DIST_MAX_LIMIT",
    "F_PAD_PIXELS", "F_PAD_CLOSE", "ZONE_MAX", "ZONE_IDLE_MAX", "CROWD_PERSON_TH",
    "MOSAIC_FIT",
)
PARAMS_FILE = Path(__file__).resolve().parent / "fight_params.json"


def load_params(path: Optional[Path] = None) -> dict:
    """Aplica los parámetros calibrados de `fight_params.json` si existe.

    Sobrescribe las constantes del módulo (que es lo que lee todo el pipeline),
    ignorando claves desconocidas: un JSON con basura no debe cambiar el
    comportamiento en silencio. Devuelve lo aplicado."""
    p = Path(path) if path is not None else PARAMS_FILE
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"[FIGHT] {p.name} ilegible ({e}); uso los umbrales de fábrica.")
        return {}
    applied = {}
    for k in _PARAM_KEYS:
        if k in raw:
            try:
                cur = globals()[k]
                applied[k] = type(cur)(raw[k]) if isinstance(cur, (int, float)) else raw[k]
            except (TypeError, ValueError):
                continue
    if applied:
        globals().update(applied)
        print(f"[FIGHT] Umbrales calibrados desde {p.name}: " +
              ", ".join(f"{k}={v}" for k, v in applied.items()))
    return applied


_EDGE_DIR = Path(__file__).resolve().parents[1]          # .../jetson
_CF_DIR = _EDGE_DIR / "Models" / "ContadorFlujo"


def _resolve_yolo(name: str) -> str:
    """Ubica los pesos YOLO en el árbol del nodo: jetson/<name> →
    Models/ContadorFlujo/<name> → nombre pelado (y entonces ultralytics lo baja).

    Sin esto, ultralytics resuelve el nombre contra el CWD y se descarga otra copia
    de los pesos cada vez que el nodo se lanza desde otra carpeta — y encima deja el
    .pt tirado ahí. Es la misma resolución que usa el contador de flujo, así que las
    dos familias comparten el MISMO archivo de pesos."""
    for cand in (_EDGE_DIR / name, _CF_DIR / name):
        if cand.is_file():
            return str(cand)
    return name


def _vram_mb() -> Tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (torch.cuda.memory_allocated() / 1024 ** 2,
            torch.cuda.memory_reserved()  / 1024 ** 2)


# =============================================================================
# CROP ZONE STABILIZER
# =============================================================================
class CropZoneStabilizer:
    def __init__(self):
        self._smooth = None
        self._state  = "stable"
        self._queue  = []

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = float(a[0]), float(a[1]), float(a[2]), float(a[3])
        bx1, by1, bx2, by2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a1 = (ax2 - ax1) * (ay2 - ay1)
        a2 = (bx2 - bx1) * (by2 - by1)
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, bbox_new, clip_frame, display_crop):
        arr = np.array(bbox_new, dtype=float)
        if self._smooth is None:
            self._smooth = arr.copy()
            return display_crop, [clip_frame]
        iou = self._iou(arr, self._smooth)
        if self._state == "stable":
            if iou >= STAB_IOU_TH:
                self._smooth = STAB_EMA_ALPHA * arr + (1.0 - STAB_EMA_ALPHA) * self._smooth
                return display_crop, [clip_frame]
            self._state = "pending"
            self._queue = [(clip_frame, display_crop)]
            return display_crop, []
        else:
            if self._iou(arr, self._smooth) >= STAB_IOU_TH:
                flush = [cf for cf, _ in self._queue] + [clip_frame]
                self._smooth = STAB_EMA_ALPHA * arr + (1.0 - STAB_EMA_ALPHA) * self._smooth
                self._state = "stable"; self._queue = []
                return display_crop, flush
            self._queue.append((clip_frame, display_crop))
            if len(self._queue) >= STAB_PENDING_MAX:
                flush = [cf for cf, _ in self._queue] + [clip_frame]
                self._smooth = arr.copy()
                self._state = "stable"; self._queue = []
                return display_crop, flush
            return self._queue[-1][1], []

    def reset(self):
        self._smooth = None
        self._state  = "stable"
        self._queue  = []


# =============================================================================
# FIGHT ZONE
# =============================================================================
class FightZone:
    _id_counter = 0

    def __init__(self, pair_ids: Tuple[int, int]):
        FightZone._id_counter += 1
        self.zone_id   = FightZone._id_counter
        self.pair_ids  = tuple(sorted(pair_ids))

        self.buffer:         List[np.ndarray] = []
        self.stabilizer:     CropZoneStabilizer = CropZoneStabilizer()
        self.smoothed_score: float = 0.0
        self.consec_pos:     float = 0.0
        self.is_confirmed:   bool  = False
        self.processing:     bool  = False
        self.clips_analyzed: int   = 0
        self.last_raw:       float = 0.0

        self.idle_frames:    int = 0
        self.last_crop:      Optional[np.ndarray] = None
        self.last_bbox:      Optional[Tuple] = None
        self.created_frame:  int = 0
        self.last_fed_frame: int = 0

    def matches(self, p1: int, p2: int) -> bool:
        return tuple(sorted((p1, p2))) == self.pair_ids

    def update_score(self, raw: float):
        self.last_raw = raw
        self.clips_analyzed += 1
        if raw >= F_FAST_TRACK_TH:
            self.smoothed_score = raw
            self.consec_pos = 10
        else:
            self.smoothed_score = (F_SMOOTH_FACTOR * raw +
                                   (1.0 - F_SMOOTH_FACTOR) * self.smoothed_score)
            if self.smoothed_score > F_SET_SCORE_TH:
                self.consec_pos = min(6, self.consec_pos + 1)
            else:
                self.consec_pos = max(0, self.consec_pos - 1)
        self.is_confirmed = (self.consec_pos >= F_CONSEC_MIN)

    @property
    def should_discard(self) -> bool:
        if self.is_confirmed and self.smoothed_score > 1.0:
            return False
        return self.idle_frames > ZONE_IDLE_MAX

    def color(self) -> Tuple[int, int, int]:
        if self.is_confirmed:
            return (0, 0, 255)
        if self.smoothed_score >= F_SET_SCORE_TH:
            return (0, 140, 255)
        if self.smoothed_score >= 1.5:
            return (0, 220, 180)
        return (80, 80, 80)


# =============================================================================
# CLIP FIGHT EXPERT — stateless
# =============================================================================
class CLIPFightExpert:
    """Juez semántico: mira un mosaico temporal y devuelve la diferencia de logits
    entre los prompts de pelea y los de escena normal.

    Es **compartible**: no guarda estado de ninguna cámara ni de ninguna zona, así
    que un solo ejemplar sirve a todo el nodo (ver `shared_judge`)."""

    # Normalización de CLIP (idéntica a la de su `preprocess`), aplicada aquí con
    # OpenCV para no pasar por PIL: la conversión numpy→PIL→tensor costaba varios
    # ms de CPU por mosaico, y en la placa la CPU es el recurso disputado.
    _MEAN = (0.48145466, 0.4578275, 0.40821073)
    _STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self):
        self.model = self.preprocess = self.pos_text = self.neg_text = None
        self._dtype = torch.float32
        self._input_res = 224
        self._logit_scale = 100.0
        self._mean = torch.tensor(self._MEAN, dtype=torch.float32).view(3, 1, 1)
        self._std = torch.tensor(self._STD, dtype=torch.float32).view(3, 1, 1)
        self.name = ""

    def describe(self) -> str:
        return (f"{self.name or 'CLIP'} {self._input_res}px "
                f"{'fp16' if self._dtype == torch.float16 else 'fp32'} en {clip_device}")

    def load(self, model, preprocess=None, name: str = ""):
        self.model, self.preprocess, self.name = model, preprocess, name
        # El dtype que espera la torre visual (fp16 en CUDA, fp32 en CPU). Se toma
        # de la capa de entrada, no de `parameters()`: en CLIP conviven fp16 y fp32.
        try:
            self._dtype = model.visual.conv1.weight.dtype
        except AttributeError:                     # backbones ResNet (RN50, …)
            self._dtype = next(model.parameters()).dtype
        self._input_res = int(getattr(model.visual, "input_resolution", 224) or 224)
        self._logit_scale = float(model.logit_scale.exp().item())
        with torch.no_grad():
            pos = [
                "two people throwing punches and hitting each other hard",
                "violent street brawl with people kicking and punching aggressively",
                "two men fighting violently on the street throwing fists",
                "person striking another person with a punch or kick attack",
                "people wrestling aggressively on the ground fighting",
                "violent physical assault with hitting kicking and striking",
                "street fight with people grabbing and hitting each other",
                "two people in a violent altercation throwing blows at each other",
            ]
            neg = [
                "two people shaking hands and greeting each other calmly",
                "friends having a calm conversation standing close together",
                "two people hugging each other warmly and peacefully",
                "people standing still and talking face to face quietly",
                "two persons walking slowly side by side on the street",
                "pedestrians standing near each other waiting calmly",
                "people posing together for a photo smiling",
                "two friends meeting and greeting with a handshake or hug",
                "crowd of people walking together in a busy public area",
                "group of people moving through a crowded street normally",
                "people passing each other in a busy pedestrian zone",
                "bystanders standing around in a public space doing nothing",
                "people walking in different directions in a crowd",
                "two people standing close together in a group of people",
                "large crowd of pedestrians moving normally through a plaza",
                "many people walking and going about their daily routine outdoors",
                "busy street scene with lots of people going in different directions",
                "people accidentally bumping into each other while walking in a crowd",
                "dense crowd at a market or public space walking around normally",
                "people standing close together waiting in a line or queue patiently",
                "group of thirty people moving through a busy public area calmly",
                "crowd scene with many pedestrians walking side by side peacefully",
            ]
            pf = model.encode_text(clip.tokenize(pos).to(clip_device))
            nf = model.encode_text(clip.tokenize(neg).to(clip_device))
            self.pos_text = (pf / pf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)
            self.neg_text = (nf / nf.norm(dim=-1, keepdim=True)).mean(0, keepdim=True)

    @staticmethod
    def _build_mosaic(frames: List[np.ndarray]) -> np.ndarray:
        cw, ch = CLIP_FRAME_SIZE
        canvas = np.zeros((ch * MOSAIC_ROWS, cw * MOSAIC_COLS, 3), dtype=np.uint8)
        for idx in range(MOSAIC_ROWS * MOSAIC_COLS):
            r, c = idx // MOSAIC_COLS, idx % MOSAIC_COLS
            if idx < len(frames) and frames[idx] is not None and frames[idx].size > 0:
                cell = cv2.resize(frames[idx], (cw, ch))
            else:
                cell = np.zeros((ch, cw, 3), dtype=np.uint8)
            canvas[r * ch:(r + 1) * ch, c * cw:(c + 1) * cw] = cell
        return canvas

    @staticmethod
    def _ksm_select(frames: List[np.ndarray]) -> List[np.ndarray]:
        valid = [(i, f) for i, f in enumerate(frames) if f is not None and f.size > 0]
        if len(valid) <= KSM_SELECT:
            return [f for _, f in valid]
        activity = []
        for k, (i, f) in enumerate(valid):
            if k == 0:
                activity.append(0.0)
            else:
                prev = cv2.cvtColor(valid[k-1][1], cv2.COLOR_BGR2GRAY).astype(np.float32)
                curr = cv2.cvtColor(f,              cv2.COLOR_BGR2GRAY).astype(np.float32)
                activity.append(float(np.mean(np.abs(curr - prev))))
        if len(activity) > 1:
            activity[0] = activity[1]
        top = sorted(sorted(range(len(activity)), key=lambda k: activity[k], reverse=True)[:KSM_SELECT])
        return [valid[k][1] for k in top]

    def _to_tensor(self, mosaic_bgr: np.ndarray) -> torch.Tensor:
        """Mosaico BGR → tensor normalizado (3, n, n) listo para `encode_image`."""
        n = self._input_res
        if MOSAIC_FIT == "crop":
            # Réplica del `preprocess` de CLIP: escala el lado corto a n y recorta
            # el centro. Ojo: con un mosaico 3×2 eso TIRA los bordes de las
            # columnas laterales, así que el juez no ve el mosaico completo. Es lo
            # que hacía el pipeline original; se conserva por si la calibración
            # dice que era mejor.
            h, w = mosaic_bgr.shape[:2]
            s = n / min(h, w)
            img = cv2.resize(mosaic_bgr,
                             (max(int(round(w * s)), n), max(int(round(h * s)), n)),
                             interpolation=cv2.INTER_CUBIC)
            y0, x0 = (img.shape[0] - n) // 2, (img.shape[1] - n) // 2
            img = img[y0:y0 + n, x0:x0 + n]
        else:
            # "squash": el mosaico ENTERO en el cuadro de entrada. Deforma un poco
            # cada celda, pero CLIP ve las seis fases del gesto, que es justo para
            # lo que se monta el mosaico.
            img = cv2.resize(mosaic_bgr, (n, n), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb).permute(2, 0, 1)
        return ((t - self._mean) / self._std).to(dtype=self._dtype)

    @torch.no_grad()
    def predict_batch(self, frames_list: List[List[np.ndarray]]) -> List[dict]:
        """Juzga VARIOS buffers en un solo forward.

        Es la razón de ser del despachador: tres zonas de dos cámaras son seis
        mosaicos de 224×224, y pasarlos juntos por la GPU cuesta casi lo mismo que
        pasar uno solo."""
        empty = {"raw": 0.0, "fast_track": False, "mosaic": None}
        out: List[dict] = [dict(empty) for _ in frames_list]
        mosaics: List[Optional[np.ndarray]] = []
        for frames in frames_list:
            mosaics.append(self._build_mosaic(self._ksm_select(frames)) if frames else None)
        idx = [i for i, m in enumerate(mosaics) if m is not None]
        if not idx:
            return out
        batch = torch.stack([self._to_tensor(mosaics[i]) for i in idx]).to(clip_device)
        feat = self.model.encode_image(batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        raws = (self._logit_scale *
                (feat @ self.pos_text.T - feat @ self.neg_text.T)).squeeze(-1).float().cpu().numpy()
        for k, i in enumerate(idx):
            raw = float(raws[k])
            out[i] = {"raw": raw, "fast_track": raw >= F_FAST_TRACK_TH,
                      "mosaic": mosaics[i]}
        return out

    def predict(self, frames: List[np.ndarray]) -> dict:
        """Un solo buffer (uso sincrónico: CLI y calibración)."""
        return self.predict_batch([frames])[0]


# =============================================================================
# JUEZ COMPARTIDO + DESPACHADOR (una sola GPU, un solo CLIP)
# =============================================================================
# Antes cada cámara construía su propio FightProductionDetector y con él su propio
# CLIP y su propio hilo de inferencia: dos cámaras eran dos copias del MISMO modelo
# (~350 MB cada una) y dos hilos peleándose por la única GPU de la placa. Aquí el
# modelo y el worker son del proceso, no de la cámara; lo único que sigue siendo por
# cámara es YOLO, porque su tracker (ByteTrack) mantiene estado propio de la escena.
_JUDGE_LOCK = threading.Lock()
_JUDGES: Dict[str, CLIPFightExpert] = {}


def shared_judge(clip_model_name: str) -> CLIPFightExpert:
    """Devuelve el juez CLIP del proceso para ese backbone, cargándolo la 1ª vez."""
    with _JUDGE_LOCK:
        judge = _JUDGES.get(clip_model_name)
        if judge is not None:
            return judge
        print(f"[FIGHT] Cargando CLIP {clip_model_name} (compartido por todas las cámaras)...")
        # En CUDA, `clip.load` ya deja el modelo en fp16 — pero NO del todo: sus
        # LayerNorm se quedan en fp32 a propósito (el propio CLIP parchea el
        # forward para castear ahí). Un `model.half()` encima los convierte y
        # revienta con «expected scalar type Float but found Half». Se deja tal
        # como lo entrega `clip.load`.
        model, preprocess = clip.load(clip_model_name, device=clip_device)
        model.eval()
        judge = CLIPFightExpert()
        judge.load(model, preprocess, name=clip_model_name)
        _JUDGES[clip_model_name] = judge
        print(f"[FIGHT] CLIP listo: {judge.describe()}.")
        return judge


class _ClipDispatcher:
    """Cola única de mosaicos hacia el juez, con batching y un solo hilo.

    Contrapresión: si la cola está llena el mosaico se descarta (el detector
    seguirá mandando el siguiente buffer). Nunca bloquea la captura."""

    BATCH_MAX = 4          # mosaicos por forward
    DRAIN_WAIT_S = 0.02    # margen para que se junte el batch antes de disparar

    def __init__(self, judge: CLIPFightExpert, maxsize: int = 8):
        self.judge = judge
        self._q: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="fight-clip", daemon=True)
        self._thread.start()

    def submit(self, det, zone_id: int, frames: List[np.ndarray]) -> bool:
        try:
            self._q.put_nowait((det, zone_id, frames))
            return True
        except queue.Full:
            return False

    def stop(self):
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass

    def _run(self):
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            batch = [item]
            # Espera corta para agrupar lo que venga de otras zonas/cámaras.
            deadline = time.monotonic() + self.DRAIN_WAIT_S
            while len(batch) < self.BATCH_MAX:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    nxt = self._q.get(timeout=timeout)
                except queue.Empty:
                    break
                if nxt is None:
                    self._stop.set()
                    break
                batch.append(nxt)
            self._judge_batch(batch)

    def _judge_batch(self, batch):
        t0 = time.perf_counter()
        try:
            results = self.judge.predict_batch([frames for _, _, frames in batch])
        except Exception as e:  # noqa: BLE001 — un mosaico raro no tumba el juez
            print(f"[FIGHT] Error en el juez CLIP: {e}")
            import traceback
            traceback.print_exc()
            for det, zone_id, _ in batch:
                det._clip_failed(zone_id)
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0 / max(len(batch), 1)
        for (det, zone_id, _), res in zip(batch, results):
            try:
                det._apply_clip_result(zone_id, res["raw"], dt_ms)
            except Exception:  # noqa: BLE001
                det._clip_failed(zone_id)


_DISPATCH_LOCK = threading.Lock()
_DISPATCHERS: Dict[str, _ClipDispatcher] = {}


def shared_dispatcher(judge: CLIPFightExpert) -> _ClipDispatcher:
    """Despachador del proceso para ese juez (uno por backbone CLIP)."""
    key = judge.name or "default"
    with _DISPATCH_LOCK:
        d = _DISPATCHERS.get(key)
        if d is None:
            d = _ClipDispatcher(judge)
            _DISPATCHERS[key] = d
        return d


# =============================================================================
# FIGHT PRODUCTION DETECTOR
# =============================================================================
class FightProductionDetector:

    def __init__(
        self,
        gen_model="yolo11n.pt",
        tracker_yaml="bytetrack.yaml",
        clip_model_name="ViT-B/32",
        frame_skip=0,
        save=False,
        save_alerts=False,
    ):
        self.clip_model_name = clip_model_name
        self.frame_skip      = max(0, int(frame_skip))
        self.save            = save
        # Snapshots propios en ALERTAS_FIGHT/: en el nodo edge NO hacen falta (la
        # plataforma ya recorta y guarda el clip del evento), y en la Jetson son
        # escrituras a la SD en el camino caliente. Solo para el CLI.
        self.save_alerts     = bool(save_alerts)

        # ── YOLO ──────────────────────────────────────────────────────────────
        # Por cámara, no compartido: ByteTrack lleva el estado de los tracks de
        # ESTA escena. Es lo barato (yolo11n son ~6 MB de pesos); lo caro es CLIP,
        # y ese sí se comparte.
        self.gen_model    = YOLO(_resolve_yolo(gen_model))
        self.tracker_yaml = tracker_yaml

        # ── Escala espacial (se fija con el primer frame) ──────────────────────
        self._scale_fit = 0.0        # ancho/REF_FRAME_WIDTH; 0 = sin fijar
        self.dist_max   = float(F_DIST_MAX_LIMIT)
        self.pad_far    = int(F_PAD_PIXELS)
        self.pad_close  = int(F_PAD_CLOSE)

        # ── Multi-ROI ─────────────────────────────────────────────────────────
        self.fight_zones: List[FightZone] = []

        self.f_prev_boxes: Dict[int, np.ndarray] = {}
        self.zone_sticky:     Dict[int, int]   = {}
        self.zone_last_valid: Dict[int, tuple] = {}

        self.f_clips_total = 0

        # ── Último frame de personas detectadas (para overlay) ─────────────────
        self._last_person_xyxy = np.zeros((0, 4), dtype=np.float32)
        self._last_person_ids  = np.zeros((0,),   dtype=np.int32)

        # ── Profiling ─────────────────────────────────────────────────────────
        self.t_yolo_ms = 0.0
        self.t_clip_ms = 0.0

        # ── CLIP (compartido en todo el proceso) ──────────────────────────────
        self.f_expert = shared_judge(clip_model_name)
        self._dispatch = shared_dispatcher(self.f_expert)
        self._zone_lock = threading.Lock()   # el worker del juez toca las zonas

        # ── Alertas / grabación (solo CLI) ────────────────────────────────────
        self.alert_dir  = "ALERTAS_FIGHT"
        self._imwrite_q = None
        self._t_imwrite = None
        if self.save_alerts:
            os.makedirs(self.alert_dir, exist_ok=True)
            self._imwrite_q = queue.Queue(maxsize=4)
            self._t_imwrite = threading.Thread(target=self._imwrite_worker, daemon=True)
            self._t_imwrite.start()
        if save:
            os.makedirs("grabaciones", exist_ok=True)

        print(f"[FIGHT] Detector listo.  YOLO={gen_model}  CLIP={clip_model_name}  "
              f"device={clip_device}  frame_skip={frame_skip}")

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _ema(prev, new):
        return prev * (1.0 - _TIMING_ALPHA) + new * _TIMING_ALPHA

    def _async_save(self, path: str, img: np.ndarray):
        if self._imwrite_q is None:
            return
        try:
            self._imwrite_q.put_nowait((path, img.copy()))
        except queue.Full:
            pass

    def _fit_scale(self, W: int) -> None:
        """Ajusta los umbrales espaciales al ancho REAL del frame.

        Los 300 px de distancia máxima y los paddings se midieron en 1280 de ancho.
        El nodo entrega frames reescalados (~640 de ancho con `proc_short_side:
        360`), donde esos mismos 300 px son casi media pantalla: sin reescalar,
        cualquier par de personas de la escena formaría zona."""
        s = max(float(W), 1.0) / float(REF_FRAME_WIDTH)
        if abs(s - self._scale_fit) < 1e-6:
            return
        self._scale_fit = s
        self.dist_max = max(F_DIST_MAX_LIMIT * s, 24.0)
        self.pad_far = max(int(round(F_PAD_PIXELS * s)), 8)
        self.pad_close = max(int(round(F_PAD_CLOSE * s)), 12)
        print(f"[FIGHT] Escala espacial para {W}px de ancho (x{s:.2f}): "
              f"dist_max={self.dist_max:.0f}px pad={self.pad_far}/{self.pad_close}px")

    def _imwrite_worker(self):
        while True:
            item = self._imwrite_q.get()
            if item is None:
                break
            path, img = item
            cv2.imwrite(path, img)

    # =========================================================================
    # WORKER CLIP
    # =========================================================================
    def _zone_by_id(self, zone_id: int) -> Optional[FightZone]:
        with self._zone_lock:
            return next((z for z in self.fight_zones if z.zone_id == zone_id), None)

    def _submit_clip(self, zone: FightZone) -> None:
        """Manda el buffer de una zona al juez compartido. Si la cola está llena se
        descarta el mosaico (la zona volverá a llenar buffer)."""
        if zone.processing:
            return
        if self._dispatch.submit(self, zone.zone_id, list(zone.buffer)):
            zone.processing = True

    def _apply_clip_result(self, zone_id: int, raw: float, dt_ms: float) -> None:
        """Callback del despachador: aplica el veredicto del juez a la zona.

        Lo llama el hilo del juez, no el de la cámara: `_zone_lock` evita leer la
        lista de zonas mientras el pipeline la reescribe."""
        self.t_clip_ms = self._ema(self.t_clip_ms, dt_ms)
        self.f_clips_total += 1
        zone = self._zone_by_id(zone_id)
        if zone is None:
            return
        try:
            zone.update_score(raw)
            tag = (f"CONFIRMADO(x{zone.consec_pos:.0f})" if zone.is_confirmed
                   else (f"pico({zone.consec_pos:.1f})"
                         if zone.smoothed_score > F_SET_SCORE_TH else "no"))
            print(f"[FIGHT z{zone_id} {zone.pair_ids}] {tag}  "
                  f"ema={zone.smoothed_score:.2f}  raw={raw:.2f}  "
                  f"consec={zone.consec_pos:.1f}/{F_CONSEC_MIN}  "
                  f"clips={zone.clips_analyzed}  t={dt_ms:.0f}ms")
        finally:
            zone.processing = False

    def _clip_failed(self, zone_id: int) -> None:
        """El juez no pudo con este mosaico: liberar la zona para que reintente."""
        zone = self._zone_by_id(zone_id)
        if zone is not None:
            zone.processing = False

    # =========================================================================
    # GESTIÓN DE ZONAS
    # =========================================================================
    def _find_or_create_zone(self, p1: int, p2: int, frame_idx: int) -> Optional[FightZone]:
        for z in self.fight_zones:
            if z.matches(p1, p2):
                return z
        if len(self.fight_zones) >= ZONE_MAX:
            return None
        z = FightZone((p1, p2))
        z.created_frame = frame_idx
        with self._zone_lock:
            self.fight_zones.append(z)
        print(f"[FIGHT] Nueva zona z{z.zone_id} ids=({p1},{p2})  "
              f"total_zonas={len(self.fight_zones)}")
        return z

    def _prune_zones(self):
        before = len(self.fight_zones)
        keep = [z for z in self.fight_zones if not z.should_discard]
        # Una zona con mosaico en vuelo no se tira: el juez volvería con un
        # zone_id que ya no existe y su veredicto se perdería.
        keep += [z for z in self.fight_zones if z.should_discard and z.processing]
        with self._zone_lock:
            self.fight_zones = keep
        removed = before - len(self.fight_zones)
        if removed:
            print(f"[FIGHT] {removed} zona(s) descartada(s). Activas: {len(self.fight_zones)}")

    @property
    def _primary_zone(self) -> Optional[FightZone]:
        if not self.fight_zones:
            return None
        return max(self.fight_zones, key=lambda z: z.smoothed_score)

    @property
    def any_confirmed(self) -> bool:
        return any(z.is_confirmed for z in self.fight_zones)

    # =========================================================================
    # HELPERS ESPACIALES
    # =========================================================================
    @staticmethod
    def _iou(b1, b2) -> float:
        xA, yA = max(b1[0], b2[0]), max(b1[1], b2[1])
        xB, yB = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter  = max(0.0, xB - xA) * max(0.0, yB - yA)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    @staticmethod
    def _center_dist(b1, b2) -> float:
        cx1 = (b1[0] + b1[2]) / 2.0; cy1 = (b1[1] + b1[3]) / 2.0
        cx2 = (b2[0] + b2[2]) / 2.0; cy2 = (b2[1] + b2[3]) / 2.0
        return float(np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2))

    def _movement_score(self, p_id: int, box_p) -> float:
        prev = self.f_prev_boxes.get(p_id)
        if prev is None:
            return 0.0
        return (1.0 - self._iou(box_p, prev)) * 50.0

    def _score_pair(self, b1, b2, p1: int, p2: int) -> float:
        iou  = self._iou(b1, b2)
        dist = self._center_dist(b1, b2)
        prox = max(0.0, 1.0 - dist / self.dist_max) * 40.0
        mov  = (self._movement_score(p1, b1) + self._movement_score(p2, b2)) / 2.0
        return prox + iou * 100.0 + mov

    @staticmethod
    def _union_bbox(b1, b2, pad: int, W: int, H: int) -> Tuple[int, int, int, int]:
        x1 = int(max(0, min(b1[0], b2[0]) - pad))
        y1 = int(max(0, min(b1[1], b2[1]) - pad))
        x2 = int(min(W, max(b1[2], b2[2]) + pad))
        y2 = int(min(H, max(b1[3], b2[3]) + pad))
        return x1, y1, x2, y2

    # =========================================================================
    # PIPELINE PRINCIPAL
    # =========================================================================
    def _process_fight(self, frame: np.ndarray, person_xyxy, person_ids, frame_idx: int):
        n = len(person_xyxy)
        H, W = frame.shape[:2]

        candidatos = []
        for i in range(n):
            for j in range(i + 1, n):
                b1, b2 = person_xyxy[i], person_xyxy[j]
                p1 = int(person_ids[i]) if i < len(person_ids) else -1
                p2 = int(person_ids[j]) if j < len(person_ids) else -1
                iou  = self._iou(b1, b2)
                dist = self._center_dist(b1, b2)
                if dist < self.dist_max or iou > F_MIN_IOU_CRITICAL:
                    sc = self._score_pair(b1, b2, p1, p2)
                    candidatos.append((sc, iou, dist, i, j, p1, p2))
        candidatos.sort(key=lambda x: x[0], reverse=True)

        ids_visibles = set()
        for i, box_p in enumerate(person_xyxy):
            p_id = int(person_ids[i]) if i < len(person_ids) else -1
            self.f_prev_boxes[p_id] = box_p.copy()
            ids_visibles.add(p_id)
        self.f_prev_boxes = {k: v for k, v in self.f_prev_boxes.items() if k in ids_visibles}

        active_pair_keys = set()

        for cand in candidatos:
            sc, iou, dist, i, j, p1, p2 = cand
            b1, b2 = person_xyxy[i], person_xyxy[j]
            if p1 == -1 or p2 == -1:
                continue
            if sc < ZONE_MIN_PAIR_SCORE:
                continue
            zone = self._find_or_create_zone(p1, p2, frame_idx)
            if zone is None:
                continue

            active_pair_keys.add(zone.zone_id)
            zone.idle_frames = 0
            zone.last_fed_frame = frame_idx

            pad = self.pad_close if iou > 0.05 else self.pad_far
            x1, y1, x2, y2 = self._union_bbox(b1, b2, pad, W, H)
            raw_crop = frame[y1:y2, x1:x2]
            if raw_crop is None or raw_crop.size == 0:
                continue

            zone.last_bbox = (x1, y1, x2, y2)

            mov1 = self._movement_score(p1, b1)
            mov2 = self._movement_score(p2, b2)
            if (mov1 + mov2) / 2.0 < F_MIN_MOVEMENT_FOR_CLIP:
                zone.buffer = []
                self.zone_sticky[zone.zone_id]     = F_STICKY_FRAMES
                self.zone_last_valid[zone.zone_id] = (b1.copy(), b2.copy(), pad)
                continue

            p_crop       = cv2.resize(raw_crop, _CLIP_VIEW_SIZE)
            p_clip_frame = cv2.resize(raw_crop, CLIP_FRAME_SIZE)
            zone.last_crop = p_crop

            self.zone_sticky[zone.zone_id]     = F_STICKY_FRAMES
            self.zone_last_valid[zone.zone_id] = (b1.copy(), b2.copy(), pad)

            stable_crop, frames_to_flush = zone.stabilizer.update(
                (x1, y1, x2, y2), p_clip_frame, p_crop)
            zone.last_crop = stable_crop

            for cf in frames_to_flush:
                zone.buffer.append(cf)
                if len(zone.buffer) >= F_BUFFER_SIZE:
                    self._submit_clip(zone)
                    zone.buffer = []

        is_crowd = n > CROWD_PERSON_TH
        for z in self.fight_zones:
            if z.zone_id in active_pair_keys:
                continue
            if is_crowd and z.last_bbox is not None:
                zx1, zy1, zx2, zy2 = z.last_bbox
                persons_in_zone = sum(
                    1 for box in person_xyxy
                    if zx1 <= (box[0] + box[2]) / 2 <= zx2
                    and zy1 <= (box[1] + box[3]) / 2 <= zy2
                )
                if persons_in_zone >= 2:
                    z.idle_frames = 0
                    continue
            sticky = self.zone_sticky.get(z.zone_id, 0)
            if sticky > 0 and z.zone_id in self.zone_last_valid:
                b1, b2, pad = self.zone_last_valid[z.zone_id]
                x1, y1, x2, y2 = self._union_bbox(b1, b2, pad, W, H)
                raw_crop = frame[y1:y2, x1:x2]
                if raw_crop is not None and raw_crop.size > 0:
                    p_crop       = cv2.resize(raw_crop, _CLIP_VIEW_SIZE)
                    p_clip_frame = cv2.resize(raw_crop, CLIP_FRAME_SIZE)
                    z.last_crop  = p_crop
                    stable_crop, frames_to_flush = z.stabilizer.update(
                        (x1, y1, x2, y2), p_clip_frame, p_crop)
                    z.last_crop = stable_crop
                    for cf in frames_to_flush:
                        z.buffer.append(cf)
                        if len(z.buffer) >= F_BUFFER_SIZE:
                            self._submit_clip(z)
                            z.buffer = []
                self.zone_sticky[z.zone_id] = sticky - 1
                active_pair_keys.add(z.zone_id)
            else:
                z.idle_frames += 1
                if z.idle_frames > 5:
                    z.consec_pos  = max(0, z.consec_pos - 0.1)
                    z.is_confirmed = (z.consec_pos >= F_CONSEC_MIN)

        self._prune_zones()

    # =========================================================================
    # OVERLAY — display mínimo sobre frame original
    # =========================================================================
    def _draw_overlay(self, frame: np.ndarray, fps: float, f_idx: int,
                      sx: float = 1.0, sy: float = 1.0) -> np.ndarray:
        H, W   = frame.shape[:2]
        out    = frame.copy()
        pz     = self._primary_zone
        confirmed_now = self.any_confirmed

        # ── Bboxes individuales de personas (coords nativas → display) ───────
        for i, xyxy in enumerate(self._last_person_xyxy):
            px1 = int(xyxy[0] * sx); py1 = int(xyxy[1] * sy)
            px2 = int(xyxy[2] * sx); py2 = int(xyxy[3] * sy)
            pid = int(self._last_person_ids[i]) if i < len(self._last_person_ids) else -1
            cv2.rectangle(out, (px1, py1), (px2, py2), (0, 255, 0), 1)
            cv2.putText(out, f"#{pid}", (px1 + 2, py2 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)

        # ── ROI rectangles por zona (coords nativas → display) ───────────────
        for z in self.fight_zones:
            if z.last_bbox is None:
                continue
            x1 = int(z.last_bbox[0] * sx); y1 = int(z.last_bbox[1] * sy)
            x2 = int(z.last_bbox[2] * sx); y2 = int(z.last_bbox[3] * sy)
            color     = z.color()
            thickness = 3 if z.is_confirmed else 2
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label_y = max(y1 - 8, 14)
            cv2.putText(out,
                        f"z{z.zone_id} {z.pair_ids}  ema={z.smoothed_score:.2f}",
                        (x1 + 4, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)

        # ── Banner superior (50px) ───────────────────────────────────────────
        BANNER_H = 50
        if confirmed_now:
            cv2.rectangle(out, (0, 0), (W, BANNER_H), (0, 0, 160), -1)
            cv2.putText(out, "PELEA DETECTADA", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 60, 255), 2, cv2.LINE_AA)
            if pz:
                cv2.putText(out,
                            f"z{pz.zone_id} {pz.pair_ids}  "
                            f"ema={pz.smoothed_score:.2f}  raw={pz.last_raw:.2f}  "
                            f"consec={pz.consec_pos:.0f}/{F_CONSEC_MIN}",
                            (350, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 180, 255), 1, cv2.LINE_AA)
            cv2.rectangle(out, (0, 0), (W - 1, H - 1), (0, 0, 255), 4)
        else:
            cv2.rectangle(out, (0, 0), (W, BANNER_H), (0, 60, 0), -1)
            cv2.putText(out, "NORMAL", (10, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 0), 2, cv2.LINE_AA)
            if pz and pz.smoothed_score > 0.5:
                cv2.putText(out,
                            f"z{pz.zone_id} ema={pz.smoothed_score:.2f}  "
                            f"zonas={len(self.fight_zones)}/{ZONE_MAX}",
                            (200, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 200, 120), 1, cv2.LINE_AA)

        # ── Info bar inferior (30px) ─────────────────────────────────────────
        cv2.rectangle(out, (0, H - 30), (W, H), (10, 10, 10), -1)
        va, _ = _vram_mb()
        cv2.putText(out,
                    f"FPS:{fps:.1f}  skip:{self.frame_skip}  f:{f_idx}  "
                    f"CLIP:{self.clip_model_name}  "
                    f"YOLO:{self.t_yolo_ms:.0f}ms  CLIP:{self.t_clip_ms:.0f}ms  "
                    f"VRAM:{va:.0f}MB  "
                    f"zonas:{len(self.fight_zones)}/{ZONE_MAX}  CLIPs:{self.f_clips_total}",
                    (6, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 200, 160), 1, cv2.LINE_AA)

        return out

    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    def step(self, frame: np.ndarray, frame_idx: int) -> set:
        """Procesa UN frame BGR (resolución nativa): YOLO de personas + pipeline de
        zonas FIGHT. No dibuja ni guarda nada. Devuelve el set de zone_ids
        confirmadas tras este frame. Reusado por `run()` (CLI) y por el adaptador
        `Model` (nodo edge)."""
        self._fit_scale(frame.shape[1])
        t_yolo0 = time.perf_counter()
        res = self.gen_model.track(
            frame,
            classes=[CLASS_PERSON],
            persist=True,
            verbose=False,
            tracker=self.tracker_yaml,
        )[0]
        self.t_yolo_ms = self._ema(self.t_yolo_ms, (time.perf_counter() - t_yolo0) * 1000.0)

        person_xyxy = np.zeros((0, 4), dtype=np.float32)
        person_ids  = np.zeros((0,),   dtype=np.int32)
        if res.boxes is not None and len(res.boxes) > 0:
            b        = res.boxes
            all_xyxy = b.xyxy.cpu().numpy().astype(np.float32)
            all_ids  = (b.id.cpu().numpy().astype(np.int32)
                        if b.id is not None
                        else np.full(len(all_xyxy), -1, dtype=np.int32))
            all_cls  = b.cls.cpu().numpy().astype(np.int32)
            p_mask   = (all_cls == CLASS_PERSON)
            person_xyxy = all_xyxy[p_mask]
            person_ids  = all_ids[p_mask]

        self._process_fight(frame, person_xyxy, person_ids, frame_idx)
        self._last_person_xyxy = person_xyxy
        self._last_person_ids  = person_ids
        return {z.zone_id for z in self.fight_zones if z.is_confirmed}

    def max_smoothed_score(self) -> float:
        """Mayor score suavizado entre las zonas activas (0 si no hay)."""
        return max((z.smoothed_score for z in self.fight_zones), default=0.0)

    def run(self, source, resize=None, show=True, summary=True):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] No se pudo abrir: {source}")
            return

        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if resize:
            panel_w, panel_h = resize
        elif native_w > DISPLAY_SIZE[0] or native_h > DISPLAY_SIZE[1]:
            panel_w, panel_h = DISPLAY_SIZE
        else:
            panel_w, panel_h = native_w, native_h
        if (panel_w, panel_h) != (native_w, native_h):
            print(f"[RESIZE] {native_w}×{native_h} → {panel_w}×{panel_h}")

        writer = None
        if self.save:
            ts     = time.strftime("%Y%m%d_%H%M%S")
            writer = cv2.VideoWriter(
                f"grabaciones/FIGHT_PROD_{ts}.avi",
                cv2.VideoWriter_fourcc(*"XVID"), 20.0, (panel_w, panel_h))

        f_raw = 0
        f_idx = 0
        t_start    = time.perf_counter()
        t_last     = time.perf_counter()
        fps_smooth = 0.0

        # Seguimiento de alertas — detectar transiciones confirmed/no
        prev_confirmed_ids: set = set()

        # Factores de escala: detección (nativa) → display (960×540)
        sx = panel_w / native_w
        sy = panel_h / native_h

        print(f"\n{'=' * 62}")
        print("  FIGHT PRODUCTION DETECTOR")
        print(f"  CLIP        : {self.clip_model_name}")
        print(f"  device      : {clip_device}  frame_skip={self.frame_skip}")
        print(f"  detección   : {native_w}×{native_h}  (nativa, YOLO full-res)")
        print(f"  display     : {panel_w}×{panel_h}  (scale={sx:.2f}x{sy:.2f})")
        print(f"  zonas_max   : {ZONE_MAX}  idle_max={ZONE_IDLE_MAX}")
        print(f"  score_th    : {F_SET_SCORE_TH}  fast_tk={F_FAST_TRACK_TH}  consec={F_CONSEC_MIN}")
        print(f"  Presiona Q para salir")
        print(f"{'=' * 62}\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            f_raw += 1
            if self.frame_skip > 0 and (f_raw - 1) % (self.frame_skip + 1) != 0:
                continue

            t_now = time.perf_counter()
            fps_smooth = 0.9 * fps_smooth + 0.1 / max(t_now - t_last, 1e-6)
            t_last = t_now

            # ── Detección + pipeline FIGHT (extraído a step() para reuso) ──────
            self.step(frame, f_idx)

            # ── Resize solo para display ───────────────────────────────────────
            display_frame = (cv2.resize(frame, (panel_w, panel_h))
                             if (panel_w, panel_h) != (native_w, native_h) else frame)

            # ── Transiciones de alerta ────────────────────────────────────────
            cur_confirmed_ids = {z.zone_id for z in self.fight_zones if z.is_confirmed}

            new_alerts  = cur_confirmed_ids - prev_confirmed_ids
            resolutions = prev_confirmed_ids - cur_confirmed_ids

            for zid in new_alerts:
                z = next((z for z in self.fight_zones if z.zone_id == zid), None)
                ids_str = str(z.pair_ids) if z else "?"
                ts = time.strftime("%Y%m%d_%H%M%S")
                print(f"\n{'!' * 55}")
                print(f"  *** ALERTA PELEA  frame={f_idx}  zona=z{zid}{ids_str} ***")
                print(f"{'!' * 55}\n")
                self._async_save(
                    os.path.join(self.alert_dir, f"FIGHT_z{zid}_{ts}.jpg"),
                    display_frame,
                )

            for zid in resolutions:
                print(f"  --- PELEA OFF  zona=z{zid}  frame={f_idx}")

            prev_confirmed_ids = cur_confirmed_ids

            # ── Overlay sobre display_frame con coords escaladas ──────────────
            out = self._draw_overlay(display_frame, fps_smooth, f_idx, sx, sy)

            if show:
                cv2.imshow("FightDetector Production", out)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[PROD] Detenido por usuario.")
                    break

            if writer is not None:
                writer.write(out)

            # Live snapshot para monitoreo remoto
            self._async_save("FIGHT_PROD_LIVE.jpg", out)

            if f_idx % 30 == 0:
                z_summary = " | ".join(
                    f"z{z.zone_id}({z.pair_ids[0]},{z.pair_ids[1]})"
                    f" s={z.smoothed_score:.2f} c={z.consec_pos:.1f}"
                    for z in self.fight_zones
                ) or "—"
                print(f"[F:{f_idx:05d}] {fps_smooth:.1f}fps  P:{len(person_xyxy)}  "
                      f"{'PELEA' if self.any_confirmed else 'normal'}  "
                      f"YOLO:{self.t_yolo_ms:.0f}ms CLIP:{self.t_clip_ms:.0f}ms  "
                      f"zonas=[{z_summary}]")

            f_idx += 1

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        # ── Shutdown limpio ────────────────────────────────────────────────────
        # El juez CLIP y su hilo son del proceso (compartidos), así que aquí solo
        # se cierra lo propio de esta ejecución.
        if self._imwrite_q is not None:
            self._imwrite_q.put(None)
        if self._t_imwrite is not None:
            self._t_imwrite.join(timeout=3)

        t_total = time.perf_counter() - t_start

        if summary:
            va, vr = _vram_mb()
            print(f"\n{'=' * 62}")
            print("  RESUMEN FIGHT PRODUCTION DETECTOR")
            print(f"{'=' * 62}")
            print(f"  CLIP model        : {self.clip_model_name}")
            print(f"  device            : {clip_device}")
            print(f"  frame-skip        : {self.frame_skip}")
            print(f"  detección         : {native_w}×{native_h}  (resolución nativa)")
            print(f"  display           : {panel_w}×{panel_h}  (scale={sx:.2f}x{sy:.2f})")
            print(f"  Frames leídos     : {f_raw}")
            print(f"  Frames procesados : {f_idx}  (con YOLO, frame-skip={self.frame_skip})")
            print(f"  Tiempo total      : {t_total:.1f}s")
            print(f"  FPS procesados    : {f_idx / max(t_total, 1e-6):.1f}  (solo frames YOLO)")
            print(f"  FPS equivalente   : {f_raw / max(t_total, 1e-6):.1f}  (incluyendo frames saltados)")
            print(f"  CLIP batches      : {self.f_clips_total}")
            print(f"  ── Timings (EMA al cierre) ──")
            print(f"  YOLO ms/llamada   : {self.t_yolo_ms:.1f}ms")
            print(f"  CLIP ms/batch     : {self.t_clip_ms:.1f}ms")
            print(f"  ── VRAM (última lectura) ──")
            print(f"  VRAM alloc        : {va:.0f} MB")
            print(f"  VRAM reserved     : {vr:.0f} MB")
            print(f"{'=' * 62}\n")


# =============================================================================
# ADAPTADOR para el nodo edge (contrato StreamModel — ver edge/inference/base.py)
# =============================================================================
class _CamStream:
    """Pipeline por cámara: su propio FightDetector, cola y worker. Convierte la
    confirmación de pelea en `ClipResult` (score 1.0 / clase 'fight') y la empuja."""

    EMIT_INTERVAL_S = 1.0    # cadencia mínima de ClipResult hacia el EventEngine
    QUEUE_MAX = 2            # back-pressure: en tiempo real se descartan frames viejos

    def __init__(self, camera_id, detector, on_result, stop_event, frame_sink=None,
                 infer_every_n=1):
        self.camera_id = camera_id
        self._det = detector
        self._on_result = on_result
        self._stop = stop_event
        self._frame_sink = frame_sink   # callback(cam_id, rgb) para la vista en vivo anotada
        self._every = max(1, int(infer_every_n or 1))
        self._seen = 0                  # frames recibidos (para el submuestreo)
        self._q = queue.Queue(maxsize=self.QUEUE_MAX)
        self._thread = threading.Thread(target=self._run, name=f"fight-{camera_id}", daemon=True)
        self._idx = 0
        self._confirmed = False
        self._last_emit = 0.0

    def start(self):
        self._thread.start()

    def submit(self, frame_bgr, ts):
        # Submuestreo explícito: con varias cámaras conviene analizar 1 de cada N
        # y dejar GPU libre, en vez de dejar que la cola descarte a ciegas.
        self._seen += 1
        if self._every > 1 and (self._seen % self._every):
            return
        try:
            self._q.put_nowait((frame_bgr, ts))
        except queue.Full:                       # descarta el más viejo (tiempo real)
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._q.put_nowait((frame_bgr, ts))
            except queue.Full:
                pass

    def join(self, timeout=5):
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _run(self):
        import logging
        log = logging.getLogger("heuristic.fight_detector")
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                break
            frame_bgr, ts = item
            t0 = time.perf_counter()
            try:
                confirmed_ids = self._det.step(frame_bgr, self._idx)
            except Exception:  # noqa: BLE001 — un frame no debe tumbar el worker
                log.exception("step() del FightDetector [%s] falló", self.camera_id)
                continue
            self._idx += 1
            now_confirmed = bool(confirmed_ids)
            changed = now_confirmed != self._confirmed
            self._confirmed = now_confirmed
            latency_ms = (time.perf_counter() - t0) * 1000.0
            # Vista en vivo con los bounding boxes que dibuja el propio modelo
            # (personas + zonas de pelea + banner). Se empuja el frame anotado.
            if self._frame_sink is not None:
                try:
                    fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
                    annotated = self._det._draw_overlay(frame_bgr, fps, self._idx)
                    self._frame_sink(self.camera_id,
                                     cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
                except Exception:  # noqa: BLE001 — el overlay no debe tumbar el worker
                    log.exception("overlay del FightDetector [%s] falló", self.camera_id)
            now = time.monotonic()
            if changed or (now - self._last_emit) >= self.EMIT_INTERVAL_S:
                self._last_emit = now
                self._emit(ts, latency_ms)

    def _emit(self, ts, latency_ms):
        from edge.types import ClipResult
        if self._confirmed:
            score = 1.0
        else:  # sin confirmar: fracción del score suavizado, siempre bajo umbral
            sm = self._det.max_smoothed_score()
            score = min(0.49, max(0.0, sm / (2.0 * F_SET_SCORE_TH)))
        class_id = 1 if self._confirmed else 0
        self._on_result(ClipResult(
            camera_id=self.camera_id, t_start=ts, t_end=ts,
            score=score, class_id=class_id,
            class_probs=[1.0 - score, score], latency_ms=latency_ms,
        ))


class Model:
    """Adaptador stream_processor MULTI-CÁMARA: un FightDetector por cámara, creado
    perezosamente al llegar el 1er frame de esa cámara (en un hilo aparte, para no
    bloquear la captura mientras carga YOLO).

    El juez CLIP —lo caro— es **uno para todo el proceso** (`shared_judge`), con un
    único hilo que junta los mosaicos de todas las cámaras en un solo forward. Lo
    que se paga por cámara es solo YOLO11n + su tracker.

    Presupuesto de RAM (`mem_budget_mb`, por defecto 4700): antes de crear el
    detector de una cámara nueva se estima el **pico** de RAM del proceso con esa
    cámara incluida; si se pasaría del tope —o dejaría a la placa sin los MB libres
    que necesita— esa cámara se OMITE con aviso. La 1ª siempre entra.
    """

    # Cifras medidas en la Jetson (7451 MB totales, ~1500 se los lleva el escritorio)
    # el 2026-08-19, con vídeo real y el modelo caliente:
    #   1 cámara  -> 4500 MB de proceso, 164 MB libres en la placa
    #   2 cámaras -> 4770 MB  ... y el nodo se congela: ningún hilo avanza, el
    #                dashboard deja de aceptar conexiones y Ctrl+C no lo cierra.
    # Lo que cuesta no es la cámara, es el arranque: el CLIP es compartido
    # (`shared_judge`) y por cámara extra solo se paga YOLO11n + su tracker.
    #
    # El presupuesto se compara contra el pico **estimado** del proceso, no contra el
    # RSS del momento: cuando se decide, los modelos todavía no están cargados y el
    # RSS no dice nada. Con el tope viejo (5000 MB contra el RSS de ese instante) no
    # frenaba nunca a nadie y el cuelgue llegaba diez minutos después, sin aviso.
    MEM_BUDGET_MB = 4700            # pico de RAM permitido al proceso (ajustable en META)
    EST_FIRST_CAM_MB = 4500         # 1ª cámara: torch + CUDA + CLIP + YOLO + búferes
    EST_PER_CAM_MB = 300            # cada cámara extra: YOLO11n + tracker
    #: RAM que hay que dejarle a la placa (escritorio, red, el propio SO). Segunda
    #: barrera: protege aunque alguien suba `mem_budget_mb` por encima de lo que hay.
    MIN_SYS_AVAIL_MB = 1200

    def __init__(self, cfg=None):
        self._cfg = cfg
        self._on_result = None
        self._frame_sink = None        # callback(cam_id, rgb) para la vista en vivo anotada
        self._cams: Dict[str, _CamStream] = {}
        self._loading: set = set()
        self._denied: set = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._budget_mb = int(META.get("mem_budget_mb", self.MEM_BUDGET_MB))
        self._yolo = META.get("yolo_model", "yolo11n.pt")
        self._clip = META.get("clip_model", "ViT-B/32")
        self._every = max(1, int(META.get("infer_every_n", 1) or 1))

    # ── contrato StreamModel ────────────────────────────────────────────────
    def set_frame_sink(self, fn):
        """Opcional: el nodo pasa aquí su `monitor.update_frame` para que la vista
        en vivo muestre el frame anotado por el modelo (bboxes de personas/zonas)."""
        self._frame_sink = fn

    def start(self, on_result):
        self._on_result = on_result
        # Los detectores se crean al llegar frames de cada cámara (ver feed()).
        print(f"[FIGHT] Multi-cámara listo. YOLO={self._yolo} CLIP={self._clip} "
              f"presupuesto RAM={self._budget_mb}MB.")

    def feed(self, camera_id, frame_rgb, ts):
        cam = self._cams.get(camera_id)
        if cam is None:
            self._ensure_cam(camera_id)       # dispara carga en bg; se dropea este frame
            return
        cam.submit(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), ts)   # RGB->BGR (cv2/YOLO)

    def stop(self):
        self._stop.set()
        with self._lock:
            cams = list(self._cams.values())
        for c in cams:
            c.join(timeout=5)
        # El juez y su hilo son del proceso: se cierran al apagar el modelo.
        with _DISPATCH_LOCK:
            for d in _DISPATCHERS.values():
                d.stop()

    def class_name(self, class_id):
        names = META["classes"]
        return names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"

    # ── gestión de detectores por cámara ─────────────────────────────────────
    @staticmethod
    def _rss_mb() -> float:
        try:
            import os
            import psutil
            return psutil.Process(os.getpid()).memory_info().rss / 1e6
        except Exception:  # noqa: BLE001 — sin psutil no aplicamos presupuesto
            return 0.0

    @staticmethod
    def _sys_total_mb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / 1e6
        except Exception:  # noqa: BLE001 — sin psutil no aplicamos presupuesto
            return 0.0

    def _ensure_cam(self, camera_id):
        with self._lock:
            if camera_id in self._cams or camera_id in self._loading \
                    or camera_id in self._denied:
                return
            # Se cuentan también las que están cargando: las cámaras arrancan a la
            # vez, así que al decidir la segunda la primera aún está en `_loading` y
            # `_cams` está vacío. Mirando solo `_cams`, el presupuesto no frenaba a
            # ninguna de las dos — el fallo que dejó colgado el nodo el 2026-08-19.
            n_previas = len(self._cams) + len(self._loading)
            pico_mb = self.EST_FIRST_CAM_MB + self.EST_PER_CAM_MB * n_previas
            # La 1ª cámara siempre entra: sin ella el nodo no tendría modelo ninguno.
            # Denegar una cámara con un aviso claro es mucho mejor que aceptarla y
            # congelar el proceso entero un rato después.
            reason = None
            if n_previas and pico_mb > self._budget_mb:
                reason = (f"pico estimado {pico_mb:.0f}MB con {n_previas + 1} cámaras > "
                          f"tope {self._budget_mb}MB del proceso")
            else:
                total_mb = self._sys_total_mb()
                if n_previas and total_mb and pico_mb > (total_mb - self.MIN_SYS_AVAIL_MB):
                    reason = (f"pico estimado {pico_mb:.0f}MB dejaría a la placa "
                              f"(RAM total {total_mb:.0f}MB) por debajo de los "
                              f"{self.MIN_SYS_AVAIL_MB}MB libres que necesita")
            if reason:
                self._denied.add(camera_id)
                print(f"[FIGHT] Sin RAM para '{camera_id}' ({reason}); se omite y el "
                      f"nodo sigue con las demás. Sube META['mem_budget_mb'] si de "
                      f"verdad hay memoria, o arranca la placa sin escritorio (~1.5GB).")
                return
            self._loading.add(camera_id)
        threading.Thread(target=self._load_cam, args=(camera_id,),
                         name=f"fight-load-{camera_id}", daemon=True).start()

    def _load_cam(self, camera_id):
        import logging
        log = logging.getLogger("heuristic.fight_detector")
        try:
            print(f"[FIGHT] Cargando detector para '{camera_id}' "
                  f"(YOLO={self._yolo}, CLIP={self._clip})...")
            det = FightProductionDetector(save=False, save_alerts=False,
                                          gen_model=self._yolo,
                                          clip_model_name=self._clip)
            cam = _CamStream(camera_id, det, self._on_result, self._stop,
                             frame_sink=self._frame_sink,
                             infer_every_n=self._every)
            cam.start()
        except Exception:  # noqa: BLE001
            log.exception("No se pudo cargar el detector de '%s'", camera_id)
            with self._lock:
                self._loading.discard(camera_id)
            return
        with self._lock:
            self._cams[camera_id] = cam
            self._loading.discard(camera_id)
        print(f"[FIGHT] '{camera_id}' activa (RAM del proceso {self._rss_mb():.0f}MB).")


def build(cfg=None):
    """Factory que usa el registro modular para construir el modelo."""
    return Model(cfg)


# Umbrales calibrados (si `fight_params.json` existe). Se aplica al importar, así
# que vale igual para el nodo, el CLI y la herramienta de calibración.
load_params()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="FightDetector Production — display mínimo sobre frame original",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Ejemplos:\n"
            "  python FightDetector_Production.py video.mp4\n"
            "  python FightDetector_Production.py rtsp://...\n"
            "  python FightDetector_Production.py video.mp4 --clip-model ViT-B/32\n"
            "  python FightDetector_Production.py video.mp4 --frame-skip 1 --save\n"
            "  python FightDetector_Production.py rtsp://... --no-display --no-summary\n"
            "  python FightDetector_Production.py video.mp4 --resize 1280x720 --save\n"
        ),
    )
    ap.add_argument("source", help="Ruta al video o URL RTSP")
    ap.add_argument(
        "--clip-model", default="ViT-L/14",
        help="Modelo CLIP (default: ViT-L/14). Opciones: ViT-B/32, ViT-B/16, ViT-L/14",
    )
    ap.add_argument(
        "--frame-skip", type=int, default=0, metavar="N",
        help="Saltar N frames entre detecciones (default: 0 = todos)",
    )
    ap.add_argument(
        "--resize", default=None, metavar="WxH",
        help="Redimensionar frames (ej. 1280x720)",
    )
    ap.add_argument("--save",       action="store_true", help="Guardar video en grabaciones/")
    ap.add_argument("--no-display", action="store_true", help="Sin ventana OpenCV (headless)")
    ap.add_argument("--no-summary", action="store_true", help="Omitir resumen final")
    args = ap.parse_args()

    # Fuente: int para webcam, str para archivo/rtsp
    src = args.source
    try:
        src = int(src)
    except ValueError:
        pass

    resize = None
    if args.resize:
        try:
            rw, rh = args.resize.lower().split("x")
            resize = (int(rw), int(rh))
        except ValueError:
            print("[ERROR] --resize: usar formato WxH  (ej. 1280x720)")
            sys.exit(1)

    FightProductionDetector(
        clip_model_name=args.clip_model,
        frame_skip=args.frame_skip,
        save=args.save,
    ).run(
        src,
        resize=resize,
        show=not args.no_display,
        summary=not args.no_summary,
    )
