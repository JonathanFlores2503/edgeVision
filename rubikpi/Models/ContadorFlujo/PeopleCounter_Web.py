#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PeopleCounter_Web.py
====================
Contador de personas ENTRADA/SALIDA por LINEA DE CRUCE (misma logica que
trackingObjects/contar_martillos.py, pero bidireccional: un sentido suma y
el otro resta).

  - En el editor web (boton "Zonas") dibujas UNA LINEA de 2 puntos
    atravesando la puerta.
  - Cada persona trackeada (YOLO + ByteTrack) tiene su centroide.
  - Cuando el segmento (centroide anterior → centroide actual) CRUZA la
    linea, se cuenta segun el lado hacia el que cruzo:
        un lado  = ENTRADA (sube)
        el otro  = SALIDA  (baja)
    (--invert cambia el sentido sin redibujar)
  - DENTRO = inicial + entradas - salidas. El punto de partida se fija en
    la tarjeta ("Hay N personas dentro ahora → Fijar y contar") y todo se
    persiste en counters/<fuente>.json.

YOLO NO ve el frame completo: solo un recorte generoso alrededor de la
linea (usa --full-frame para desactivarlo).

USO (dentro de Panel_Web.py, o solo):
  uv run --with flask,ultralytics,torchvision,lap python Panel_Web.py \
         --port 8030 --zones-json ../zones.json --scene outdoor --thr 0.65
"""

import json
import os
import os.path as osp
import sys
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# =============================================================================
# PATHS
# =============================================================================
SCRIPT_DIR   = osp.dirname(osp.realpath(__file__))            # .../areaRest/NewModels
AREAREST_DIR = osp.normpath(osp.join(SCRIPT_DIR, ".."))
JT_DIR       = osp.normpath(osp.join(SCRIPT_DIR, "..", ".."))
BASE_DIR     = osp.join(JT_DIR, "Base")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SCRIPT_DIR)

from base_detector import BaseDetector  # noqa: E402
from face_bubble import FaceBubble  # noqa: E402
from params_mixin import ParamsMixin  # noqa: E402

DEFAULT_MODEL = osp.join(AREAREST_DIR, "yolo11x.pt")

# =============================================================================
# CONFIG
# =============================================================================
# ByteTrack afinado para puerta (track_buffer alto → re-asocia tras oclusion)
_TRACKER_LOCAL = osp.join(osp.dirname(osp.realpath(__file__)),
                          "bytetrack_contador.yaml")
TRACKER_YAML   = _TRACKER_LOCAL if osp.isfile(_TRACKER_LOCAL) else "bytetrack.yaml"
DETECT_CONF    = 0.25
DETECT_CLASSES = [0]          # persona (COCO)

# Recorte alrededor de la linea (px nativos): margen lateral y vertical
# para que quepan las personas completas antes y despues de cruzar.
CROP_PAD = 300
CROP_MIN = 480                # lado minimo del recorte

# Frames sin ver a un track antes de olvidar su centroide anterior
TRACK_TTL_FRAMES = 60
# Frames minimos entre dos cruces del mismo track (anti-rebote en la linea)
CROSS_COOLDOWN_FRAMES = 15

# ── Anti falsos positivos por oclusion ─────────────────────────────────────
# Un track debe llevar vivo N frames antes de poder contar (ids recien
# nacidos sobre la linea no cuentan).
MIN_TRACK_AGE = 5
# Salto maximo del centroide entre frames = factor * altura del bbox.
# Mas que eso = ByteTrack re-asocio tras oclusion (teletransporte): NO se
# evalua cruce con ese segmento.
MAX_JUMP_FACTOR = 1.5
# ── Cruce inferido (fantasmas): track perdido cerca de la linea en un lado
#    + track nuevo cerca de la linea en el lado contrario = cruce ocluido ──
GHOST_TTL      = 45    # frames de vida del fantasma
GHOST_NEAR_PX  = 260   # distancia max del punto a la linea
GHOST_MATCH_PX = 420   # distancia max entre punto perdido y reaparicion

# ── Evidencia por evento: foto + crop + metadata en EVENTOS_CONTADOR/ ──────
EVENTS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), "EVENTOS_CONTADOR")
CROP_MARGIN = 0.20     # margen extra alrededor de la persona en el crop

COLOR_LINE   = (0, 0, 230)       # rojo — linea de conteo
COLOR_TRACK  = (0, 200, 255)
COLOR_IN     = (0, 200, 50)      # verde
COLOR_OUT    = (0, 50, 220)      # rojo


def _dir(a, b, c) -> int:
    """Orientacion del triplete (signo del producto cruz) — contar_martillos."""
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 1 if v > 0 else (-1 if v < 0 else 0)


def cruza(p1, p2, a, b) -> bool:
    """True si el segmento p1->p2 cruza el segmento a->b (la linea dibujada)."""
    d1 = _dir(a, b, p1)
    d2 = _dir(a, b, p2)
    d3 = _dir(p1, p2, a)
    d4 = _dir(p1, p2, b)
    return d1 != d2 and d3 != d4


def centroid(bbox) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _pt_seg_dist(p, a, b) -> float:
    """Distancia de un punto al segmento a-b."""
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 <= 1e-9:
        return ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / l2))
    qx, qy = ax + t * dx, ay + t * dy
    return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5


# =============================================================================
# DETECTOR
# =============================================================================

class PeopleCounterWebDetector(ParamsMixin, BaseDetector):
    NAME = "CONTADOR_PERSONAS"

    # Parametros ajustables desde el panel (boton "Ajustes" de la tarjeta)
    PARAMS = [
        dict(key="det_conf", label="Confianza YOLO", type="float",
             min=0.05, max=0.9, step=0.01,
             help="Umbral minimo para aceptar una deteccion de persona"),
        dict(key="invert", label="Invertir sentido", type="bool",
             help="Lo que sumaba (ENTRADA) ahora resta (SALIDA)"),
        dict(key="min_track_age", label="Edad min. del track (frames)",
             type="int", min=0, max=60, step=1,
             help="Frames vivos antes de poder contar (anti falsos +)"),
        dict(key="cross_cooldown", label="Cooldown de cruce (frames)",
             type="int", min=0, max=120, step=1,
             help="Frames minimos entre dos cruces del mismo track"),
    ]

    # ------------------------------------------------------------------ #
    def add_arguments(self, parser):
        parser.add_argument("--model", default=DEFAULT_MODEL,
                            help=f"Pesos YOLO (default: {DEFAULT_MODEL})")
        parser.add_argument("--invert", action="store_true",
                            help="Invierte el sentido: lo que sumaba ahora resta")
        parser.add_argument("--full-frame", action="store_true",
                            help="YOLO ve el frame completo (sin recorte)")
        parser.add_argument("--zone-expand", type=int, default=0,
                            help="(sin uso en el contador por linea)")

    # ------------------------------------------------------------------ #
    def configure(self, args):
        """Sincrono (antes del worker): estado minimo para que los endpoints
        del panel no fallen mientras setup() carga el modelo."""
        super().configure(args)
        self._zones_lock = threading.Lock()
        self._pending_regions = None
        self.regions_orig: List[List[List[int]]] = []
        self.zones_error = "cargando..."
        self.zone_expand = 0          # compat con el editor (no se usa)
        self.full_frame  = False      # px del editor = px nativos
        self.count_in = self.count_out = self.offset = 0
        # ajustables en caliente desde el panel (por fuente, no en args
        # porque args se comparte entre todas las fuentes)
        self.det_conf       = DETECT_CONF
        self.invert         = bool(args.invert)
        self.min_track_age  = MIN_TRACK_AGE
        self.cross_cooldown = CROSS_COOLDOWN_FRAMES

    # ------------------------------------------------------------------ #
    def setup(self):
        from ultralytics import YOLO
        print(f"[{self.NAME}] Cargando YOLO {osp.basename(self.args.model)} ...")
        self.model = YOLO(self.args.model)

        # Linea por fuente (mismas rutas que el editor de web_server_zones)
        skey = getattr(self, "source_key", "")
        if skey:
            p = osp.join(SCRIPT_DIR, "zones_web", f"{skey}.json")
            if osp.isfile(p):
                try:
                    with open(p) as f:
                        self.regions_orig = json.load(f).get("regions", [])
                    print(f"[{self.NAME}] Linea cargada de {osp.basename(p)}")
                except Exception as e:
                    print(f"[{self.NAME}] Error leyendo linea: {e}")
        self._check_zones()

        # Estado del conteo por cruce de linea
        self.prev_pt:    Dict[int, Tuple[float, float]] = {}  # tid -> centroide previo
        self.last_seen:  Dict[int, int] = {}                  # tid -> ultimo frame
        self.last_cross: Dict[int, int] = {}                  # tid -> frame del ultimo cruce
        self.first_seen: Dict[int, int] = {}                  # tid -> primer frame
        self.ghosts:     List[dict] = []   # tracks perdidos cerca de la linea
        self.faces = FaceBubble()          # rostro por track (nube en video)
        self._load_counts()   # persistencia: sobrevive reinicios

        self._scaled = False
        self._crop_rect = None
        self._line: Optional[Tuple] = None    # ((ax,ay),(bx,by))
        self._draw = {"dets": []}

    # ------------------------------------------------------------------ #
    def _check_zones(self):
        """La 'zona' del contador es UNA LINEA: 1 region de exactamente 2 puntos."""
        self.zones_error = ""
        if not self.regions_orig:
            self.zones_error = "SIN LINEA: dibuja una linea de 2 puntos con ✎ Zonas"
        elif len(self.regions_orig) != 1 or len(self.regions_orig[0]) != 2:
            self.zones_error = (f"DIBUJA UNA SOLA LINEA de 2 puntos "
                                f"(hay {len(self.regions_orig)} figura(s) de "
                                f"{[len(r) for r in self.regions_orig]} puntos)")
        if self.zones_error:
            print(f"[{self.NAME}] {self.zones_error}")

    # ------------------------------------------------------------------ #
    #  Contadores: fijar "N personas dentro" + persistencia por fuente
    # ------------------------------------------------------------------ #
    def _counts_path(self) -> Optional[str]:
        skey = getattr(self, "source_key", "")
        if not skey:
            return None
        d = osp.join(SCRIPT_DIR, "counters")
        os.makedirs(d, exist_ok=True)
        return osp.join(d, f"{skey}.json")

    def _load_counts(self):
        p = self._counts_path()
        if p and osp.isfile(p):
            try:
                with open(p) as f:
                    d = json.load(f)
                self.count_in  = int(d.get("in", 0))
                self.count_out = int(d.get("out", 0))
                self.offset    = int(d.get("offset", 0))
                print(f"[{self.NAME}] Contadores restaurados: "
                      f"dentro={self.inside()} (in={self.count_in} "
                      f"out={self.count_out} inicial={self.offset})")
            except Exception as e:
                print(f"[{self.NAME}] Error leyendo contadores: {e}")

    def _save_counts(self):
        p = self._counts_path()
        if not p:
            return
        try:
            with open(p, "w") as f:
                json.dump({"in": self.count_in, "out": self.count_out,
                           "offset": self.offset}, f)
        except Exception as e:
            print(f"[{self.NAME}] Error guardando contadores: {e}")

    def inside(self) -> int:
        return self.offset + self.count_in - self.count_out

    def set_inside(self, n: int):
        """Fija cuantas personas hay dentro AHORA; in/out arrancan en 0."""
        self.count_in  = 0
        self.count_out = 0
        self.offset    = max(0, int(n))
        self._save_counts()
        print(f"[{self.NAME}] Contador fijado: hay {self.offset} persona(s) "
              f"dentro — contando desde ahi (in/out reiniciados)")

    def get_counts(self) -> dict:
        return {"inside": self.inside(), "in": self.count_in,
                "out": self.count_out, "offset": self.offset}

    # ------------------------------------------------------------------ #
    #  Linea en caliente (interfaz del editor de web_server_zones.py)
    # ------------------------------------------------------------------ #
    def set_zones(self, regions, zone_expand: Optional[int] = None,
                  opts: Optional[dict] = None):
        with self._zones_lock:
            self._pending_regions = [list(map(list, r)) for r in regions]

    def get_zones(self):
        with self._zones_lock:
            if self._pending_regions is not None:
                return self._pending_regions
        return self.regions_orig

    def _apply_zones(self, regions):
        self.regions_orig = regions
        self._scaled = False
        self.prev_pt.clear()      # trayectorias viejas ya no son validas
        self.last_cross.clear()
        self.first_seen.clear()
        self.ghosts.clear()
        self._check_zones()
        print(f"[{self.NAME}] Linea recargada "
              f"(contadores se conservan: IN={self.count_in} OUT={self.count_out})")

    # ------------------------------------------------------------------ #
    def _prepare_scale(self, frame):
        """Coords nativas. Recorte generoso alrededor de la linea."""
        H, W = frame.shape[:2]
        self._line = None
        self._crop_rect = None
        if not self.zones_error and self.regions_orig:
            (ax, ay), (bx, by) = self.regions_orig[0]
            self._line = ((int(ax), int(ay)), (int(bx), int(by)))
            if not self.args.full_frame:
                x1 = min(ax, bx) - CROP_PAD
                y1 = min(ay, by) - CROP_PAD
                x2 = max(ax, bx) + CROP_PAD
                y2 = max(ay, by) + CROP_PAD
                # lado minimo util
                if x2 - x1 < CROP_MIN:
                    cx = (x1 + x2) // 2
                    x1, x2 = cx - CROP_MIN // 2, cx + CROP_MIN // 2
                if y2 - y1 < CROP_MIN:
                    cy = (y1 + y2) // 2
                    y1, y2 = cy - CROP_MIN // 2, cy + CROP_MIN // 2
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(W, int(x2)), min(H, int(y2))
                if x2 - x1 >= 32 and y2 - y1 >= 32:
                    self._crop_rect = (x1, y1, x2, y2)
                    print(f"[{self.NAME}] YOLO solo analiza alrededor de la "
                          f"linea: {x2 - x1}x{y2 - y1}px de {W}x{H} "
                          f"({100 * (x2 - x1) * (y2 - y1) / (W * H):.0f}% del frame)")
        if self._crop_rect is None:
            print(f"[{self.NAME}] YOLO en frame completo {W}x{H}")
        self._scaled = True

    # ------------------------------------------------------------------ #
    #  Evidencia por evento: foto del frame + crop de la persona + metadata
    # ------------------------------------------------------------------ #
    def _events_dir(self) -> str:
        skey = getattr(self, "source_key", "") or "fuente"
        d = osp.join(EVENTS_DIR, skey)
        os.makedirs(d, exist_ok=True)
        return d

    def _save_event_evidence(self, ev: str, tid: int, frame, bbox,
                             inferido: bool) -> str:
        """Guarda foto completa (marcada), crop de la persona y metadata.
        Retorna el nombre base de los archivos."""
        try:
            now  = datetime.now()
            base = f"{now.strftime('%Y%m%d_%H%M%S')}_{ev}_id{tid}"
            d    = self._events_dir()

            # Foto completa con la linea, el bbox y la etiqueta del evento
            full = frame.copy()
            if self._line is not None:
                cv2.line(full, self._line[0], self._line[1], COLOR_LINE, 3)
            color = COLOR_IN if ev == "ENTRADA" else COLOR_OUT
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(full, (x1, y1), (x2, y2), color, 3)
            label = (f"{ev} id{tid} {now.strftime('%H:%M:%S')} "
                     f"dentro={self.inside()}")
            cv2.rectangle(full, (0, 0), (560, 42), (20, 20, 20), -1)
            cv2.putText(full, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)
            cv2.imwrite(osp.join(d, base + ".jpg"), full)

            # Crop de la persona (con margen)
            if bbox is not None:
                H, W = frame.shape[:2]
                x1, y1, x2, y2 = bbox
                mx = int((x2 - x1) * CROP_MARGIN)
                my = int((y2 - y1) * CROP_MARGIN)
                cx1, cy1 = max(0, x1 - mx), max(0, y1 - my)
                cx2, cy2 = min(W, x2 + mx), min(H, y2 + my)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    cv2.imwrite(osp.join(d, base + "_crop.jpg"), crop)

            # Rostro del track (nube) + nombre reconocido (AdaFace)
            face = self.faces.get_face(tid)
            if face is not None:
                cv2.imwrite(osp.join(d, base + "_rostro.jpg"), face)
            fname, fscore = self.faces.get_name(tid)

            # Metadata (una linea JSON por evento — facil de procesar)
            meta = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "evento":    ev,                  # ENTRADA / SALIDA
                "track_id":  tid,
                "inferido":  inferido,            # True = cruce por oclusion
                "dentro":    self.inside(),
                "entradas":  self.count_in,
                "salidas":   self.count_out,
                "foto":      base + ".jpg",
                "crop":      (base + "_crop.jpg") if bbox is not None else None,
                "rostro":    (base + "_rostro.jpg") if face is not None else None,
                "rostro_nombre": fname,
                "rostro_score":  round(fscore, 3) if fname else None,
                "fuente":    getattr(self, "source_key", ""),
            }
            with open(osp.join(d, "eventos.jsonl"), "a") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            return base
        except Exception as e:
            print(f"[{self.NAME}] Error guardando evidencia: {e}")
            return ""

    # ------------------------------------------------------------------ #
    #  Fantasmas: cruces ocultos por oclusion
    # ------------------------------------------------------------------ #
    def _count_event(self, entrada: bool, frame_idx: int, tid: int,
                     frame=None, bbox=None, inferido: bool = False):
        if self.invert:
            entrada = not entrada
        if entrada:
            self.count_in += 1
            ev = "ENTRADA"
        else:
            self.count_out += 1
            ev = "SALIDA"
        self.last_cross[tid] = frame_idx
        self._save_counts()
        base = ""
        if frame is not None:
            base = self._save_event_evidence(ev, tid, frame, bbox, inferido)
        tag = "  [INFERIDO por oclusion]" if inferido else ""
        foto = f"  foto={base}.jpg" if base else ""
        print(f"[{self.NAME}][F{frame_idx:05d}] {ev}  id={tid}  "
              f"→ dentro={self.inside()} "
              f"(in={self.count_in} out={self.count_out}){tag}{foto}")

    def _add_ghost(self, pt, frame_idx: int):
        """Track perdido: si estaba cerca de la linea, recordarlo un rato."""
        if self._line is None or pt is None:
            return
        a, b = self._line
        if _pt_seg_dist(pt, a, b) <= GHOST_NEAR_PX:
            side = _dir(a, b, pt)
            if side != 0:
                self.ghosts.append({"pt": pt, "side": side,
                                    "frame": frame_idx})

    def _try_ghost_match(self, tid: int, pt, frame_idx: int,
                         frame=None, bbox=None):
        """Track NUEVO cerca de la linea: ¿es alguien que se perdio por
        oclusion en el lado contrario? → contar el cruce que no vimos."""
        if self._line is None:
            return
        a, b = self._line
        if _pt_seg_dist(pt, a, b) > GHOST_NEAR_PX:
            return
        side = _dir(a, b, pt)
        if side == 0:
            return
        for g in self.ghosts:
            if (g["side"] != side
                    and frame_idx - g["frame"] <= GHOST_TTL
                    and ((g["pt"][0] - pt[0]) ** 2
                         + (g["pt"][1] - pt[1]) ** 2) ** 0.5 <= GHOST_MATCH_PX):
                self.ghosts.remove(g)
                self._count_event(entrada=(side > 0), frame_idx=frame_idx,
                                  tid=tid, frame=frame, bbox=bbox,
                                  inferido=True)
                return

    # ------------------------------------------------------------------ #
    def process_frame(self, frame, frame_idx):
        with self._zones_lock:
            pending = self._pending_regions
            self._pending_regions = None
        if pending is not None:
            self._apply_zones(pending)

        if not self._scaled:
            self._prepare_scale(frame)

        if self._crop_rect is not None:
            cx1, cy1, cx2, cy2 = self._crop_rect
            proc = frame[cy1:cy2, cx1:cx2]
            ox, oy = cx1, cy1
        else:
            proc = frame
            ox = oy = 0

        res = self.model.track(proc, classes=DETECT_CLASSES, persist=True,
                               verbose=False, tracker=TRACKER_YAML)[0]

        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            ids  = (res.boxes.id.cpu().numpy().astype(int)
                    if res.boxes.id is not None
                    else np.full(len(xyxy), -1, dtype=int))
            for (x1, y1, x2, y2), c, tid in zip(xyxy, conf, ids):
                if c < self.det_conf:
                    continue
                dets.append(dict(bbox=(int(x1) + ox, int(y1) + oy,
                                       int(x2) + ox, int(y2) + oy),
                                 conf=float(c), track_id=int(tid)))

        do_diag = (frame_idx % 30 == 1)

        for d in dets:
            tid = d["track_id"]
            if tid < 0:
                continue
            self.last_seen[tid] = frame_idx

            actual = centroid(d["bbox"])
            previo = self.prev_pt.get(tid)
            bbox_h = max(d["bbox"][3] - d["bbox"][1], 1)

            # Rostro del track (para la nube y la evidencia)
            self.faces.update(tid, frame, d["bbox"], frame_idx)

            if tid not in self.first_seen:
                # id NUEVO: ¿reaparicion de alguien ocluido del otro lado?
                self.first_seen[tid] = frame_idx
                self._try_ghost_match(tid, actual, frame_idx,
                                      frame=frame, bbox=d["bbox"])

            if self._line is not None and previo is not None:
                a, b = self._line
                jump = ((actual[0] - previo[0]) ** 2
                        + (actual[1] - previo[1]) ** 2) ** 0.5
                if jump > MAX_JUMP_FACTOR * bbox_h:
                    # Teletransporte (re-asociacion tras oclusion): NO evaluar
                    # cruce con este segmento; el punto viejo pasa a fantasma.
                    self._add_ghost(previo, frame_idx)
                    self._try_ghost_match(tid, actual, frame_idx,
                                          frame=frame, bbox=d["bbox"])
                    if do_diag:
                        print(f"[{self.NAME}][F{frame_idx:05d}] diag id={tid} "
                              f"salto {int(jump)}px (> {MAX_JUMP_FACTOR}x"
                              f"h={bbox_h}) → cruce ignorado (oclusion)")
                else:
                    aged   = (frame_idx - self.first_seen[tid]
                              >= self.min_track_age)
                    cooled = (frame_idx - self.last_cross.get(tid, -10**9)
                              >= self.cross_cooldown)
                    if aged and cooled and cruza(previo, actual, a, b):
                        # Lado hacia el que quedo el centroide = direccion
                        self._count_event(entrada=_dir(a, b, actual) > 0,
                                          frame_idx=frame_idx, tid=tid,
                                          frame=frame, bbox=d["bbox"])

            if do_diag and self._line is not None:
                lado = _dir(self._line[0], self._line[1], actual)
                print(f"[{self.NAME}][F{frame_idx:05d}] diag id={tid} "
                      f"centroide=({int(actual[0])},{int(actual[1])}) "
                      f"lado={'+' if lado > 0 else '-' if lado < 0 else '0'} "
                      f"edad={frame_idx - self.first_seen[tid]}")

            self.prev_pt[tid] = actual

        # Olvidar tracks perdidos hace rato (si murieron cerca de la linea,
        # quedan como fantasma por si reaparecen del otro lado)
        for tid in list(self.last_seen):
            if frame_idx - self.last_seen[tid] > TRACK_TTL_FRAMES:
                self._add_ghost(self.prev_pt.get(tid), self.last_seen[tid])
                self.last_seen.pop(tid, None)
                self.prev_pt.pop(tid, None)
                self.last_cross.pop(tid, None)
                self.first_seen.pop(tid, None)

        # Expirar fantasmas viejos
        self.ghosts = [g for g in self.ghosts
                       if frame_idx - g["frame"] <= GHOST_TTL]
        self.faces.purge(list(self.last_seen))

        self._draw = {"dets": dets}

        if self.zones_error:
            extra = self.zones_error
        elif self._crop_rect is not None:
            cw = self._crop_rect[2] - self._crop_rect[0]
            ch = self._crop_rect[3] - self._crop_rect[1]
            extra = f"personas: {len(dets)} | YOLO en {cw}x{ch}px (solo linea)"
        else:
            extra = f"personas: {len(dets)}"

        return {
            "alert": False,
            "label": "",
            "score": None,
            "boxes": [],
            "extra": extra,
            "inside": self.inside(),
        }

    # ------------------------------------------------------------------ #
    def annotate(self, frame, result, stats):
        out = frame.copy()
        h, w = out.shape[:2]
        dets = (getattr(self, "_draw", None) or {}).get("dets", [])

        # Exponer conteos en la API (GET /api/sources incluye estos campos)
        if isinstance(stats, dict):
            stats.update(self.get_counts())

        # ── Area de analisis YOLO (recorte, gris) ──────────────────────
        if self._crop_rect is not None:
            cx1, cy1, cx2, cy2 = self._crop_rect
            cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (140, 140, 140), 1)
            cv2.putText(out, "area de analisis (YOLO)",
                        (cx1 + 4, max(70, cy1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 170, 170), 1)

        # ── Linea de conteo + etiquetas de sentido ─────────────────────
        if self._line is not None:
            a, b = self._line
            cv2.line(out, a, b, COLOR_LINE, 3)
            cv2.circle(out, a, 7, COLOR_LINE, -1)
            cv2.circle(out, b, 7, COLOR_LINE, -1)
            # normal de la linea → de que lado esta ENTRADA
            mx, my = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            nx, ny = -dy / n * 55, dx / n * 55   # lado positivo de _dir
            in_lbl, out_lbl = ("ENTRA", "SALE")
            if self.invert:
                in_lbl, out_lbl = out_lbl, in_lbl
            cv2.putText(out, in_lbl, (int(mx + nx) - 30, int(my + ny)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_IN, 2)
            cv2.putText(out, out_lbl, (int(mx - nx) - 30, int(my - ny)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_OUT, 2)

        # ── Tracks: bbox + centroide + colita de trayectoria ───────────
        for d in dets:
            tid = d["track_id"]
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_TRACK, 2)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(out, (cx, cy), 7, COLOR_LINE, -1)
            cv2.circle(out, (cx, cy), 8, (255, 255, 255), 1)
            prev = self.prev_pt.get(tid)
            if prev is not None:
                cv2.line(out, (int(prev[0]), int(prev[1])), (cx, cy),
                         (255, 255, 255), 2)
            cv2.putText(out, f"id{tid}", (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TRACK, 2)
            # Nube con el rostro ampliado
            self.faces.draw_bubble(out, tid, d["bbox"])

        # ── Banner con contadores ──────────────────────────────────────
        cv2.rectangle(out, (0, 0), (w, 56), (30, 30, 30), -1)
        cv2.putText(out, f"ENTRARON: {self.count_in}", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_IN, 2)
        cv2.putText(out, f"SALIERON: {self.count_out}", (w // 3, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_OUT, 2)
        cv2.putText(out, f"DENTRO: {self.inside()}", (2 * w // 3, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        self.draw_infobar(out, stats, result.get("extra", ""))
        return out

    # ------------------------------------------------------------------ #
    def teardown(self):
        self.model = None
        self.prev_pt.clear()
        self.last_seen.clear()
        self.last_cross.clear()
        self.first_seen.clear()
        self.ghosts.clear()


# =============================================================================
# ENTRY POINT — panel web con editor de zonas/linea
# =============================================================================

if __name__ == "__main__":
    from web_server_zones import main_web_zones
    main_web_zones(PeopleCounterWebDetector)
