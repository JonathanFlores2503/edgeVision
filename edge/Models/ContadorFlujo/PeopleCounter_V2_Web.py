
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PeopleCounter_V2_Web.py
=======================
Contador de personas V2 — CORREDOR DE CONTEO (diseño de DISENO_Contador.html).
La V1 (PeopleCounter_Web.py, linea simple) queda intacta; esta version la
refina con el metodo del box + escalera de lineas:

  - Dibujas UN BOX de 4 puntos atravesando la puerta (editor web).
    El PRIMER lado que dibujas (clicks 1-2) = AFUERA. --invert lo voltea.
  - Dentro se genera una ESCALERA de N_LINES lineas internas.
  - Cada track tiene su PROGRESO p (0-100%) a lo largo del eje
    AFUERA→ADENTRO. Las lineas son umbrales: brincarse una entre frames
    cuenta igual (no depende del instante exacto del cruce).

REGLAS DE DECISION (R1-R5):
  R1  Mayoria: cruzo 50%+1 de las lineas con direccion neta → cuenta.
  R2  Salida del box: si salio por el lado contrario al que entro y no
      habia contado → cuenta por lado de salida.
  R3  Track perdido dentro del box: con mayoria → cuenta (inferido);
      sin mayoria → queda PENDIENTE (fantasma).
  R4  Id nuevo dentro del box: hereda el progreso de un fantasma
      compatible (cruce partido en dos ids por oclusion).
  R5  Se asoma y regresa por donde vino → NO cuenta.

GUARDIAN DE LUZ:
  brillo promedio del corredor (0-255, suavizado):
    > LUZ_BAJA   NORMAL
    LUZ_OSCURO..LUZ_BAJA  BAJA LUZ → conf reducida + CLAHE
    < LUZ_OSCURO OSCURO → ALERTA "sin luz suficiente" + conteo EN PAUSA

Se conserva de la V1: contador persistente + "Fijar N dentro", evidencia por
evento (foto + crop + rostro con nombre + jsonl), nube de rostro, recorte de
analisis, ByteTrack afinado.

USO (dentro de Panel_Web.py — aparece como CONTADOR_V2 en el selector).
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
SCRIPT_DIR   = osp.dirname(osp.realpath(__file__))
AREAREST_DIR = osp.normpath(osp.join(SCRIPT_DIR, ".."))
JT_DIR       = osp.normpath(osp.join(SCRIPT_DIR, "..", ".."))
BASE_DIR     = osp.join(JT_DIR, "Base")

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, SCRIPT_DIR)

from base_detector import BaseDetector  # noqa: E402
from face_bubble import FaceBubble  # noqa: E402
from params_mixin import ParamsMixin  # noqa: E402

DEFAULT_MODEL = osp.join(AREAREST_DIR, "yolo11x.pt")
_TRACKER_LOCAL = osp.join(SCRIPT_DIR, "bytetrack_contador.yaml")
TRACKER_YAML = _TRACKER_LOCAL if osp.isfile(_TRACKER_LOCAL) else "bytetrack.yaml"

EVENTS_DIR = osp.join(SCRIPT_DIR, "EVENTOS_CONTADOR")

# ═══════════════════════════════════════════════════════════════════════════
#  VARIABLES DE PRUEBA (debug)
# ═══════════════════════════════════════════════════════════════════════════
# ── Corredor ───────────────────────────────────────────────────────────────
N_LINES = 5           # lineas internas de la escalera (mayoria = N//2+1)
                      #   ↑ mas lineas = mas evidencia requerida (mas estricto)
DETECT_CLASSES = [0]  # persona (COCO)
IMGSZ = 256           # tamaño de entrada YOLO (256/320 = mucho mas rapido en CPU que 640; edge sin GPU)
CONF_NORMAL   = 0.25  # confianza YOLO con luz normal
CONF_BAJA_LUZ = 0.15  # confianza YOLO en baja luz (detecta en penumbra)

# ── Guardian de luz (brillo 0-255 del corredor, suavizado EMA) ────────────
LUZ_OSCURO = 35       # debajo: ALERTA + conteo en pausa
LUZ_BAJA   = 70       # debajo: aviso + conf reducida + CLAHE
LUZ_EMA    = 0.05     # suavizado (~2 s a 15-20 fps)

# ── Tracks / fantasmas (R3-R4) ─────────────────────────────────────────────
MIN_TRACK_AGE   = 3    # frames vivos antes de poder contar
TRACK_TTL       = 60   # frames sin verlo → track perdido (dispara R3)
PENDING_TTL     = 60   # frames que vive un fantasma esperando heredero (R4)
PENDING_MATCH_P = 0.35 # distancia max de progreso |p_fantasma - p_nuevo|

# ── Recorte de analisis alrededor del box ──────────────────────────────────
CROP_PAD    = 200      # margen lateral px
CROP_TOP_X  = 1.0      # margen superior extra = este factor * alto del box
CROP_MIN    = 480      # lado minimo del recorte
# ═══════════════════════════════════════════════════════════════════════════

# ── Paleta (mejora de colores V2) — BGR ────────────────────────────────────
C_BOX_BORDE   = (235, 235, 235)   # blanco suave
C_BOX_FILL    = (180, 120,  40)   # azul acero (con alpha)
C_ESCALERA    = (200, 170, 120)   # azul claro tenue
C_AFUERA      = (  0, 190, 255)   # ambar
C_ADENTRO     = ( 90, 200,  60)   # verde
C_PERSONA     = (255, 210,  80)   # cyan claro
C_PROG_IN     = ( 90, 200,  60)   # barra progreso entrando
C_PROG_OUT    = ( 60,  80, 230)   # barra progreso saliendo
C_TXT         = (245, 245, 245)
C_LUZ = {"NORMAL": (90, 200, 60), "BAJA": (0, 190, 255),
         "OSCURO": (60, 60, 230)}


class PeopleCounterV2Detector(ParamsMixin, BaseDetector):
    NAME = "CONTADOR_V2"

    # Parametros ajustables desde el panel (boton "Ajustes" de la tarjeta)
    # (invert / n_lines / rotar ya se ajustan en el editor de Zonas)
    PARAMS = [
        dict(key="conf_normal", label="Confianza YOLO (luz normal)",
             type="float", min=0.05, max=0.9, step=0.01),
        dict(key="conf_baja", label="Confianza YOLO (baja luz)",
             type="float", min=0.05, max=0.9, step=0.01,
             help="Se usa cuando la luz cae debajo del umbral BAJA"),
        dict(key="min_track_age", label="Edad min. del track (frames)",
             type="int", min=0, max=60, step=1,
             help="Frames vivos antes de poder contar (anti falsos +)"),
    ]

    # ------------------------------------------------------------------ #
    def add_arguments(self, parser):
        parser.add_argument("--model", default=DEFAULT_MODEL,
                            help=f"Pesos YOLO (default: {DEFAULT_MODEL})")
        parser.add_argument("--invert", action="store_true",
                            help="Voltea AFUERA/ADENTRO sin redibujar el box")
        parser.add_argument("--full-frame", action="store_true",
                            help="YOLO ve el frame completo (sin recorte)")
        parser.add_argument("--zone-expand", type=int, default=0,
                            help="(sin uso en V2)")

    # ------------------------------------------------------------------ #
    def configure(self, args):
        super().configure(args)
        self._zones_lock = threading.Lock()
        self._pending_regions = None
        self.regions_orig: List[List[List[int]]] = []
        self.zones_error = "cargando..."
        self.zone_expand = 0          # compat editor
        self.full_frame  = False      # px del editor = px nativos
        self.count_in = self.count_out = self.offset = 0
        # Opciones del corredor (ajustables desde el editor web, por fuente)
        self.n_lines   = N_LINES      # lineas internas de la escalera
        self.axis_rot  = False        # rotar eje 90° (orientacion de lineas)
        self.invert_ui = False        # voltear AFUERA/ADENTRO desde la web
        # ajustables en caliente desde el panel (por fuente)
        self.conf_normal   = CONF_NORMAL
        self.conf_baja     = CONF_BAJA_LUZ
        self.min_track_age = MIN_TRACK_AGE

    def _inv(self) -> bool:
        """Inversion efectiva: flag CLI XOR ajuste del editor web."""
        return bool(self.args.invert) != bool(self.invert_ui)

    def corridor_opts(self) -> dict:
        """Para el editor web (GET zones): opciones actuales del corredor."""
        return {"corridor": True, "n_lines": self.n_lines,
                "invert": self.invert_ui, "axis_rot": self.axis_rot}

    # ------------------------------------------------------------------ #
    def setup(self):
        from ultralytics import YOLO
        print(f"[{self.NAME}] Cargando YOLO {osp.basename(self.args.model)} ...")
        self.model = YOLO(self.args.model)

        skey = getattr(self, "source_key", "")
        if skey:
            p = osp.join(SCRIPT_DIR, "zones_web", f"{skey}.json")
            if osp.isfile(p):
                try:
                    with open(p) as f:
                        data = json.load(f)
                    self.regions_orig = data.get("regions", [])
                    self._apply_opts(data)
                    print(f"[{self.NAME}] Box cargado de {osp.basename(p)}")
                except Exception as e:
                    print(f"[{self.NAME}] Error leyendo box: {e}")
        self._check_zones()

        # Estado del corredor
        # tid -> {"p_start","p","inside","counted","born","seen","heredado"}
        self.tracks: Dict[int, dict] = {}
        self.pendings: List[dict] = []       # fantasmas R3/R4
        self.faces = FaceBubble()
        self._load_counts()

        # Guardian de luz
        self.luz = 128.0
        self.luz_estado = "NORMAL"
        self._clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

        self._scaled = False
        self._crop_rect = None
        self._geo = None                     # geometria del corredor
        self._draw = {"dets": []}

    # ------------------------------------------------------------------ #
    def _check_zones(self):
        self.zones_error = ""
        if not self.regions_orig:
            self.zones_error = "SIN CORREDOR: dibuja un box de 4 puntos con ✎ Zonas"
        elif len(self.regions_orig) != 1 or len(self.regions_orig[0]) != 4:
            self.zones_error = (f"DIBUJA UN SOLO BOX de 4 puntos "
                                f"(hay {len(self.regions_orig)} figura(s) de "
                                f"{[len(r) for r in self.regions_orig]} puntos)")
        if self.zones_error:
            print(f"[{self.NAME}] {self.zones_error}")

    # ------------------------------------------------------------------ #
    #  Contadores persistentes (igual que V1)
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
        self.count_in = 0
        self.count_out = 0
        self.offset = max(0, int(n))
        self._save_counts()
        print(f"[{self.NAME}] Contador fijado: hay {self.offset} persona(s) "
              f"dentro — contando desde ahi (in/out reiniciados)")

    def get_counts(self) -> dict:
        return {"inside": self.inside(), "in": self.count_in,
                "out": self.count_out, "offset": self.offset}

    # ------------------------------------------------------------------ #
    #  Box en caliente (editor web)
    # ------------------------------------------------------------------ #
    def set_zones(self, regions, zone_expand: Optional[int] = None,
                  opts: Optional[dict] = None):
        with self._zones_lock:
            self._pending_regions = ([list(map(list, r)) for r in regions],
                                     opts or {})

    def get_zones(self):
        with self._zones_lock:
            if self._pending_regions is not None:
                return self._pending_regions[0]
        return self.regions_orig

    def _apply_opts(self, opts: dict):
        if "n_lines" in opts:
            self.n_lines = max(1, min(9, int(opts["n_lines"])))
        if "invert" in opts:
            self.invert_ui = bool(opts["invert"])
        if "axis_rot" in opts:
            self.axis_rot = bool(opts["axis_rot"])

    def _apply_zones(self, regions, opts: Optional[dict] = None):
        self.regions_orig = regions
        if opts:
            self._apply_opts(opts)
        self._scaled = False
        self.tracks.clear()
        self.pendings.clear()
        self._check_zones()
        print(f"[{self.NAME}] Corredor recargado: {self.n_lines} lineas, "
              f"eje {'ROTADO 90°' if self.axis_rot else 'normal'}, "
              f"{'INVERTIDO' if self._inv() else 'normal'} "
              f"(contadores se conservan: IN={self.count_in} "
              f"OUT={self.count_out})")

    # ------------------------------------------------------------------ #
    #  Geometria del corredor
    # ------------------------------------------------------------------ #
    def _prepare_scale(self, frame):
        H, W = frame.shape[:2]
        self._geo = None
        self._crop_rect = None
        if not self.zones_error and self.regions_orig:
            box = np.array(self.regions_orig[0], dtype=np.float32)
            # eje AFUERA→ADENTRO: del centro del lado 1 (clicks 1-2) al
            # centro del lado opuesto. Con axis_rot se usa el otro par de
            # lados (por si el box se dibujo empezando por el lado "largo"
            # y la escalera quedo paralela al paso).
            if self.axis_rot:
                o = (box[1] + box[2]) / 2.0
                e = (box[3] + box[0]) / 2.0
            else:
                o = (box[0] + box[1]) / 2.0
                e = (box[2] + box[3]) / 2.0
            v = e - o
            l2 = float(v @ v)
            if l2 > 1:
                n = self.n_lines
                self._geo = {"poly": box.astype(np.int32), "o": o, "v": v,
                             "l2": l2, "box": box,
                             "thr": [i / (n + 1) for i in range(1, n + 1)],
                             "mayoria": n // 2 + 1}
                print(f"[{self.NAME}] Corredor listo: {n} lineas, "
                      f"mayoria={self._geo['mayoria']}, "
                      f"eje AFUERA→ADENTRO {int(np.hypot(*v))}px"
                      + ("  [eje ROTADO 90°]" if self.axis_rot else "")
                      + ("  [INVERTIDO]" if self._inv() else ""))
            if self._geo is not None and not self.args.full_frame:
                bh = int(box[:, 1].max() - box[:, 1].min())
                x1 = int(box[:, 0].min()) - CROP_PAD
                y1 = int(box[:, 1].min()) - CROP_PAD - int(CROP_TOP_X * bh)
                x2 = int(box[:, 0].max()) + CROP_PAD
                y2 = int(box[:, 1].max()) + CROP_PAD
                if x2 - x1 < CROP_MIN:
                    cx = (x1 + x2) // 2
                    x1, x2 = cx - CROP_MIN // 2, cx + CROP_MIN // 2
                if y2 - y1 < CROP_MIN:
                    cy = (y1 + y2) // 2
                    y1, y2 = cy - CROP_MIN // 2, cy + CROP_MIN // 2
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 >= 32 and y2 - y1 >= 32:
                    self._crop_rect = (x1, y1, x2, y2)
                    print(f"[{self.NAME}] YOLO analiza {x2-x1}x{y2-y1}px de "
                          f"{W}x{H} ({100*(x2-x1)*(y2-y1)/(W*H):.0f}% del frame)")
        if self._crop_rect is None:
            print(f"[{self.NAME}] YOLO en frame completo {W}x{H}")
        self._scaled = True

    def _progress(self, pt) -> float:
        g = self._geo
        d = (np.array(pt, dtype=np.float32) - g["o"]) @ g["v"] / g["l2"]
        return float(min(1.0, max(0.0, d)))

    def _inside_box(self, pt) -> bool:
        return cv2.pointPolygonTest(self._geo["poly"],
                                    (float(pt[0]), float(pt[1])), False) >= 0

    def _lines_between(self, p_a: float, p_b: float) -> int:
        lo, hi = min(p_a, p_b), max(p_a, p_b)
        return sum(1 for t in self._geo["thr"] if lo < t <= hi)

    # ------------------------------------------------------------------ #
    #  Conteo + evidencia
    # ------------------------------------------------------------------ #
    def _count(self, hacia_adentro: bool, tid: int, frame_idx: int,
               regla: str, lineas: str, frame=None, bbox=None):
        entrada = hacia_adentro
        if self._inv():
            entrada = not entrada
        if entrada:
            self.count_in += 1
            ev = "ENTRADA"
        else:
            self.count_out += 1
            ev = "SALIDA"
        self._save_counts()
        base = ""
        if frame is not None:
            base = self._save_evidence(ev, tid, frame, bbox, regla, lineas)
        print(f"[{self.NAME}][F{frame_idx:05d}] {ev}  id={tid}  [{regla} "
              f"{lineas}]  → dentro={self.inside()} "
              f"(in={self.count_in} out={self.count_out})"
              + (f"  foto={base}.jpg" if base else ""))

    def _save_evidence(self, ev, tid, frame, bbox, regla, lineas) -> str:
        try:
            now = datetime.now()
            base = f"{now.strftime('%Y%m%d_%H%M%S')}_{ev}_id{tid}"
            d = osp.join(EVENTS_DIR, getattr(self, "source_key", "fuente"))
            os.makedirs(d, exist_ok=True)

            full = frame.copy()
            if self._geo is not None:
                cv2.polylines(full, [self._geo["poly"]], True, C_BOX_BORDE, 2)
            color = C_PROG_IN if ev == "ENTRADA" else C_PROG_OUT
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(full, (x1, y1), (x2, y2), color, 3)
            label = (f"{ev} id{tid} [{regla} {lineas}] "
                     f"{now.strftime('%H:%M:%S')} dentro={self.inside()}")
            cv2.rectangle(full, (0, 0), (720, 42), (20, 20, 20), -1)
            cv2.putText(full, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2)
            cv2.imwrite(osp.join(d, base + ".jpg"), full)

            if bbox is not None:
                H, W = frame.shape[:2]
                x1, y1, x2, y2 = bbox
                mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
                crop = frame[max(0, y1-my):min(H, y2+my),
                             max(0, x1-mx):min(W, x2+mx)]
                if crop.size > 0:
                    cv2.imwrite(osp.join(d, base + "_crop.jpg"), crop)

            face = self.faces.get_face(tid)
            if face is not None:
                cv2.imwrite(osp.join(d, base + "_rostro.jpg"), face)
            fname, fscore = self.faces.get_name(tid)

            meta = {
                "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                "evento": ev, "track_id": tid,
                "regla": regla, "lineas": lineas,
                "luz": round(self.luz, 1),
                "dentro": self.inside(),
                "entradas": self.count_in, "salidas": self.count_out,
                "foto": base + ".jpg",
                "crop": (base + "_crop.jpg") if bbox is not None else None,
                "rostro": (base + "_rostro.jpg") if face is not None else None,
                "rostro_nombre": fname,
                "rostro_score": round(fscore, 3) if fname else None,
                "fuente": getattr(self, "source_key", ""),
            }
            with open(osp.join(d, "eventos.jsonl"), "a") as f:
                f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            return base
        except Exception as e:
            print(f"[{self.NAME}] Error guardando evidencia: {e}")
            return ""

    def _log_luz_event(self, evento: str):
        try:
            d = osp.join(EVENTS_DIR, getattr(self, "source_key", "fuente"))
            os.makedirs(d, exist_ok=True)
            with open(osp.join(d, "eventos.jsonl"), "a") as f:
                f.write(json.dumps({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "evento": evento, "luz": round(self.luz, 1),
                    "fuente": getattr(self, "source_key", ""),
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    def process_frame(self, frame, frame_idx):
        with self._zones_lock:
            pending = self._pending_regions
            self._pending_regions = None
        if pending is not None:
            self._apply_zones(pending[0], pending[1])
        if not self._scaled:
            self._prepare_scale(frame)

        # ── Recorte ────────────────────────────────────────────────────
        if self._crop_rect is not None:
            cx1, cy1, cx2, cy2 = self._crop_rect
            proc = frame[cy1:cy2, cx1:cx2]
            ox, oy = cx1, cy1
        else:
            proc = frame
            ox = oy = 0

        # ── Guardian de luz ────────────────────────────────────────────
        gris = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
        self.luz = (1 - LUZ_EMA) * self.luz + LUZ_EMA * float(gris.mean())
        estado_ant = self.luz_estado
        self.luz_estado = ("OSCURO" if self.luz < LUZ_OSCURO
                           else "BAJA" if self.luz < LUZ_BAJA else "NORMAL")
        if self.luz_estado != estado_ant:
            if self.luz_estado == "OSCURO":
                print(f"[{self.NAME}][F{frame_idx:05d}] ⚠ SIN LUZ SUFICIENTE "
                      f"(brillo={self.luz:.0f}) — conteo EN PAUSA")
                self._log_luz_event("PAUSA_POR_LUZ")
            elif estado_ant == "OSCURO":
                print(f"[{self.NAME}][F{frame_idx:05d}] Luz recuperada "
                      f"(brillo={self.luz:.0f}) — conteo reanudado")
                self._log_luz_event("REANUDA_POR_LUZ")
            else:
                print(f"[{self.NAME}][F{frame_idx:05d}] Luz: {estado_ant} → "
                      f"{self.luz_estado} (brillo={self.luz:.0f})")

        if self.luz_estado == "OSCURO":
            # No hay luz: alerta explicita, sin conteo (no contar basura)
            self._draw = {"dets": []}
            return {"alert": True,
                    "label": f"SIN LUZ SUFICIENTE (brillo {self.luz:.0f})",
                    "score": None, "boxes": [],
                    "extra": f"CONTEO EN PAUSA | luz: {self.luz:.0f}",
                    "inside": self.inside()}

        conf = self.conf_normal
        if self.luz_estado == "BAJA":
            conf = self.conf_baja
            lab = cv2.cvtColor(proc, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            proc = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # ── Deteccion + tracking ───────────────────────────────────────
        res = self.model.track(proc, classes=DETECT_CLASSES, persist=True,
                               imgsz=IMGSZ, verbose=False, tracker=TRACKER_YAML)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            ids = (res.boxes.id.cpu().numpy().astype(int)
                   if res.boxes.id is not None
                   else np.full(len(xyxy), -1, dtype=int))
            for (x1, y1, x2, y2), c, tid in zip(xyxy, confs, ids):
                if c < conf:
                    continue
                dets.append(dict(bbox=(int(x1)+ox, int(y1)+oy,
                                       int(x2)+ox, int(y2)+oy),
                                 conf=float(c), track_id=int(tid)))

        do_diag = (frame_idx % 30 == 1)

        # ── Corredor: progreso + reglas ────────────────────────────────
        for d in dets:
            tid = d["track_id"]
            if tid < 0 or self._geo is None:
                continue
            self.faces.update(tid, frame, d["bbox"], frame_idx)

            pt = ((d["bbox"][0] + d["bbox"][2]) / 2.0,
                  (d["bbox"][1] + d["bbox"][3]) / 2.0)
            p = self._progress(pt)
            dentro = self._inside_box(pt)
            st = self.tracks.get(tid)

            if st is None:
                st = {"p_start": p, "p": p, "inside": dentro,
                      "counted": False, "born": frame_idx,
                      "seen": frame_idx, "regla": ""}
                # R4: ¿heredero de un fantasma?
                if dentro:
                    for g in self.pendings:
                        if (frame_idx - g["frame"] <= PENDING_TTL
                                and abs(g["p"] - p) <= PENDING_MATCH_P):
                            st["p_start"] = g["p_start"]
                            st["regla"] = "R4-heredado"
                            self.pendings.remove(g)
                            print(f"[{self.NAME}][F{frame_idx:05d}] id={tid} "
                                  f"hereda fantasma (p_start="
                                  f"{st['p_start']:.0%})")
                            break
                self.tracks[tid] = st

            st["seen"] = frame_idx
            prev_inside = st["inside"]

            if dentro:
                if not prev_inside:            # re-entro: nueva travesia
                    st.update(p_start=p, counted=False, regla="")
                st["inside"] = True
                st["p"] = p
                if (not st["counted"]
                        and frame_idx - st["born"] >= self.min_track_age):
                    lc = self._lines_between(st["p_start"], p)
                    if lc >= self._geo["mayoria"]:
                        regla = "R1" if not st["regla"] else "R4"
                        self._count(hacia_adentro=(p > st["p_start"]),
                                    tid=tid, frame_idx=frame_idx,
                                    regla=regla,
                                    lineas=f"{lc}/{self.n_lines}",
                                    frame=frame, bbox=d["bbox"])
                        st["counted"] = True
            else:
                if prev_inside:                # salio del box → R2
                    if not st["counted"]:
                        lado_salida = p > 0.5
                        lado_entrada = st["p_start"] > 0.5
                        lc = self._lines_between(st["p_start"], p)
                        if lado_salida != lado_entrada and lc >= 1:
                            self._count(hacia_adentro=lado_salida,
                                        tid=tid, frame_idx=frame_idx,
                                        regla="R2",
                                        lineas=f"{lc}/{self.n_lines}",
                                        frame=frame, bbox=d["bbox"])
                    st["counted"] = False      # travesia terminada
                st["inside"] = False
                st["p"] = p

            if do_diag:
                lc = self._lines_between(st["p_start"], st["p"])
                print(f"[{self.NAME}][F{frame_idx:05d}] diag id={tid} "
                      f"p={p:.0%} lineas={lc}/{self.n_lines} "
                      f"{'dentro' if dentro else 'fuera'} "
                      f"luz={self.luz:.0f}")

        # ── Tracks perdidos → R3 ───────────────────────────────────────
        for tid in list(self.tracks):
            st = self.tracks[tid]
            if frame_idx - st["seen"] > TRACK_TTL:
                if self._geo is not None and st["inside"] and not st["counted"]:
                    lc = self._lines_between(st["p_start"], st["p"])
                    if lc >= self._geo["mayoria"]:
                        self._count(hacia_adentro=(st["p"] > st["p_start"]),
                                    tid=tid, frame_idx=frame_idx,
                                    regla="R3-inferido",
                                    lineas=f"{lc}/{self.n_lines}",
                                    frame=frame, bbox=None)
                    else:
                        self.pendings.append({"p_start": st["p_start"],
                                              "p": st["p"],
                                              "frame": st["seen"]})
                del self.tracks[tid]
        self.pendings = [g for g in self.pendings
                         if frame_idx - g["frame"] <= PENDING_TTL]
        self.faces.purge(list(self.tracks))

        self._draw = {"dets": dets}

        if self.zones_error:
            extra = self.zones_error
        else:
            cw = (f"{self._crop_rect[2]-self._crop_rect[0]}x"
                  f"{self._crop_rect[3]-self._crop_rect[1]}px"
                  if self._crop_rect else "frame completo")
            extra = (f"personas: {len(dets)} | luz: {self.luz:.0f} "
                     f"({self.luz_estado}) | YOLO en {cw}")

        return {"alert": False, "label": "", "score": None, "boxes": [],
                "extra": extra, "inside": self.inside()}

    # ------------------------------------------------------------------ #
    def annotate(self, frame, result, stats):
        out = frame.copy()
        h, w = out.shape[:2]
        dets = (getattr(self, "_draw", None) or {}).get("dets", [])

        # Exponer conteos en la API (GET /api/sources incluye estos campos)
        if isinstance(stats, dict):
            stats.update(self.get_counts())

        if self._crop_rect is not None:
            cx1, cy1, cx2, cy2 = self._crop_rect
            cv2.rectangle(out, (cx1, cy1), (cx2, cy2), (110, 110, 110), 1)

        # ── Corredor + escalera ────────────────────────────────────────
        if self._geo is not None:
            g = self._geo
            overlay = out.copy()
            cv2.fillPoly(overlay, [g["poly"]], C_BOX_FILL)
            cv2.addWeighted(overlay, 0.14, out, 0.86, 0, out)
            cv2.polylines(out, [g["poly"]], True, C_BOX_BORDE, 2)

            box = g["poly"].astype(np.float32)
            # Escalera: lineas DISCONTINUAS visibles (contorno oscuro +
            # color vivo) y numeradas — visibles sobre cualquier fondo
            for i, t in enumerate(g["thr"], 1):
                if self.axis_rot:
                    a = box[1] + (box[0] - box[1]) * t
                    b = box[2] + (box[3] - box[2]) * t
                else:
                    a = box[0] + (box[3] - box[0]) * t
                    b = box[1] + (box[2] - box[1]) * t
                segs = 9
                for s in range(segs):
                    if s % 2:
                        continue
                    q1 = a + (b - a) * (s / segs)
                    q2 = a + (b - a) * ((s + 1) / segs)
                    p1 = (int(q1[0]), int(q1[1]))
                    p2 = (int(q2[0]), int(q2[1]))
                    cv2.line(out, p1, p2, (25, 25, 25), 4)
                    cv2.line(out, p1, p2, (60, 130, 255), 2)
                num = (int(a[0]), int(a[1]))
                cv2.putText(out, str(i), (num[0] - 4, num[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 3)
                cv2.putText(out, str(i), (num[0] - 4, num[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (60, 130, 255), 1)

            # Etiquetas: el color sigue a la etiqueta (ADENTRO siempre
            # verde, AFUERA siempre ambar), con contorno para legibilidad
            lados   = ("AFUERA", "ADENTRO")
            colores = (C_AFUERA, C_ADENTRO)
            if self._inv():
                lados   = ("ADENTRO", "AFUERA")
                colores = (C_ADENTRO, C_AFUERA)
            if self.axis_rot:
                m1 = ((box[1] + box[2]) / 2).astype(int)
                m2 = ((box[3] + box[0]) / 2).astype(int)
            else:
                m1 = ((box[0] + box[1]) / 2).astype(int)
                m2 = ((box[2] + box[3]) / 2).astype(int)
            for txt, m, col in ((lados[0], m1, colores[0]),
                                (lados[1], m2, colores[1])):
                cv2.putText(out, txt, (m[0] - 45, m[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (25, 25, 25), 4)
                cv2.putText(out, txt, (m[0] - 45, m[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)

        # ── Personas: bbox + centroide + barra de progreso ─────────────
        for d in dets:
            tid = d["track_id"]
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(out, (x1, y1), (x2, y2), C_PERSONA, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(out, (cx, cy), 6, (60, 60, 230), -1)
            cv2.circle(out, (cx, cy), 7, (255, 255, 255), 1)

            st = self.tracks.get(tid)
            if st is not None and self._geo is not None:
                p = st["p"]
                avanza = p >= st["p_start"]
                colb = C_PROG_IN if avanza != self._inv() else C_PROG_OUT
                bw = max(60, x2 - x1)
                by = max(60, y1 - 14)
                cv2.rectangle(out, (x1, by), (x1 + bw, by + 9),
                              (25, 28, 33), -1)
                cv2.rectangle(out, (x1, by), (x1 + int(bw * p), by + 9),
                              colb, -1)
                cv2.rectangle(out, (x1, by), (x1 + bw, by + 9),
                              (90, 95, 105), 1)
                lc = self._lines_between(st["p_start"], p)
                cv2.putText(out, f"id{tid} {p:.0%} {lc}/{self.n_lines}",
                            (x1, by - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, C_TXT, 1)
            else:
                cv2.putText(out, f"id{tid}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_PERSONA, 1)
            self.faces.draw_bubble(out, tid, d["bbox"])

        # ── Banner: contadores + semaforo de luz ───────────────────────
        cv2.rectangle(out, (0, 0), (w, 56), (24, 27, 32), -1)
        cv2.putText(out, f"ENTRARON: {self.count_in}", (12, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_PROG_IN, 2)
        cv2.putText(out, f"SALIERON: {self.count_out}", (w // 3, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_PROG_OUT, 2)
        cv2.putText(out, f"DENTRO: {self.inside()}", (2 * w // 3, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TXT, 2)
        cl = C_LUZ.get(self.luz_estado, C_TXT)
        cv2.circle(out, (w - 130, 28), 10, cl, -1)
        cv2.putText(out, f"{self.luz:.0f}", (w - 112, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cl, 2)
        if self.luz_estado == "OSCURO":
            cv2.rectangle(out, (0, 56), (w, 96), (40, 40, 200), -1)
            cv2.putText(out, "SIN LUZ SUFICIENTE PARA DETECTAR - "
                        "CONTEO EN PAUSA", (12, 84),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        self.draw_infobar(out, stats, result.get("extra", ""))
        return out

    # ------------------------------------------------------------------ #
    def teardown(self):
        self.model = None
        self.tracks.clear()
        self.pendings.clear()


if __name__ == "__main__":
    from web_server_zones import main_web_zones
    main_web_zones(PeopleCounterV2Detector)
