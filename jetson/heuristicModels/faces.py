"""heuristicModels/faces.py — Reconocimiento facial (rostros con nombre) en el nodo edge.

**Adaptador, no reimplementación.** El modelo vive en Full_CV
(`models/faces/model.py` + el motor compartido `models/facecore/`): SCRFD det_10g
detecta los rostros del frame, un tracker por IoU les da un id estable, AdaFace IR101
los embebe cada `refresh` frames y el nombre se decide por VOTO a lo largo del track.
Aquí solo se traduce ese modelo al contrato del nodo (`stream_processor` → `ClipResult`).

Se envuelve en vez de copiarse a propósito: la lógica de tracking, voto y demografía
es la misma que corre en Full_CV, así que un arreglo allí llega aquí sin portarlo a
mano. El precio es que este plugin necesita el árbol de Full_CV en disco (ver
`full_cv_root`) — si no está, sale deshabilitado con el motivo, no revienta el nodo.

Qué dispara un evento (`alerta` en META):
  - `"desconocido"` (por defecto): un rostro que persiste sin nombre. Es la semántica
    del modelo original — control de acceso: la galería es la lista de AUTORIZADOS.
  - `"reconocido"`: un rostro de la galería aparece en cámara — lista de BÚSQUEDA.
    En transporte público es casi siempre la que se quiere: si la galería es una lista
    de personas buscadas, lo interesante es el acierto, no el pasajero anónimo.

Multi-cámara: el motor (SCRFD + AdaFace, ~267 MB) es **uno para todo el proceso**
—lo carga `facecore` como singleton—, y por cámara solo se paga el estado del tracker.
Cada cámara tiene su hilo y su cola con descarte del más viejo, porque `feed()` corre
en el hilo de captura y no puede bloquearse.
"""
from __future__ import annotations

import os
import queue
import sys
import threading
import time
from pathlib import Path

META = {
    "key": "faces",
    "label": "Reconocimiento facial (rostros con nombre)",
    "family": "heuristic",
    "kind": "stream_processor",
    # 0 se emite cuando no hay NADIE en el frame; 3 cuando hay personas pero ninguna
    # muestra la cara (de espaldas, lejos, tapada). Los índices 0-2 no se tocan: ya hay
    # eventos guardados en la BD con esa numeración.
    "classes": ["sin personas", "rostro reconocido", "rostro desconocido",
                "persona sin rostro"],
    # Se comprueban con find_spec sin importar nada: si falta alguna, el dashboard lo
    # lista deshabilitado y dice cuál. `insightface`+`onnxruntime` son OBLIGATORIAS:
    # este modelo es 100% facial y no tiene modo degradado.
    "requires": ["cv2", "numpy", "torch", "insightface", "onnxruntime", "ultralytics"],
    "weights": [
        "weights/Adaface/adaface_ir101_webface12m.pt",   # en este árbol
        "det_10g.onnx (buffalo_l, se descarga solo)",
        "yolo11n.pt",                                    # personas
    ],
    "detail": "Detecta PERSONAS (YOLO11n) y les pone nombre con el rostro que tengan "
              "dentro (SCRFD det_10g + AdaFace IR101): una persona de espaldas sigue "
              "contando. Estima sexo y edad si hay genderage.onnx.",
    "entry": "build",

    # ── Ajustes ───────────────────────────────────────────────────────────────
    # El nodo no expone parámetros por plugin (el panel de admin solo elige modelo y
    # umbral por cámara), así que se editan aquí, igual que en FightDetector_Production.
    #
    # Dónde está el árbol de Full_CV. Se puede sobrescribir sin tocar el archivo con
    # la variable de entorno EDGE_FULL_CV_ROOT.
    "full_cv_root": "/home/jetson/Documents/Jonathan/Full_CV",
    # Galería de rostros: carpeta con una subcarpeta por persona (`<Nombre>/*.jpg`).
    # Relativa al directorio desde el que se lanza el nodo, como storage.db_path.
    "rostros_dir": "data/rostros",
    # Checkpoint de AdaFace IR101 (~250 MB). Vive en el árbol del NODO, no en el de
    # Full_CV: los pesos son de este dispositivo (igual que model.weights_dir). Si no
    # está, se usa el que `facecore` traiga por defecto y el error lo dirá.
    "adaface_weights": "weights/Adaface/adaface_ir101_webface12m.pt",
    "alerta": "reconocido",      # "reconocido" (lista de búsqueda) | "desconocido"
    "thr": 0.45,                 # coseno mínimo para asignar un nombre
    # 640 y no 320 por medición sobre la cámara real (2026-08-19, plano cenital de
    # oficina a 1920x1080, caras de 33-60 px): a 320 salían 6 caras en 6 frames y DOS
    # frames se quedaban a cero; a 640 salen 14. Con `onnxruntime-gpu` esto cuesta
    # 31 ms/frame en vez de los 570 ms que costaba en CPU, así que ya se puede pagar.
    # A 960 salen 21, pero las 7 nuevas miden 13-14 px: las tira `min_face` y de todas
    # formas un rostro de 14 px no da un embedding utilizable. Súbelo solo si la cámara
    # es de plano general y necesitas rostros de verdad lejanos.
    "det_size": 640,             # resolución interna de SCRFD (más = rostros lejanos)
    "min_face": 28,              # lado mínimo del rostro en píxeles
    # 1 y no 2: con `infer_every_n=2` encima, un `detect_every=2` dejaba la detección en
    # 1 de cada 4 frames de cámara (~3.7/s a 15 fps), y en un plano cenital la cara solo
    # está de frente un segundo. Medido que cabe: SCRFD 31 ms + YOLO 33 ms por frame
    # analizado, contra los 7.5/s que pide la cámara.
    "detect_every": 1,           # detectar 1 de cada N frames analizados
    "refresh": 6,                # re-embeber un track cada N frames
    "unknown_frames": 15,        # frames sin nombre para confirmar DESCONOCIDO
    "demographics": True,        # sexo y edad aproximada (opcional, no genera alertas)
    "infer_every_n": 2,          # de los frames que llegan, analizar 1 de cada N

    # ── Personas (YOLO11n) ────────────────────────────────────────────────────
    # El orden real de la escena es persona → rostro, no al revés: primero se ve
    # quién hay y después, si se le ve la cara, quién es. Sin esto una persona de
    # espaldas o lejana simplemente no existía para el nodo.
    "personas": True,            # False = solo rostros (como antes; ahorra ~500 MB)
    "yolo_model": "yolo11n.pt",  # mismo archivo de pesos que el resto de heurísticos
    # Medido sobre la misma cámara: con imgsz=640/conf=0.35 YOLO veía 5 de las ~9
    # personas de la escena (las sentadas al fondo se le escapaban); con 960/0.25 ve 8.
    # 1280 no aporta sobre 960 y cuesta el doble.
    "person_conf": 0.25,         # confianza mínima de YOLO para aceptar una persona
    "person_imgsz": 960,         # resolución de entrada de YOLO
    # ── Resolución de captura que necesita este modelo ────────────────────────
    # El nodo reduce cada frame a `capture.proc_short_side` (360 px por defecto) ANTES
    # de que el modelo lo vea, para no llenar la RAM. Para rostros eso es letal y se
    # midió: en la cámara de la oficina (1080p, plano cenital) las caras miden 33-60 px,
    # que a 360 de lado corto quedan en 11-20 px — por debajo de `min_face` y de lo que
    # SCRFD puede detectar. Resultado: personas sí, rostros CERO, sin un solo error en
    # el log. Con `capture_short_side: 0` (nativa) el rostro llega a tamaño completo.
    # El precio es RAM: el ring buffer son ring_s x fps x W x H x 3 por cámara, así que
    # se recorta a `capture_ring_s` (12 s a 1080p son ~1.1 GB; 4 s son ~370 MB y siguen
    # cubriendo el pre_roll de 3 s).
    "capture_short_side": 0,     # 0 = nativa; 720 si hace falta ahorrar RAM
    "capture_ring_s": 4.0,
    # Overlay: en el frame van SOLO las cajas (rostro con nombre+confianza, persona sin
    # nombre). El banner, la infobar de SCRFD, FPS/latencia y la galería se pintan en el
    # grid HTML que hay debajo del vídeo en el dashboard, no quemados en la imagen.
    "hud": False,
    "label_demographics": False,

    # Presupuesto de RAM, con la misma aritmética que el Fight Detector: se estima el
    # PICO del proceso, no el RSS del momento (cuando se decide, los pesos aún no están
    # cargados y el RSS no dice nada). Cifras conservadoras: torch+CUDA+SCRFD+AdaFace
    # +YOLO11n. El YOLO es UNO para todo el proceso (detección sin tracker, así que
    # compartirlo no mezcla escenas), por eso la cámara extra sigue siendo barata.
    # `est_first_cam_mb` es lo MEDIDO en esta placa (RSS del nodo con faces + YOLO y una
    # cámara), no una estimación de despacho. Con esto, una segunda cámara se deniega
    # sola, que es lo correcto: no cabe sin liberar el escritorio.
    "mem_budget_mb": 4700,
    "est_first_cam_mb": 4900,
    "est_per_cam_mb": 120,
    "min_sys_avail_mb": 1200,
}

#: Cada cuánto se emite un ClipResult si el estado no cambia (el motor de eventos
#: necesita ver el score bajo para cerrar, no solo el alto para abrir).
EMIT_INTERVAL_S = 1.0
#: Cada cuánto se resume la escena en el log. Existe porque «no detecta rostros» no se
#: puede diagnosticar desde el dashboard si no puedes entrar al panel: esta línea dice si
#: el problema es que no hay caras detectables, que no se reconocen, o que el modelo va
#: lento. Solo se imprime cuando hay algo que contar.
LOG_ESCENA_S = 10.0
#: Score que se publica cuando hay rostros pero no son el caso de alerta. Debe quedar
#: por debajo de events.score_threshold (0.55 por defecto) para no abrir eventos.
SCORE_PRESENTE = 0.30


def _full_cv_root() -> Path:
    return Path(os.environ.get("EDGE_FULL_CV_ROOT")
                or META["full_cv_root"]).expanduser()


def _adaface_weights() -> Path:
    w = Path(os.environ.get("EDGE_ADAFACE_WEIGHTS")
             or META["adaface_weights"]).expanduser()
    return w if w.is_absolute() else (Path.cwd() / w)


def _rostros_dir() -> Path:
    d = Path(os.environ.get("EDGE_ROSTROS_DIR") or META["rostros_dir"]).expanduser()
    return d if d.is_absolute() else (Path.cwd() / d)


def _import_full_cv():
    """Importa el modelo de Full_CV, dejando su raíz en sys.path.

    Los imports son de Full_CV (`app.contract`, `models.facecore`), no del nodo, así
    que su raíz tiene que estar en sys.path. Se hace aquí y no arriba del módulo
    porque el registro lee META por AST sin importar nada: el listado del dashboard
    tiene que ser barato y no puede depender de que Full_CV esté instalado.
    """
    root = _full_cv_root()
    if not (root / "models" / "faces" / "model.py").is_file():
        raise FileNotFoundError(
            f"No encuentro el modelo de rostros en {root}. Ajusta META['full_cv_root'] "
            f"en heuristicModels/faces.py o exporta EDGE_FULL_CV_ROOT=/ruta/a/Full_CV.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from models.faces import META as FICHA          # noqa: N811 — ficha de Full_CV
    from models.faces import model as faces_model
    return FICHA, faces_model


class _Personas:
    """YOLO11n de personas, UNO para todo el proceso.

    Se comparte porque aquí YOLO se usa en modo `predict` (sin tracker): la inferencia
    no guarda estado de la escena, así que dos cámaras pueden turnarse el mismo modelo
    sin mezclar nada. Es lo que evita pagar ~500 MB por cámara. El lock serializa las
    llamadas: con 1-2 cámaras y ~15 ms por frame no se nota, y a cambio la segunda
    cámara sale casi gratis en RAM.
    """

    _inst = None
    _inst_lock = threading.Lock()

    @classmethod
    def get(cls):
        with cls._inst_lock:
            if cls._inst is None:
                cls._inst = cls()
            return cls._inst

    @classmethod
    def liberar(cls):
        with cls._inst_lock:
            cls._inst = None

    def __init__(self):
        import torch
        from ultralytics import YOLO
        self._lock = threading.Lock()
        self._conf = float(META.get("person_conf", 0.35))
        self._imgsz = int(META.get("person_imgsz", 640))
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # ultralytics 8.4 sustituyó `half=True` por `quantize=16` y el viejo avisa por
        # cada frame. 16 = fp16, None = fp32 (en CPU fp16 no ayuda).
        self._quantize = 16 if self._device == "cuda" else None
        self._model = YOLO(_resolve_yolo(str(META.get("yolo_model", "yolo11n.pt"))))
        self.t_ms = 0.0                  # media móvil del coste por frame
        print(f"[FACES] Personas: YOLO11n en {self._device}"
              f"{' fp16' if self._quantize == 16 else ''} conf={self._conf} "
              f"imgsz={self._imgsz}.", flush=True)

    def detectar(self, frame_bgr):
        """Devuelve las cajas de persona del frame: lista de (x1, y1, x2, y2, conf)."""
        t0 = time.perf_counter()
        with self._lock:
            res = self._model.predict(frame_bgr, classes=[0], conf=self._conf,
                                      imgsz=self._imgsz, quantize=self._quantize,
                                      device=self._device, verbose=False)[0]
        dt = (time.perf_counter() - t0) * 1000.0
        self.t_ms = dt if not self.t_ms else (0.8 * self.t_ms + 0.2 * dt)
        out = []
        if res.boxes is not None and len(res.boxes):
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
        return out


def _resolve_yolo(name: str) -> str:
    """Ubica los pesos de YOLO en el árbol del nodo, no en el CWD.

    Sin esto ultralytics resuelve el nombre contra el directorio desde el que se lanzó
    el nodo y se **descarga otra copia** de los pesos, dejándola tirada ahí. Es la misma
    resolución que usan el contador de flujo y el Fight Detector, así que los tres
    comparten el mismo archivo.
    """
    aquí = Path(__file__).resolve().parent
    for cand in (aquí.parent / name, aquí / name, Path.cwd() / name):
        if cand.is_file():
            return str(cand)
    return name


def _centro_dentro(caja_rostro, caja_persona) -> bool:
    """¿El centro del rostro cae dentro de la caja de la persona?

    Se compara el CENTRO y no el solape: un rostro es una fracción pequeña de la
    persona, así que el IoU siempre sale ridículo y no sirve para asociar. El centro
    dentro de la caja es el criterio simple que funciona incluso con gente pegada.
    """
    fx1, fy1, fx2, fy2 = caja_rostro
    px1, py1, px2, py2 = caja_persona[:4]
    cx, cy = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0
    return px1 <= cx <= px2 and py1 <= cy <= py2


class _Cam:
    """Una cámara: su cola, su hilo y su instancia del modelo.

    El modelo de Full_CV guarda el estado del tracking en la instancia (`self._tracks`),
    así que **no** se puede compartir entre cámaras: dos fuentes mezclarían los ids y
    los votos de nombre. Lo caro (los pesos) sí se comparte, vía `facecore`.
    """

    QUEUE_MAX = 2       # tiempo real: interesa el frame de ahora, no la cola de hace 3 s

    def __init__(self, camera_id, fmodel, on_result, stop_event, frame_sink=None,
                 infer_every_n=1, alerta="reconocido", personas=None):
        self.camera_id = camera_id
        self._m = fmodel
        self._on_result = on_result
        self._stop = stop_event
        self._frame_sink = frame_sink
        self._every = max(1, int(infer_every_n or 1))
        self._alerta = alerta
        self._personas = personas          # `_Personas` compartido, o None
        # Último estado de la escena, para el grid del dashboard. Se publica por
        # referencia atómica (se reemplaza el dict entero) en vez de con un lock: el
        # lector solo quiere la foto más reciente y nunca la ve a medio escribir.
        self.snap = {}
        self._seen = 0
        self._q: "queue.Queue" = queue.Queue(maxsize=self.QUEUE_MAX)
        self._thread = threading.Thread(target=self._run, name=f"faces-{camera_id}",
                                        daemon=True)
        self._last_class = 0
        self._last_emit = 0.0
        # Media móvil de la latencia de rostros. El valor instantáneo alterna entre ~0 ms
        # (frame que solo trackea, por `detect_every`) y ~70 ms (frame que detecta y
        # embebe): en el grid eso es un número parpadeando que no dice nada.
        self._lat_ema = 0.0
        self._last_log = 0.0

    def start(self):
        self._thread.start()

    def submit(self, frame_bgr, ts):
        self._seen += 1
        if self._every > 1 and (self._seen % self._every):
            return
        try:
            self._q.put_nowait((frame_bgr, ts))
        except queue.Full:                       # descarta el más viejo
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
        import cv2
        log = logging.getLogger("heuristic.faces")
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
                res = self._m.infer_frame(frame_bgr)
            except Exception:  # noqa: BLE001 — un frame no debe tumbar el worker
                log.exception("infer_frame() de faces [%s] falló", self.camera_id)
                continue
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # ── Personas: el orden real es persona → rostro ─────────────────────
            # Primero quién hay en la escena (YOLO), después a quién se le ve la cara
            # y quién es (SCRFD+AdaFace). Una persona de espaldas cuenta como persona
            # sin nombre en vez de no existir.
            personas = []
            if self._personas is not None:
                try:
                    personas = self._personas.detectar(frame_bgr)
                except Exception:  # noqa: BLE001 — sin personas seguimos con rostros
                    log.exception("YOLO de personas [%s] falló", self.camera_id)

            visible = self._tracks_visibles()
            sin_rostro = self._pintar_personas(res, personas, visible, cv2)

            # Vista en vivo: SOLO cajas. El nombre y la confianza van sobre el rostro
            # (los dibuja el modelo de Full_CV con `hud=False`) y la persona sin cara
            # lleva su propia caja. Todo lo demás —FPS, latencia, SCRFD, galería— se
            # pinta en el grid HTML bajo el vídeo, no quemado en la imagen.
            if self._frame_sink is not None and res.frame is not None:
                try:
                    self._frame_sink(self.camera_id,
                                     cv2.cvtColor(res.frame, cv2.COLOR_BGR2RGB))
                except Exception:  # noqa: BLE001 — el overlay no tumba el worker
                    log.exception("overlay de faces [%s] falló", self.camera_id)

            self.snap = self._snapshot(res, visible, personas, sin_rostro, latency_ms)
            self._log_escena(log)

            class_id, score = self._puntuar(res, n_personas=len(personas))
            now = time.monotonic()
            if class_id != self._last_class or (now - self._last_emit) >= EMIT_INTERVAL_S:
                self._last_class = class_id
                self._last_emit = now
                self._emit(ts, class_id, score, latency_ms)

    def _log_escena(self, log):
        """Resumen periódico de la escena en el log del nodo."""
        s = self.snap
        if not (s.get("personas") or s.get("rostros")):
            return
        ahora = time.monotonic()
        if ahora - self._last_log < LOG_ESCENA_S:
            return
        self._last_log = ahora
        caras = ", ".join(
            f"{c['nombre'] or ('id' + str(c['id']))}:{c['estado']}"
            + (f" {c['score']:.2f}" if c.get("score") else "")
            for c in (s.get("caras") or []))
        log.info("[FACES] %s personas=%s (sin rostro %s) rostros=%s rec=%s desc=%s "
                 "| yolo %s ms rostros %s ms | %s",
                 self.camera_id, s.get("personas"), s.get("sin_rostro"),
                 s.get("rostros"), s.get("reconocidos"), s.get("desconocidos"),
                 s.get("yolo_ms"), s.get("lat_ms"), caras or "—")

    # ── escena: personas + rostros ────────────────────────────────────────────
    def _tracks_visibles(self):
        """Tracks de rostro vivos ahora mismo, o [] si el interno del modelo cambia."""
        try:
            return [t for t in self._m._tracks if t.misses == 0]
        except Exception:  # noqa: BLE001 — el grid no vale una excepción
            return []

    def _pintar_personas(self, res, personas, visible, cv2):
        """Dibuja las cajas de persona sobre el frame ya anotado. Devuelve cuántas
        personas NO tienen rostro asociado.

        La persona que sí tiene rostro no se etiqueta: su nombre y confianza ya están
        escritos sobre la cara, y repetirlo llenaría el frame de texto (que es justo lo
        que se quitó). Solo se marca con color: verde si el rostro tiene nombre, rojo si
        es un desconocido confirmado. La persona sin cara visible lleva su caja gris con
        la palabra `persona`, para que se vea que el nodo la está contando.
        """
        sin_rostro = 0
        frame = res.frame if res.frame is not None else None
        for caja in personas:
            px1, py1, px2, py2 = [int(v) for v in caja[:4]]
            rostros = [t for t in visible if _centro_dentro(t.bbox, caja)]
            con_nombre = [t for t in rostros if t.name]
            if con_nombre:
                color, etiqueta = (0, 200, 50), ""          # verde: identificada
            elif rostros:
                color, etiqueta = (0, 50, 220), ""          # rojo: cara sin nombre
            else:
                sin_rostro += 1
                color, etiqueta = (170, 170, 170), "persona"
            if frame is None:
                continue
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 1)
            if etiqueta:
                cv2.putText(frame, etiqueta, (px1 + 3, max(12, py1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return sin_rostro

    def _snapshot(self, res, visible, personas, sin_rostro, latency_ms) -> dict:
        """Foto de la escena para el grid del dashboard (`GET /api/faces`).

        Es lo que antes iba quemado en la infobar del frame: cuentas, parámetros del
        detector, estado de la galería y la lista de rostros con su confianza.
        """
        stats = res.stats or {}
        self._lat_ema = (float(latency_ms) if not self._lat_ema
                         else 0.8 * self._lat_ema + 0.2 * float(latency_ms))
        try:
            gal = self._m._gallery.info()
        except Exception:  # noqa: BLE001
            gal = {}
        filas = []
        for t in visible:
            filas.append({
                "id": t.tid,
                "nombre": t.name or "",
                "score": round(float(t.score), 3) if t.name else None,
                "det_score": round(float(t.det_score), 3),
                "sexo": t.sex or "",
                "edad": t.age,
                "estado": ("reconocido" if t.name
                           else ("analizando" if t.n_embeds == 0 else "desconocido")),
            })
        filas.sort(key=lambda f: (f["nombre"] == "", -(f["score"] or 0)))
        return {
            "personas": len(personas),
            "sin_rostro": sin_rostro,
            "rostros": len(visible),
            "reconocidos": int(stats.get("reconocidos") or 0),
            "desconocidos": int(stats.get("desconocidos") or 0),
            "estado": stats.get("estado", ""),
            "lat_ms": round(self._lat_ema, 1),
            "lat_pico_ms": round(float(latency_ms), 1),
            "yolo_ms": (round(self._personas.t_ms, 1)
                        if self._personas is not None else None),
            "det_size": int(META["det_size"]),
            "thr": float(META["thr"]),
            "demografia": bool(stats.get("hombres") is not None),
            "galeria": {"personas": int(gal.get("n_persons") or 0),
                        "rostros": int(gal.get("n_faces") or 0)},
            "caras": filas,
            "ts": time.time(),
        }

    def _puntuar(self, res, n_personas=0):
        """`InferenceResult` del modelo → (class_id, score) del contrato del nodo.

        El score gobierna los eventos: alto solo en el caso que se declaró como alerta
        en META['alerta']; el otro caso se publica presente-pero-bajo-umbral para que
        se vea en el dashboard sin abrir evento.
        """
        conocidos = [d for d in res.detections if d.label.startswith("RECONOCIDO")]
        desconocidos = [d for d in res.detections if not d.label.startswith("RECONOCIDO")]
        if not res.detections:
            # Nadie con cara visible. Si YOLO ve gente, el frame NO está vacío: se
            # publica «persona sin rostro» presente-pero-bajo-umbral, que es la
            # diferencia entre «no hay nadie» y «hay alguien de espaldas».
            return (3, SCORE_PRESENTE) if n_personas else (0, 0.0)
        if self._alerta == "desconocido":
            # Solo cuenta el desconocido CONFIRMADO (el modelo ya aplicó
            # `unknown_frames`): un rostro recién visto no es una alerta todavía.
            confirmado = any(e.get("level") == "alert" for e in res.events) or \
                any("DESCONOCIDO" in str(s) for s in (res.stats.get("estado"),))
            if desconocidos and confirmado:
                return 2, 1.0
            return (1, SCORE_PRESENTE) if conocidos else (2, SCORE_PRESENTE)
        # alerta == "reconocido": la galería es una lista de búsqueda.
        if conocidos:
            return 1, self._score_identidad(conocidos)
        return 2, SCORE_PRESENTE

    def _score_identidad(self, conocidos) -> float:
        """Confianza de IDENTIDAD (coseno del voto), no de detección de cara.

        `Detection.score` del modelo es el `det_score` de SCRFD: lo seguro que está de
        que ahí hay una cara, que es 0.9 incluso cuando el nombre casó por los pelos.
        Lo que debe gobernar el evento es lo seguro que está de QUIÉN es, y eso vive en
        el track (`_FaceTrack.score`, el mejor coseno votado). Así el umbral por cámara
        del dashboard funciona como un segundo filtro de identidad, más estricto que el
        `thr` de META, y se puede apretar sin tocar el archivo.
        """
        try:
            vivos = [t for t in self._m._tracks if t.misses == 0 and t.name]
            if vivos:
                return max(min(1.0, float(t.score)) for t in vivos)
        except Exception:  # noqa: BLE001 — si cambia el interno, no se cae el nodo
            pass
        return max(float(d.score) for d in conocidos)

    def _emit(self, ts, class_id, score, latency_ms):
        from edge.types import ClipResult, utcnow
        probs = [0.0] * len(META["classes"])
        probs[class_id] = score
        if class_id != 0:
            probs[0] = max(0.0, 1.0 - score)
        else:
            probs[0] = 1.0
        self._on_result(ClipResult(
            camera_id=self.camera_id, t_start=ts, t_end=utcnow(),
            score=score, class_id=class_id, class_probs=probs,
            latency_ms=latency_ms,
        ))


class Model:
    """Adaptador `stream_processor` multi-cámara del modelo de rostros de Full_CV.

    Una instancia de `FacesModel` por cámara (el tracking es por escena), creada
    perezosamente al llegar su primer frame y en un hilo aparte para no bloquear la
    captura mientras cargan los pesos. El motor SCRFD+AdaFace es del proceso.
    """

    def __init__(self, cfg=None):
        self._cfg = cfg
        self._cams = {}
        self._loading = set()
        self._denied = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._on_result = None
        self._frame_sink = None
        self._ficha = None
        self._faces_mod = None
        self._engine_owner = None        # cámara que cargó el motor (la que lo libera)
        self._budget_mb = int(META.get("mem_budget_mb", 4700))
        self._alerta = str(META.get("alerta", "reconocido"))

    # ── protocolo del nodo ────────────────────────────────────────────────────
    def set_frame_sink(self, fn):
        self._frame_sink = fn

    def start(self, on_result):
        self._on_result = on_result
        gal = _rostros_dir()
        print(f"[FACES] Multi-cámara listo. Galería={gal} alerta='{self._alerta}' "
              f"det_size={META['det_size']} thr={META['thr']} "
              f"personas={'ON' if META.get('personas', True) else 'OFF'} "
              f"presupuesto RAM={self._budget_mb}MB.", flush=True)

    def feed(self, camera_id, frame_rgb, ts):
        cam = self._cams.get(camera_id)
        if cam is None:
            self._ensure_cam(camera_id)      # carga en bg; este frame se descarta
            return
        import cv2
        cam.submit(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), ts)

    def stop(self):
        self._stop.set()
        with self._lock:
            cams = list(self._cams.values())
        for c in cams:
            c.join(timeout=5)
        # `FacesModel.unload()` llama a `facecore.unload()`, que libera el motor
        # COMPARTIDO: se llama una sola vez, no por cámara, o la primera dejaría a las
        # demás sin detector.
        with self._lock:
            owner = self._cams.get(self._engine_owner)
        if owner is not None:
            try:
                owner._m.unload()
            except Exception:  # noqa: BLE001 — el apagado no se detiene por esto
                import logging
                logging.getLogger("heuristic.faces").exception(
                    "Error liberando el motor de rostros.")
        _Personas.liberar()

    def class_name(self, class_id):
        names = META["classes"]
        return names[class_id] if 0 <= class_id < len(names) else f"class_{class_id}"

    def capture_hints(self) -> dict:
        """Qué resolución de captura necesita este modelo (lo lee `edge/main.py`)."""
        return {"short_side": int(META.get("capture_short_side", 0)),
                "ring_seconds": float(META.get("capture_ring_s", 4.0))}

    # ── escena en vivo para el dashboard (duck-typing, como supports_counts) ──
    # El servidor no conoce este modelo: pregunta por estos métodos y, si están, expone
    # `GET /api/faces`. Es el mismo contrato que usa el contador de flujo con sus
    # conteos, para no meter una dependencia del monitor hacia los plugins.
    def supports_faces(self) -> bool:
        return True

    def get_faces(self) -> dict:
        with self._lock:
            cams = dict(self._cams)
            cargando = sorted(self._loading)
            denegadas = sorted(self._denied)
        out = {}
        for cid, cam in cams.items():
            snap = cam.snap or {}
            if snap:
                out[cid] = snap
        return {
            "cameras": out,
            "cargando": cargando,
            "sin_ram": denegadas,
            "modelo": {
                "det_size": int(META["det_size"]),
                "thr": float(META["thr"]),
                "min_face": int(META["min_face"]),
                "detect_every": int(META["detect_every"]),
                "infer_every_n": int(META["infer_every_n"]),
                "personas": bool(META.get("personas", True)),
                "alerta": self._alerta,
            },
            "galeria": self._galeria_info(),
        }

    # ── galería de rostros: ver y gestionar desde el panel de admin ───────────
    # Las fotos son el dato que convierte esto en algo útil (o en una demo), así que se
    # gestionan desde el navegador y no por SSH. Toda la validación de nombres vive aquí
    # porque `<persona>` y `<archivo>` llegan de la red: son rutas.
    FOTO_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    MAX_FOTO_MB = 12

    def supports_gallery(self) -> bool:
        return True

    @staticmethod
    def _seguro(nombre: str, *, es_archivo: bool) -> str:
        """Nombre de persona/archivo saneado, o "" si no es aceptable.

        Rechaza rutas (`/`, `\\`, `..`) y nombres ocultos en vez de intentar
        arreglarlos: un nombre raro es un error del cliente, no algo que adivinar.
        """
        n = (nombre or "").strip().strip(".")
        if not n or len(n) > 80:
            return ""
        if "/" in n or "\\" in n or ".." in n or n.startswith("."):
            return ""
        if es_archivo and not n.lower().endswith(Model.FOTO_EXTS):
            return ""
        return n

    def _galeria_info(self) -> dict:
        gal = _rostros_dir()
        personas = []
        n_fotos = 0
        if gal.is_dir():
            for d in sorted(p for p in gal.iterdir() if p.is_dir()):
                fotos = sorted(f.name for f in d.iterdir()
                               if f.is_file() and f.suffix.lower() in self.FOTO_EXTS)
                n_fotos += len(fotos)
                personas.append({"nombre": d.name, "n_fotos": len(fotos),
                                 "fotos": fotos})
        # `cache` es lo que el modelo tiene CARGADO en memoria; `personas` es lo que hay
        # en disco. Cuando no coinciden, el panel dice que falta reconstruir — es la
        # confusión clásica: se copian fotos y el reconocimiento sigue igual.
        cache = {}
        try:
            with self._lock:
                cam = next(iter(self._cams.values()), None)
            if cam is not None:
                info = cam._m._gallery.info()
                cache = {"personas": int(info.get("n_persons") or 0),
                         "rostros": int(info.get("n_faces") or 0)}
        except Exception:  # noqa: BLE001 — el panel no se cae por esto
            cache = {}
        return {"dir": str(gal), "personas": personas,
                "n_personas": len(personas), "n_fotos": n_fotos,
                "cache": cache, "max_mb": self.MAX_FOTO_MB,
                "formatos": [e[1:] for e in self.FOTO_EXTS],
                "al_dia": bool(cache and cache.get("personas") == len(personas))}

    def get_gallery(self) -> dict:
        return self._galeria_info()

    def gallery_photo_path(self, persona: str, archivo: str):
        """Ruta en disco de una foto de la galería, o None si el nombre no es válido."""
        per = self._seguro(persona, es_archivo=False)
        arc = self._seguro(archivo, es_archivo=True)
        if not per or not arc:
            return None
        p = _rostros_dir() / per / arc
        return p if p.is_file() else None

    def gallery_add_photo(self, persona: str, archivo: str, datos: bytes) -> dict:
        per = self._seguro(persona, es_archivo=False)
        arc = self._seguro(archivo, es_archivo=True)
        if not per:
            return {"ok": False, "error": "nombre de persona no válido"}
        if not arc:
            return {"ok": False, "error": "formato no admitido "
                                          f"({', '.join(e[1:] for e in self.FOTO_EXTS)})"}
        if len(datos) > self.MAX_FOTO_MB * 1024 * 1024:
            return {"ok": False, "error": f"la foto pasa de {self.MAX_FOTO_MB} MB"}
        if not datos:
            return {"ok": False, "error": "archivo vacío"}
        d = _rostros_dir() / per
        d.mkdir(parents=True, exist_ok=True)
        destino = d / arc
        # No se sobrescribe en silencio: se numera. Subir dos veces "foto.jpg" desde el
        # móvil es lo normal y perder la primera sería un dato borrado sin avisar.
        if destino.exists():
            base, ext = destino.stem, destino.suffix
            i = 2
            while (d / f"{base}_{i}{ext}").exists():
                i += 1
            destino = d / f"{base}_{i}{ext}"
        destino.write_bytes(datos)
        return {"ok": True, "persona": per, "foto": destino.name,
                "galeria": self._galeria_info()}

    def gallery_delete(self, persona: str, archivo: str = "") -> dict:
        """Borra una foto, o la persona entera si no se pasa archivo."""
        import shutil
        per = self._seguro(persona, es_archivo=False)
        if not per:
            return {"ok": False, "error": "nombre de persona no válido"}
        d = _rostros_dir() / per
        if not d.is_dir():
            return {"ok": False, "error": "esa persona no está en la galería"}
        if archivo:
            arc = self._seguro(archivo, es_archivo=True)
            if not arc:
                return {"ok": False, "error": "nombre de archivo no válido"}
            f = d / arc
            if not f.is_file():
                return {"ok": False, "error": "esa foto no existe"}
            f.unlink()
        else:
            shutil.rmtree(d)
        return {"ok": True, "galeria": self._galeria_info()}

    def gallery_rebuild(self) -> dict:
        """Re-embebe la galería EN CALIENTE, sin reiniciar el nodo.

        La galería es un singleton de `facecore` por directorio, así que reconstruirla
        desde una cámara la actualiza para todas: se llama una sola vez y las demás ven
        los embeddings nuevos en su siguiente voto.
        """
        with self._lock:
            cam = next(iter(self._cams.values()), None)
        if cam is None:
            return {"ok": False, "error": "el modelo aún no está cargado; espera a que "
                                          "la cámara arranque"}
        try:
            res = cam._m.rebuild_galleries()
        except Exception as e:  # noqa: BLE001 — un fallo aquí no tumba el nodo
            import logging
            logging.getLogger("heuristic.faces").exception(
                "Error reconstruyendo la galería de rostros.")
            return {"ok": False, "error": f"no se pudo reconstruir: {e}"}
        res = dict(res or {})
        res["galeria"] = self._galeria_info()
        return res

    # ── gestión por cámara ────────────────────────────────────────────────────
    @staticmethod
    def _sys_total_mb() -> float:
        try:
            import psutil
            return psutil.virtual_memory().total / 1e6
        except Exception:  # noqa: BLE001 — sin psutil no aplicamos presupuesto
            return 0.0

    def _ensure_cam(self, camera_id):
        with self._lock:
            if (camera_id in self._cams or camera_id in self._loading
                    or camera_id in self._denied):
                return
            # Se cuentan las que están cargando: las cámaras arrancan a la vez y si
            # solo se mira `_cams` el presupuesto no frena a ninguna.
            n_previas = len(self._cams) + len(self._loading)
            pico_mb = (int(META["est_first_cam_mb"])
                       + int(META["est_per_cam_mb"]) * n_previas)
            reason = None
            if n_previas and pico_mb > self._budget_mb:
                reason = (f"pico estimado {pico_mb}MB con {n_previas + 1} cámaras > "
                          f"tope {self._budget_mb}MB del proceso")
            else:
                total_mb = self._sys_total_mb()
                min_libre = int(META["min_sys_avail_mb"])
                if n_previas and total_mb and pico_mb > (total_mb - min_libre):
                    reason = (f"pico estimado {pico_mb}MB dejaría a la placa "
                              f"(RAM total {total_mb:.0f}MB) por debajo de los "
                              f"{min_libre}MB libres que necesita")
            if reason:
                self._denied.add(camera_id)
                print(f"[FACES] Sin RAM para '{camera_id}' ({reason}); se omite y el "
                      f"nodo sigue con las demás. Sube META['mem_budget_mb'] si de "
                      f"verdad hay memoria.", flush=True)
                return
            self._loading.add(camera_id)
        threading.Thread(target=self._load_cam, args=(camera_id,),
                         name=f"faces-load-{camera_id}", daemon=True).start()

    def _load_cam(self, camera_id):
        import logging
        log = logging.getLogger("heuristic.faces")
        try:
            with self._lock:
                primera = not self._cams
            if self._ficha is None:
                self._ficha, self._faces_mod = _import_full_cv()
                self._apuntar_pesos(log)
                # La galería es dato del NODO, no de Full_CV: se apunta al directorio
                # de este dispositivo en vez del `Rostros/` del repo de modelos.
                gal = _rostros_dir()
                gal.mkdir(parents=True, exist_ok=True)
                self._faces_mod.ROSTROS_DIR = str(gal)

            print(f"[FACES] Cargando modelo de rostros para '{camera_id}'"
                  f"{' (incluye el motor SCRFD+AdaFace)' if primera else ''}...", flush=True)
            m = self._faces_mod.FacesModel(self._ficha)
            m.configure({
                "thr": META["thr"],
                "det_size": META["det_size"],
                "min_face": META["min_face"],
                "detect_every": META["detect_every"],
                "refresh": META["refresh"],
                "alert_unknown": True,     # el modelo confirma el desconocido; el
                                           # score lo decide META['alerta']
                "unknown_frames": META["unknown_frames"],
                "demographics": bool(META["demographics"]),
                # El frame lleva SOLO cajas; el resto de la información se pinta en el
                # grid HTML bajo el vídeo (ver `_snapshot` y GET /api/faces).
                "hud": bool(META.get("hud", False)),
                "label_demographics": bool(META.get("label_demographics", False)),
            })
            m.load()                       # aquí se cargan los pesos (o falla claro)
            self._avisar_galeria(m, log)

            personas = None
            if bool(META.get("personas", True)):
                try:
                    personas = _Personas.get()
                except Exception:  # noqa: BLE001 — sin YOLO seguimos solo con rostros
                    log.exception("No se pudo cargar YOLO de personas; esta cámara "
                                  "seguirá solo con rostros.")
            cam = _Cam(camera_id, m, self._on_result, self._stop,
                       frame_sink=self._frame_sink,
                       infer_every_n=META["infer_every_n"], alerta=self._alerta,
                       personas=personas)
            cam.start()
            with self._lock:
                self._cams[camera_id] = cam
                self._loading.discard(camera_id)
                if self._engine_owner is None:
                    self._engine_owner = camera_id
            print(f"[FACES] '{camera_id}' activa.", flush=True)
        except Exception as e:  # noqa: BLE001 — una cámara que no carga no tumba el nodo
            with self._lock:
                self._loading.discard(camera_id)
                self._denied.add(camera_id)
            log.exception("No se pudo cargar el modelo de rostros para '%s': %s",
                          camera_id, e)

    def _apuntar_pesos(self, log):
        """Redirige AdaFace al checkpoint del nodo.

        `facecore.ensure_ready()` hace `from embedder import default_weights_path`
        DENTRO de la función, así que basta con reescribir ese atributo del módulo
        antes de la primera carga: el import lo resuelve en ese momento y se lleva el
        nuestro. Se prefiere esto a un symlink dentro de Full_CV porque deja la ruta
        declarada en el plugin, a la vista, en vez de en un enlace del sistema.
        """
        ruta = _adaface_weights()
        if not ruta.is_file():
            log.warning("No encuentro el checkpoint de AdaFace en %s; se usará el que "
                        "traiga facecore por defecto.", ruta)
            return
        from models import facecore
        if facecore.FACE_RUNNER_DIR not in sys.path:
            sys.path.append(facecore.FACE_RUNNER_DIR)
        import embedder as _emb
        _emb.default_weights_path = lambda _r=str(ruta): _r
        print(f"[FACES] AdaFace desde {ruta}", flush=True)

    def _avisar_galeria(self, m, log):
        """Una galería vacía no es un error, pero sí explica que todo salga DESCONOCIDO."""
        try:
            info = m._gallery.info()
        except Exception:  # noqa: BLE001
            return
        if info.get("n_persons"):
            return
        ejemplo = _full_cv_root() / "models" / "faces" / "Rostros"
        log.warning(
            "Galería de rostros VACÍA en %s: sin personas registradas, todos los "
            "rostros saldrán DESCONOCIDO. Registra gente creando una subcarpeta por "
            "persona con 2-4 fotos de frente (%s/Juan/foto1.jpg). Hay una galería de "
            "ejemplo en %s.", _rostros_dir(), _rostros_dir(), ejemplo)


def build(cfg):
    return Model(cfg)
