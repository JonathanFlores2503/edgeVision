"""Estado compartido entre el pipeline y el monitor web.

Guarda el último frame y el último score por cámara, y los eventos recientes.
Publica novedades a los suscriptores SSE (Server-Sent Events) del navegador.
Pensado para ser leído/escrito desde varios hilos.
"""
from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Union

import cv2
import numpy as np

from ..types import ClipResult, Event, iso


class MonitorState:
    def __init__(self, threshold: Union[float, Callable[[str], float]]):
        # Acepta un umbral fijo o un callable por cámara (thr propio / global).
        self._threshold_for: Callable[[str], float] = (
            threshold if callable(threshold) else (lambda _cam: threshold))
        self._frames: Dict[str, np.ndarray] = {}          # cam -> último frame RGB
        self._scores: Dict[str, dict] = {}                # cam -> {score,class_name,ts,latency}
        self._status: Dict[str, str] = {}                 # cam -> conectando|en vivo|sin señal
        self._fps: Dict[str, float] = {}                  # cam -> FPS de entrega (EMA)
        self._last_frame_t: Dict[str, float] = {}         # cam -> monotonic del último frame
        self._events: deque = deque(maxlen=50)
        self._meta: dict = {"device_id": "", "site": "", "model": ""}
        self._gps: dict = {"lat": None, "lon": None, "speed_kmh": None, "fix": False}
        self._event_total = 0
        self._online = False                               # ¿hay internet ahora?
        self._lock = threading.Lock()
        self._subs: List["queue.Queue[str]"] = []
        self._subs_lock = threading.Lock()

    # ── Escritura desde el pipeline ──────────────────────────────────────────
    def set_meta(self, **kw) -> None:
        with self._lock:
            self._meta.update({k: v for k, v in kw.items() if v is not None})

    def set_online(self, online: bool) -> None:
        with self._lock:
            self._online = online

    def update_gps(self, lat: float, lon: float, speed_kmh: float = None,
                   fix: bool = True, jump: bool = False) -> None:
        # `jump`=True indica un salto de ubicación (p. ej. simulado entre ciudades):
        # el mapa limpia el rastro para no dibujar una línea larga cruzando el país.
        info = {"lat": lat, "lon": lon, "speed_kmh": speed_kmh, "fix": fix}
        with self._lock:
            self._gps = dict(info)
        self._publish({"type": "gps", "jump": jump, **info})

    def update_frame(self, camera_id: str, frame_rgb: np.ndarray) -> None:
        now = time.monotonic()
        with self._lock:
            self._frames[camera_id] = frame_rgb
            self._status[camera_id] = "en vivo"
            # FPS de entrega por cámara (EMA sobre el intervalo entre frames).
            prev = self._last_frame_t.get(camera_id)
            self._last_frame_t[camera_id] = now
            if prev is not None and now > prev:
                inst = 1.0 / (now - prev)
                cur = self._fps.get(camera_id)
                self._fps[camera_id] = inst if cur is None else cur * 0.9 + inst * 0.1

    def register_camera(self, camera_id: str) -> None:
        """Da de alta una cámara configurada para que aparezca un tile aunque aún
        no llegue imagen (estado 'conectando'). Así se distingue de 'no guardada'."""
        with self._lock:
            if camera_id not in self._frames:
                self._status.setdefault(camera_id, "conectando")
        self._publish({"type": "camera_status", "camera_id": camera_id,
                       "status": self._status.get(camera_id, "conectando")})

    def set_camera_status(self, camera_id: str, status: str) -> None:
        with self._lock:
            # No degradar a 'sin señal' una cámara que ya tiene imagen viva.
            if status != "en vivo" and camera_id in self._frames and self._status.get(camera_id) == "en vivo":
                return
            self._status[camera_id] = status
        self._publish({"type": "camera_status", "camera_id": camera_id, "status": status})

    def remove_camera(self, camera_id: str) -> None:
        """Olvida una cámara (p.ej. USB desconectada o eliminada) para que salga de la interfaz."""
        with self._lock:
            self._frames.pop(camera_id, None)
            self._scores.pop(camera_id, None)
            self._status.pop(camera_id, None)
            self._fps.pop(camera_id, None)
            self._last_frame_t.pop(camera_id, None)
        self._publish({"type": "camera_removed", "camera_id": camera_id})

    def publish_stats(self, cost: Optional[dict] = None) -> None:
        """Publica un evento SSE 'stats' con FPS/latencia por cámara y el costo
        computacional (CPU/RAM/…). Lo llama un loop periódico del nodo."""
        cost = cost or {}
        with self._lock:
            cams = {}
            for cam in (set(self._frames) | set(self._scores)):
                cams[cam] = {
                    "fps": round(self._fps.get(cam, 0.0), 1),
                    "latency": self._scores.get(cam, {}).get("latency"),
                }
        self._publish({"type": "stats", "cameras": cams, **cost})

    def update_result(self, r: ClipResult, class_name: str) -> None:
        info = {"camera_id": r.camera_id, "score": round(r.score, 3),
                "class_name": class_name, "ts": iso(r.ts),
                "latency": round(r.latency_ms, 1),
                "anomaly": r.score >= self._threshold_for(r.camera_id)}
        with self._lock:
            self._scores[r.camera_id] = info
        self._publish({"type": "telemetry", **info})

    def add_event(self, ev: Event) -> None:
        msg = {"type": "event", "event_id": ev.event_id, "camera_id": ev.camera_id,
               "state": ev.state, "max_score": round(ev.max_score, 3),
               "class_name": ev.class_name,
               "t_start": iso(ev.t_start), "t_end": iso(ev.t_end) if ev.t_end else None}
        with self._lock:
            self._events.appendleft(msg)
            if ev.state == "open":
                self._event_total += 1
        self._publish(msg)

    # ── Lectura desde el servidor web ────────────────────────────────────────
    def cameras(self) -> List[str]:
        with self._lock:
            cams = set(self._frames) | set(self._scores) | set(self._status)
        return sorted(cams)

    def status(self) -> dict:
        """Resumen para el dashboard (tarjetas de estado)."""
        with self._lock:
            cams = sorted(set(self._frames) | set(self._scores) | set(self._status))
            active_anomalies = sum(1 for s in self._scores.values() if s.get("anomaly"))
            return {
                "device_id": self._meta.get("device_id", ""),
                "site": self._meta.get("site", ""),
                "model": self._meta.get("model", ""),
                "cameras_total": len(cams),
                "cameras_online": len(self._frames),
                "anomalies_active": active_anomalies,
                "events_total": self._event_total,
                "gps": dict(self._gps),
                "online": self._online,
            }

    def recent_events(self) -> List[dict]:
        with self._lock:
            return list(self._events)

    def _render_bgr(self, camera_id: str) -> Optional[np.ndarray]:
        """Último frame de la cámara anotado con el score (BGR), o None."""
        with self._lock:
            frame = self._frames.get(camera_id)
            info = self._scores.get(camera_id)
            status = self._status.get(camera_id)
        if frame is None:
            # Aún sin imagen: placeholder con el estado, para que se vea el tile.
            return self._placeholder_bgr(status) if status else None
        img = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        self._annotate(img, info)
        return img

    def _placeholder_bgr(self, status: str) -> np.ndarray:
        img = np.full((360, 640, 3), 28, dtype=np.uint8)
        msg = {"conectando": "Conectando a la camara...",
               "sin senal": "Sin senal", "sin señal": "Sin senal"}.get(status, status)
        color = (0, 165, 255) if "señal" in status or "senal" in status else (180, 180, 180)
        cv2.putText(img, msg, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        return img

    def get_jpeg(self, camera_id: str) -> Optional[bytes]:
        """Frame anotado en JPEG (para el monitor web / MJPEG)."""
        img = self._render_bgr(camera_id)
        if img is None:
            return None
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None

    def get_raw_jpeg(self, camera_id: str) -> Optional[bytes]:
        """Último frame SIN la barra de score del monitor (para el editor de zonas:
        se dibuja el corredor encima). Conserva la resolución que procesa el modelo,
        así los píxeles del box coinciden con lo que el detector ve."""
        with self._lock:
            frame = self._frames.get(camera_id)
        if frame is None:
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None

    def get_annotated_rgb(self, camera_id: str) -> Optional[np.ndarray]:
        """Frame anotado en RGB (para la app de escritorio Tkinter)."""
        img = self._render_bgr(camera_id)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

    def score_info(self, camera_id: str) -> Optional[dict]:
        with self._lock:
            return dict(self._scores[camera_id]) if camera_id in self._scores else None

    def _annotate(self, img: np.ndarray, info: Optional[dict]) -> None:
        h, w = img.shape[:2]
        score = info["score"] if info else 0.0
        anomaly = bool(info and info["anomaly"])
        cls = info["class_name"] if info else "-"
        color = (0, 0, 255) if anomaly else (0, 200, 0)

        # Barra de score arriba
        cv2.rectangle(img, (0, 0), (w, 38), (32, 32, 32), -1)
        cv2.rectangle(img, (8, 12), (8 + int((w - 120) * min(score, 1.0)), 28), color, -1)
        cv2.rectangle(img, (8, 12), (w - 112, 28), (200, 200, 200), 1)
        # La clase del clasificador solo tiene sentido cuando hay anomalía;
        # con score bajo se muestra "Normal" (evita etiquetas engañosas).
        if info:
            label = f"{score:.2f} " + (cls if anomaly else "Normal")
        else:
            label = "esperando..."
        cv2.putText(img, label, (w - 108, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if anomaly:
            cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
            cv2.putText(img, "DETECCION", (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 0, 255), 2, cv2.LINE_AA)

    # ── Pub/Sub SSE ──────────────────────────────────────────────────────────
    def subscribe(self) -> "queue.Queue[str]":
        q: "queue.Queue[str]" = queue.Queue(maxsize=100)
        with self._subs_lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: "queue.Queue[str]") -> None:
        with self._subs_lock:
            if q in self._subs:
                self._subs.remove(q)

    def _publish(self, msg: dict) -> None:
        data = json.dumps(msg)
        with self._subs_lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(data)
            except queue.Full:
                pass
