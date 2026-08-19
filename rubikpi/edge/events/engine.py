"""Motor de eventos: convierte scores por clip en eventos de anomalía.

Máquina de estados por cámara con **filtro temporal de falsos positivos**: un pico
aislado NO abre evento. Solo se confirma una anomalía real si se cumple alguna:
  1. PERSISTENCIA: el score se mantiene >= umbral durante `confirm_seconds` continuos.
  2. RÁFAGA: hay `burst_count` clips altos dentro de `burst_window_s` (aunque parpadee).
Cierra tras `close_consecutive` clips bajo umbral. Respeta `min_event_gap_s` entre eventos.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Callable, Deque, Dict, Optional

from ..config import EventsCfg
from ..types import ClipResult, Event, utcnow

log = logging.getLogger(__name__)


@dataclass
class _CamState:
    run_start: Optional[object] = None              # inicio del run continuo sobre umbral (datetime)
    anom_ts: Deque = field(default_factory=deque)   # t_end de clips altos recientes (ventana de ráfaga)
    below: int = 0
    open_event: Optional[Event] = None
    last_close_ts: Optional[object] = None          # datetime del último cierre


class EventEngine:
    def __init__(
        self,
        device_id: str,
        cfg: EventsCfg,
        class_name_fn: Callable[[int], str],
        on_open: Callable[[Event], None],
        on_close: Callable[[Event], None],
        threshold_for: Optional[Callable[[str], float]] = None,
    ):
        self.device_id = device_id
        self.cfg = cfg
        # Umbral por cámara (thr propio o global del dashboard). Si no se pasa,
        # cae al umbral fijo de config.yaml (compatibilidad).
        self.threshold_for = threshold_for or (lambda _cam: cfg.score_threshold)
        self.class_name_fn = class_name_fn
        self.on_open = on_open
        self.on_close = on_close
        self._cams: Dict[str, _CamState] = {}

    def _state(self, camera_id: str) -> _CamState:
        return self._cams.setdefault(camera_id, _CamState())

    def process(self, r: ClipResult) -> None:
        st = self._state(r.camera_id)
        is_anom = r.score >= self.threshold_for(r.camera_id)
        now = r.t_end

        if is_anom:
            if st.run_start is None:        # arranca un nuevo run continuo
                st.run_start = r.t_start
            st.anom_ts.append(now)
            st.below = 0
        else:
            st.run_start = None             # se rompe la persistencia
            st.below += 1

        # purga la ventana de ráfaga (deja solo lo de los últimos burst_window_s)
        while st.anom_ts and (now - st.anom_ts[0]).total_seconds() > self.cfg.burst_window_s:
            st.anom_ts.popleft()

        if st.open_event is None:
            self._maybe_open(r, st, is_anom)
        else:
            self._update_open(r, st)
            self._maybe_close(r, st, is_anom)

    def _maybe_open(self, r: ClipResult, st: _CamState, is_anom: bool) -> None:
        if not is_anom:
            return
        # Regla 1: persistencia (>= confirm_seconds continuos sobre umbral).
        persistent = (st.run_start is not None and
                      (r.t_end - st.run_start).total_seconds() >= self.cfg.confirm_seconds)
        # Regla 2: ráfaga (burst_count clips altos dentro de burst_window_s).
        burst = len(st.anom_ts) >= self.cfg.burst_count
        if not (persistent or burst):
            return  # pico aislado / aún sin confirmar -> se ignora

        # respetar gap mínimo entre eventos
        if st.last_close_ts is not None:
            if (utcnow() - st.last_close_ts) < timedelta(seconds=self.cfg.min_event_gap_s):
                return

        reason = "persistencia" if persistent else "ráfaga"
        start_ts = st.run_start if persistent else st.anom_ts[0]
        event_id = f"{self.device_id}:{r.camera_id}:{int(start_ts.timestamp())}"
        st.open_event = Event(
            event_id=event_id,
            device_id=self.device_id,
            camera_id=r.camera_id,
            state="open",
            t_start=start_ts,
            t_end=None,
            max_score=r.score,
            class_id=r.class_id,
            class_name=self.class_name_fn(r.class_id),
        )
        log.info("[%s] EVENTO ABIERTO (%s) %s score=%.2f clase=%s",
                 r.camera_id, reason, event_id, r.score, st.open_event.class_name)
        self.on_open(st.open_event)

    def _update_open(self, r: ClipResult, st: _CamState) -> None:
        ev = st.open_event
        if r.score > ev.max_score:
            ev.max_score = r.score
            ev.class_id = r.class_id
            ev.class_name = self.class_name_fn(r.class_id)

    def _maybe_close(self, r: ClipResult, st: _CamState, is_anom: bool) -> None:
        if is_anom or st.below < self.cfg.close_consecutive:
            return
        ev = st.open_event
        ev.state = "closed"
        ev.t_end = r.t_end
        st.last_close_ts = utcnow()
        st.open_event = None
        log.info("[%s] EVENTO CERRADO %s max_score=%.2f", r.camera_id, ev.event_id, ev.max_score)
        self.on_close(ev)
