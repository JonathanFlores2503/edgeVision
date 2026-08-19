"""Lector GPS NEO-6M (NMEA por puerto serie).

Conexión típica:
  - NEO-6M -> UART de la Jetson (p. ej. /dev/ttyTHS1), o
  - NEO-6M -> Arduino -> USB de la Jetson (p. ej. /dev/ttyUSB0).

Lee sentencias NMEA (GGA/RMC), extrae lat/lon/velocidad y actualiza el
MonitorState. Si `simulate=True` (o no hay pyserial/puerto), genera un recorrido
simulado para poder probar el mapa sin hardware.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from typing import Optional

from ..config import GpsCfg
from ..monitor.state import MonitorState

log = logging.getLogger(__name__)

# Ciudades de México (lat, lon) para el modo simulado: el punto aparece en un
# lugar real del país y se va moviendo, saltando de ciudad en ciudad al azar.
# Así se ejercita todo el mapa de México sin hardware GPS.
_MX_CITIES = [
    ("Ciudad de México", 19.4326, -99.1332),
    ("Guadalajara", 20.6597, -103.3496),
    ("Monterrey", 25.6866, -100.3161),
    ("Puebla", 19.0414, -98.2063),
    ("Tijuana", 32.5149, -117.0382),
    ("Cancún", 21.1619, -86.8515),
    ("Mérida", 20.9674, -89.5926),
    ("León", 21.1250, -101.6860),
    ("Querétaro", 20.5888, -100.3899),
    ("Chihuahua", 28.6320, -106.0691),
    ("Oaxaca", 17.0732, -96.7266),
    ("Culiacán", 24.7999, -107.3943),
    ("Veracruz", 19.1738, -96.1342),
    ("Hermosillo", 29.0729, -110.9559),
    ("Tuxtla Gutiérrez", 16.7516, -93.1029),
    ("San Luis Potosí", 22.1565, -100.9855),
    ("Aguascalientes", 21.8853, -102.2916),
    ("Morelia", 19.7008, -101.1844),
]


def _nmea_to_deg(value: str, hemi: str) -> Optional[float]:
    """Convierte ddmm.mmmm + hemisferio a grados decimales."""
    if not value:
        return None
    try:
        dot = value.index(".")
        deg = float(value[:dot - 2])
        minutes = float(value[dot - 2:])
        dec = deg + minutes / 60.0
        if hemi in ("S", "W"):
            dec = -dec
        return dec
    except (ValueError, IndexError):
        return None


def parse_nmea(line: str):
    """Devuelve (lat, lon, speed_kmh, fix) de una sentencia GGA/RMC, o None."""
    parts = line.split(",")
    tag = parts[0]
    try:
        if tag.endswith("RMC") and len(parts) >= 8:
            fix = parts[2] == "A"
            lat = _nmea_to_deg(parts[3], parts[4])
            lon = _nmea_to_deg(parts[5], parts[6])
            speed_kmh = float(parts[7]) * 1.852 if parts[7] else None  # nudos -> km/h
            if lat is not None and lon is not None:
                return lat, lon, speed_kmh, fix
        elif tag.endswith("GGA") and len(parts) >= 6:
            fix = parts[6] not in ("", "0")
            lat = _nmea_to_deg(parts[2], parts[3])
            lon = _nmea_to_deg(parts[4], parts[5])
            if lat is not None and lon is not None:
                return lat, lon, None, fix
    except (ValueError, IndexError):
        return None
    return None


class GpsReader:
    def __init__(self, cfg: GpsCfg, state: MonitorState):
        self.cfg = cfg
        self.state = state
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="gps", daemon=True)

    def start(self):
        if self.cfg.enabled or self.cfg.simulate:
            self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        if self.cfg.simulate:
            self._run_simulated()
            return
        try:
            import serial  # pyserial
        except ImportError:
            log.warning("pyserial no instalado; GPS en modo simulado. (pip install pyserial)")
            self._run_simulated()
            return
        try:
            ser = serial.Serial(self.cfg.port, self.cfg.baudrate, timeout=2)
            log.info("GPS NEO-6M abierto en %s", self.cfg.port)
        except Exception as exc:  # noqa: BLE001
            log.warning("No se pudo abrir GPS %s (%s); modo simulado.", self.cfg.port, exc)
            self._run_simulated()
            return
        while not self._stop.is_set():
            try:
                line = ser.readline().decode("ascii", errors="ignore").strip()
            except Exception:  # noqa: BLE001
                continue
            if not line.startswith("$"):
                continue
            res = parse_nmea(line)
            if res:
                lat, lon, speed, fix = res
                self.state.update_gps(lat, lon, speed, fix)
        ser.close()

    def _run_simulated(self):
        """Recorrido simulado ALEATORIO por México (para probar el mapa sin GPS).

        Arranca en una ciudad al azar y hace una caminata aleatoria a su alrededor;
        cada ~30 s salta a otra ciudad de `_MX_CITIES`. Así el punto recorre todo
        el país. La velocidad también varía de forma aleatoria."""
        name, lat, lon = random.choice(_MX_CITIES)
        log.info("GPS simulado (aleatorio en México). Inicio: %s.", name)
        speed = random.uniform(20, 60)
        ticks = 0
        jump_every = random.randint(25, 40)   # segundos hasta el siguiente salto
        while not self._stop.wait(1.0):
            ticks += 1
            jumped = ticks >= jump_every
            if jumped:                         # salta a otra ciudad al azar
                name, lat, lon = random.choice(_MX_CITIES)
                speed = random.uniform(20, 60)
                ticks = 0
                jump_every = random.randint(25, 40)
                log.info("GPS simulado: nueva ubicación -> %s.", name)
            else:                              # caminata aleatoria suave alrededor
                lat += random.uniform(-0.003, 0.003)
                lon += random.uniform(-0.003, 0.003)
                speed = max(0.0, min(120.0, speed + random.uniform(-8, 8)))
            self.state.update_gps(round(lat, 6), round(lon, 6), round(speed, 1),
                                  fix=True, jump=jumped)
