"""Carga y validación de la configuración del nodo edge (config.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class CameraCfg(BaseModel):
    id: str
    url: str
    enabled: bool = True
    transport: str = "tcp"   # tcp | udp — transporte RTSP. tcp evita errores RST/UDP típicos
                             # de cámaras Hikvision y redes con NAT/firewall.
    rotate: int = 0          # 0 | 90 | 180 | 270 — rotación aplicada EN CAPTURA, antes del
                             # ring buffer. Afecta por igual la vista en vivo, el clip y los
                             # frames que analiza el modelo (no es solo un giro visual).
    loop: bool = True        # solo fuentes de archivo (`file://video.mp4`): al terminar el
                             # vídeo vuelve a empezar, para que se comporte como una cámara
                             # que nunca se apaga. False = la cámara queda "terminado".

    @field_validator("rotate")
    @classmethod
    def _check_rotate(cls, v: int) -> int:
        if v not in (0, 90, 180, 270):
            raise ValueError("rotate debe ser 0, 90, 180 o 270")
        return v


class DeviceCfg(BaseModel):
    id: str
    site: str = ""


class CaptureCfg(BaseModel):
    usb_autodetect: bool = True       # plug-and-play: detecta cámaras USB en caliente
    usb_scan_interval_s: float = 3.0  # cada cuánto re-escanea /dev/video*
    usb_max_devices: int = 4          # tope de cámaras USB simultáneas
    # Reescalado en captura para no llenar la RAM: cada frame se reduce a este
    # lado corto (px) ANTES de guardarse en el ring buffer/cola/preview. El modelo
    # reescala a crop_size (320 en x3d_l) igual, así que 360 es casi sin pérdida.
    # El clip .mp4 de evidencia queda a esta resolución. 0 = sin reescalado (full-res).
    proc_short_side: int = 360
    # Tope de clips en la cola de inferencia (back-pressure). Cada clip son ~80
    # frames; con clips grandes, 16 era demasiada RAM. Si se llena, se descarta.
    infer_queue_max: int = 4
    # Cadencia por cámara: segundos a esperar TRAS emitir un clip antes de empezar
    # a acumular el siguiente. Iguala la producción al ritmo del worker para no
    # saturar la cola. 0 = clips contiguos, que es lo normal en la Jetson (la GPU
    # infiere más rápido de lo que las cámaras producen clips). Súbelo solo si ves
    # descartes por cola llena con muchas cámaras. El vídeo en vivo y el ring buffer
    # NO se pausan; solo se deja un hueco entre los clips que se mandan a inferir.
    clip_cooldown_s: float = 0.0


class ModelCfg(BaseModel):
    name: str = "x3d_l"
    device: str = "cuda"
    weights_dir: str
    weights_detector: str
    weights_classifier: str
    n_features: int = 192
    n_classes: int = 9
    class_names: List[str] = Field(default_factory=list)
    use_tensorrt: bool = False


class EventsCfg(BaseModel):
    score_threshold: float = 0.55
    open_consecutive: int = 2          # (legacy) ya no gobierna la apertura; ver confirm_seconds
    close_consecutive: int = 3
    pre_roll_s: float = 3.0
    post_roll_s: float = 3.0
    min_event_gap_s: float = 5.0
    # Filtro temporal de falsos positivos (confirma anomalía real):
    confirm_seconds: float = 5.0       # persistencia: tiempo continuo sobre umbral
    burst_window_s: float = 10.0       # ventana para la regla de ráfaga
    burst_count: int = 4               # nº de clips altos en la ventana para confirmar por ráfaga


class StorageCfg(BaseModel):
    db_path: str = "./data/edge.db"
    clips_dir: str = "./data/clips"
    ring_buffer_s: float = 12.0
    max_disk_mb: int = 4096
    # Vídeos subidos desde el dashboard, usados como cámara (`file://nombre.mp4`).
    videos_dir: str = "./data/videos"
    max_video_mb: int = 2048     # tope por archivo subido


class MqttCfg(BaseModel):
    host: str
    port: int = 8883
    tls: bool = True
    username: str = ""
    password: str = ""
    tls_ca: str = ""
    base_topic: str = "vad"
    telemetry_every_n_clips: int = 1


class HttpCfg(BaseModel):
    base_url: str
    device_token: str = ""
    verify_tls: bool = True
    upload_timeout_s: int = 60
    retry_max: int = 5


class OtaCfg(BaseModel):
    enabled: bool = True
    poll_interval_s: int = 300


class CloudCfg(BaseModel):
    enabled: bool = True
    mqtt: Optional[MqttCfg] = None
    http: Optional[HttpCfg] = None
    ota: OtaCfg = Field(default_factory=OtaCfg)


class MonitorCfg(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"        # 0.0.0.0 => accesible desde teléfono/PC en la LAN
    port: int = 8080


class GpsCfg(BaseModel):
    enabled: bool = False
    port: str = "/dev/ttyTHS1"   # UART Jetson; Windows: COMx; Arduino: /dev/ttyUSB0
    baudrate: int = 9600
    simulate: bool = False       # mueve un punto simulado (para probar el mapa)


class LoggingCfg(BaseModel):
    level: str = "INFO"


class EdgeConfig(BaseModel):
    device: DeviceCfg
    cameras: List[CameraCfg]
    model: ModelCfg
    capture: CaptureCfg = Field(default_factory=CaptureCfg)
    events: EventsCfg = Field(default_factory=EventsCfg)
    storage: StorageCfg = Field(default_factory=StorageCfg)
    cloud: CloudCfg = Field(default_factory=CloudCfg)
    monitor: MonitorCfg = Field(default_factory=MonitorCfg)
    gps: GpsCfg = Field(default_factory=GpsCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    @property
    def enabled_cameras(self) -> List[CameraCfg]:
        return [c for c in self.cameras if c.enabled]


def load_config(path: str | Path) -> EdgeConfig:
    """Lee y valida un config.yaml, devolviendo un EdgeConfig tipado."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Copia config.example.yaml a config.yaml y ajústalo."
        )
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return EdgeConfig.model_validate(raw)
