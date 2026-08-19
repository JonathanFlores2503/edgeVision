"""Biblioteca de vídeos subidos desde el dashboard.

La plataforma nació para cámaras RTSP/USB, pero muy a menudo no hay cámara a mano
(demos, probar un modelo, revisar una grabación de un incidente). Un vídeo subido
se trata como **una cámara más**: se reproduce en bucle y a su velocidad real, así
el resto del pipeline (ring buffer, inferencia, clips, eventos, alertas) no nota
la diferencia entre un archivo y un RTSP.

La URL de una cámara de archivo es ``file://<nombre>`` (o ``file:///ruta/abs``).
También se acepta el nombre pelado (``demo.mp4``), que se resuelve contra la
biblioteca (``data/videos/`` por defecto).

Este módulo es la ÚNICA fuente de verdad sobre dónde viven los vídeos: la captura
(`edge/capture/rtsp.py`) lo usa para abrirlos y el monitor web para subirlos,
listarlos y borrarlos.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import unquote

log = logging.getLogger(__name__)

# Contenedores que OpenCV/FFmpeg abre sin problemas en la placa.
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".ts"}

_FILE_SCHEME = re.compile(r"^file://", re.I)
# Nombre seguro: sin rutas, sin sorpresas de shell/URL. El resto se sustituye.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

_lock = threading.Lock()
_dir = Path("./data/videos").resolve()
_max_bytes = 2048 * 1024 * 1024          # 2 GB por archivo, por defecto
_probe_cache: Dict[tuple, dict] = {}      # (path, mtime, size) -> metadatos


class UploadError(Exception):
    """Subida rechazada (nombre, formato, tamaño o espacio en disco)."""


# ── Configuración ────────────────────────────────────────────────────────────
def configure(directory: str | Path | None = None, max_mb: int | None = None) -> None:
    """Fija dónde viven los vídeos y el tamaño máximo por archivo."""
    global _dir, _max_bytes
    with _lock:
        if directory is not None:
            _dir = Path(directory).expanduser().resolve()
        if max_mb is not None and max_mb > 0:
            _max_bytes = int(max_mb) * 1024 * 1024


def videos_dir() -> Path:
    with _lock:
        return _dir


def max_bytes() -> int:
    with _lock:
        return _max_bytes


# ── Fuentes de captura ───────────────────────────────────────────────────────
def is_video_source(url: str) -> bool:
    """True si la URL de una cámara apunta a un archivo de vídeo (no RTSP ni USB)."""
    u = (url or "").strip()
    if not u:
        return False
    if _FILE_SCHEME.match(u):
        return True
    # `rtsp://…/algo.mp4` es un stream de red, no un archivo.
    if "://" in u:
        return False
    return Path(u).suffix.lower() in VIDEO_EXTS


def camera_url(name: str) -> str:
    """URL de cámara para un vídeo de la biblioteca."""
    return f"file://{name}"


def source_name(url: str) -> str:
    """Nombre de archivo al que apunta la URL de una cámara de vídeo ("" si no lo
    es). Sirve para saber qué cámaras usan un vídeo aunque ya no exista en disco."""
    if not is_video_source(url):
        return ""
    return Path(_FILE_SCHEME.sub("", unquote(url.strip()))).name


def resolve(url: str) -> Optional[Path]:
    """Ruta absoluta del vídeo al que apunta `url`, o None si no existe.

    Acepta `file://nombre.mp4`, `file:///ruta/abs.mp4`, una ruta suelta o el
    nombre pelado del archivo (que se busca en la biblioteca)."""
    u = unquote((url or "").strip())
    u = _FILE_SCHEME.sub("", u)
    if not u:
        return None
    p = Path(u).expanduser()
    if p.is_absolute() and p.is_file():
        return p.resolve()
    # Relativo: primero la biblioteca (lo normal), luego el CWD del nodo.
    cand = videos_dir() / Path(u).name
    if cand.is_file():
        return cand.resolve()
    if p.is_file():
        return p.resolve()
    return None


# ── Biblioteca (listar / subir / borrar) ─────────────────────────────────────
def safe_name(name: str) -> str:
    """Nombre de archivo seguro: solo la base, sin rutas ni caracteres raros."""
    base = Path(unquote((name or "").strip())).name
    base = _UNSAFE.sub("_", base).strip("._-")
    return base[:120]


def probe(path: Path) -> dict:
    """Metadatos baratos del vídeo (fps, duración, resolución) vía OpenCV.

    Se cachea por (ruta, mtime, tamaño): la lista del panel se pide a menudo y
    abrir un contenedor por cada vídeo en cada refresco es un gasto tonto."""
    try:
        st = path.stat()
    except OSError:
        return {}
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _lock:
        hit = _probe_cache.get(key)
    if hit is not None:
        return dict(hit)

    import cv2  # import diferido: este módulo lo importa también el servidor web

    info: dict = {}
    cap = cv2.VideoCapture(str(path))
    try:
        if cap.isOpened():
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            info = {"fps": round(fps, 2) if 0 < fps < 1000 else None,
                    "width": w or None, "height": h or None,
                    "duration_s": round(frames / fps, 1) if fps > 0 and frames > 0 else None}
    except Exception:  # noqa: BLE001 — un contenedor corrupto no tumba el panel
        info = {}
    finally:
        cap.release()
    with _lock:
        if len(_probe_cache) > 256:
            _probe_cache.clear()
        _probe_cache[key] = info
    return dict(info)


def _entry(path: Path, with_probe: bool = True) -> dict:
    st = path.stat()
    d = {"name": path.name,
         "size_mb": round(st.st_size / 1e6, 1),
         "mtime": int(st.st_mtime),
         "url": camera_url(path.name)}
    if with_probe:
        d.update(probe(path))
    return d


def list_videos(with_probe: bool = True) -> List[dict]:
    """Vídeos de la biblioteca, del más reciente al más viejo."""
    d = videos_dir()
    if not d.is_dir():
        return []
    out = []
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            try:
                out.append(_entry(p, with_probe))
            except OSError:
                continue
    out.sort(key=lambda e: e["mtime"], reverse=True)
    return out


def _unique_path(name: str) -> Path:
    """Ruta libre para `name`: si ya existe, añade -1, -2… (no pisa lo subido)."""
    d = videos_dir()
    p = d / name
    if not p.exists():
        return p
    stem, suf = p.stem, p.suffix
    for i in range(1, 1000):
        q = d / f"{stem}-{i}{suf}"
        if not q.exists():
            return q
    raise UploadError("demasiados archivos con ese nombre")


def save_stream(name: str, fileobj, length: int, chunk: int = 1024 * 1024) -> dict:
    """Guarda `length` bytes de `fileobj` como un vídeo de la biblioteca.

    Escribe por trozos a un `.part` y lo renombra al final (atómico): un corte a
    media subida no deja un vídeo roto en la lista. Nunca carga el archivo en
    memoria — la placa tiene poca RAM y un vídeo puede ser de cientos de MB.
    """
    fname = safe_name(name)
    if not fname:
        raise UploadError("falta el nombre del archivo")
    if Path(fname).suffix.lower() not in VIDEO_EXTS:
        raise UploadError("formato no admitido (usa " +
                          ", ".join(sorted(e[1:] for e in VIDEO_EXTS)) + ")")
    if length <= 0:
        raise UploadError("archivo vacío")
    limit = max_bytes()
    if length > limit:
        raise UploadError(f"el archivo pesa más del máximo permitido "
                          f"({limit // (1024 * 1024)} MB)")

    d = videos_dir()
    d.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(d).free
    if length + 256 * 1024 * 1024 > free:   # deja 256 MB de aire al sistema
        raise UploadError("no hay espacio suficiente en el disco del nodo")

    dest = _unique_path(fname)
    tmp = dest.with_name(dest.name + ".part")
    written = 0
    try:
        with tmp.open("wb") as fh:
            while written < length:
                buf = fileobj.read(min(chunk, length - written))
                if not buf:
                    raise UploadError("la subida se cortó antes de terminar")
                fh.write(buf)
                written += len(buf)
        os.replace(tmp, dest)
    except UploadError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise UploadError(f"no se pudo guardar el vídeo: {e}") from e

    log.info("Vídeo subido: %s (%.1f MB).", dest.name, written / 1e6)
    return _entry(dest)


def delete(name: str) -> bool:
    """Borra un vídeo de la biblioteca. True si existía."""
    fname = safe_name(name)
    if not fname:
        return False
    p = videos_dir() / fname
    if not (p.is_file() and p.suffix.lower() in VIDEO_EXTS):
        return False
    try:
        p.unlink()
    except OSError:
        return False
    log.info("Vídeo borrado: %s", fname)
    return True


def content_type(path: Path) -> str:
    return {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
            ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
            ".webm": "video/webm", ".ts": "video/mp2t",
            ".mpg": "video/mpeg", ".mpeg": "video/mpeg"}.get(
        path.suffix.lower(), "application/octet-stream")
