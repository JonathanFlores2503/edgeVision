"""Demo WEB (navegador) — interfaz en http://localhost:<port>.

Equivalente a desktop_demo pero sirviendo la interfaz por web (MJPEG + SSE).
Útil para ver el nodo en remoto (p. ej. la Jetson sin pantalla).

Ejemplos:
  python -m tools.live_demo --config config.yaml --source 0 --mock
  python -m tools.live_demo --config config.yaml \
      --source "D:/Codes/TransLowNet_Jetson/Propuesta2/FrameStream" --mock
"""
from __future__ import annotations

import argparse
import logging
import threading
import time
import webbrowser

from edge.alerts import AlertNotifier
from edge.config import load_config
from edge.demo import DemoPipeline
from edge.monitor.server import MonitorServer
from edge.monitor.settings import Settings, default_settings
from edge.monitor.state import MonitorState

log = logging.getLogger("live_demo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--source", default="0", help="webcam (0,1...) | ruta vídeo | carpeta frames")
    ap.add_argument("--mock", action="store_true", help="scorer simulado (sin torch)")
    ap.add_argument("--camera-id", default="demo-cam")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--clip-frames", type=int, default=24)
    ap.add_argument("--no-second-cam", action="store_true",
                    help="no simular una segunda cámara (imagen espejada)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)

    state = MonitorState(cfg.events.score_threshold)
    settings = Settings("./data/settings.json", defaults=default_settings(cfg))
    notifier = AlertNotifier(settings)
    server = MonitorServer(state, port=args.port, settings=settings)
    server.start()
    url = f"http://localhost:{args.port}"
    log.info("➜  Abre la interfaz en %s", url)
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001
            pass

    second_cam = None if args.no_second_cam else f"{args.camera_id}-2"
    pipeline = DemoPipeline(cfg, state, args.source, mock=args.mock,
                            camera_id=args.camera_id, clip_frames=args.clip_frames,
                            notifier=notifier, second_camera_id=second_cam)
    worker = threading.Thread(target=pipeline.run, name="demo-pipeline", daemon=True)
    worker.start()

    try:
        while worker.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        server.stop()
        log.info("Demo finalizada.")


if __name__ == "__main__":
    main()
