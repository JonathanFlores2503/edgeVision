"""Diagnóstico de una cámara RTSP/USB con la MISMA lógica que el nodo.

Abre la fuente forzando TCP (como `edge.main`) y reporta si conecta y cuántos
frames llega a leer. Útil para aislar problemas de red/credenciales/códec sin
levantar toda la plataforma.

Ejemplos:
  python -m tools.test_rtsp "rtsp://usuario:clave@192.168.1.10:554/Streaming/Channels/101"
  python -m tools.test_rtsp 0            # webcam USB índice 0
  python -m tools.test_rtsp /dev/video0
"""
from __future__ import annotations

import argparse
import time

import cv2

from edge.capture.rtsp import CameraCapture
from edge.config import CameraCfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="URL RTSP o índice/ruta de cámara USB")
    ap.add_argument("--transport", default="tcp", choices=["tcp", "udp"])
    ap.add_argument("--frames", type=int, default=30, help="frames a leer antes de salir")
    ap.add_argument("--timeout", type=float, default=15.0, help="segundos máx de espera")
    args = ap.parse_args()

    cam = CameraCfg(id="test", url=args.url, enabled=True, transport=args.transport)
    cap_obj = CameraCapture(cam, model_name="x3d_l", ring_seconds=2.0, on_clip=lambda c: None)

    print(f"Abriendo fuente: {args.url}  (transport={args.transport})")
    print(f"OpenCV {cv2.__version__}")
    t0 = time.time()
    cap = cap_obj._open()
    if cap is None:
        print("\n❌ NO se pudo ABRIR la fuente.")
        print("   Causas típicas: la máquina no llega a esa IP/puerto (red distinta),")
        print("   usuario/clave incorrectos, o el puerto 554 bloqueado por firewall.")
        print("   Comprueba con:  ffplay \"%s\"" % args.url)
        raise SystemExit(1)

    print(f"✅ Fuente ABIERTA en {time.time() - t0:.1f}s. Leyendo frames...")
    ok_count = 0
    last = time.time()
    while ok_count < args.frames and (time.time() - t0) < args.timeout:
        ok, frame = cap.read()
        if not ok or frame is None:
            if time.time() - last > 3.0:
                print("   ...sin frames todavía (stream abierto pero no llegan imágenes).")
                last = time.time()
            continue
        ok_count += 1
        if ok_count == 1:
            h, w = frame.shape[:2]
            print(f"✅ PRIMER frame recibido: {w}x{h}")
    cap.release()

    dt = time.time() - t0
    if ok_count == 0:
        print("\n⚠️  Se ABRIÓ pero NO llegó ningún frame en %.0fs." % dt)
        print("   Suele ser códec no soportado (prueba el substream .../102) o RST por UDP")
        print("   (este test ya usa TCP). Revisa que ffplay muestre imagen de verdad.")
        raise SystemExit(2)

    print(f"\n✅ OK: {ok_count} frames en {dt:.1f}s (~{ok_count/dt:.1f} fps).")
    print("   La cámara funciona desde esta máquina. Si en la plataforma no se ve,")
    print("   asegúrate de correr `python -m edge.main` (NO el demo) y de haberla guardado.")


if __name__ == "__main__":
    main()
