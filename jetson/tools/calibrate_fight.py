"""Calibra los umbrales del detector de peleas sobre TUS vídeos.

Por qué existe: el juez del detector es CLIP, y su salida (`raw`) es una diferencia
de logits, no una probabilidad. **Su escala depende del backbone**: los 3.8 / 6.0
de fábrica se midieron con ViT-L/14, así que al pasar a ViT-B/32 (el que cabe en la
Jetson) esos números no significan lo mismo — el detector puede quedarse mudo o
disparar con cualquier cosa. Esta herramienta mide la escala real corriendo el
pipeline completo sobre vídeos etiquetados por ti y propone los umbrales.

Uso:

    # una pelea y una escena normal (se pueden repetir las banderas, o pasar carpetas)
    uv run python -m tools.calibrate_fight \\
        --fight data/videos/pelea1.mp4 --fight data/videos/pelea2.mp4 \\
        --normal data/videos/pasillo.mp4 \\
        --clip-model ViT-B/32 --write

`--write` guarda `heuristicModels/fight_params.json`, que el modelo lee al arrancar
(y también esta herramienta y el CLI del detector).

Lo que mide, por cada mosaico que el pipeline manda al juez: el `raw` de CLIP. Con
eso compara las dos poblaciones (pelea vs normal) y elige:

  • `F_SET_SCORE_TH`   — el corte que mejor separa las dos poblaciones (máximo F1),
                         que es el umbral que suma confirmaciones.
  • `F_FAST_TRACK_TH`  — el percentil 99 de los mosaicos NORMALES: por encima de
                         ahí, confirmar de golpe es razonable.

Ojo con lo que la herramienta NO puede hacer por ti: si tus vídeos «normales» no
tienen gente moviéndose junta (abrazos, empujones de broma, multitudes), los
umbrales saldrán optimistas. La calidad de la calibración es la de tus ejemplos.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".mpg", ".mpeg", ".ts"}


def _collect(paths: List[str]) -> List[Path]:
    """Expande archivos y carpetas a una lista de vídeos existentes."""
    out: List[Path] = []
    for raw in paths or []:
        p = Path(raw).expanduser()
        if p.is_dir():
            out += sorted(q for q in p.iterdir()
                          if q.is_file() and q.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            out.append(p)
        else:
            print(f"[CAL] No existe, lo salto: {p}")
    return out


class _Collector:
    """Sustituye al despachador asíncrono: aquí el juez corre en el mismo hilo y
    cada `raw` se anota. Calibrar no es tiempo real, así que se prefiere el
    determinismo (todos los mosaicos se juzgan, ninguno se descarta)."""

    def __init__(self, judge):
        self.judge = judge
        self.raws: List[float] = []

    def submit(self, det, zone_id, frames) -> bool:
        t0 = time.perf_counter()
        res = self.judge.predict(frames)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.raws.append(res["raw"])
        det._apply_clip_result(zone_id, res["raw"], dt_ms)
        return True

    def stop(self):
        pass


def _run_video(path: Path, fd_mod, clip_model: str, width: int, every: int,
               max_seconds: float) -> Tuple[List[float], dict]:
    """Corre el pipeline sobre un vídeo y devuelve (raws, info)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[CAL] No se pudo abrir {path.name}")
        return [], {}
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
    det = fd_mod.FightProductionDetector(
        gen_model=fd_mod.META.get("yolo_model", "yolo11n.pt"),
        clip_model_name=clip_model, save=False, save_alerts=False)
    # El juez, en línea: sin cola ni descartes.
    collector = _Collector(det.f_expert)
    det._dispatch = collector

    frames_max = int(max_seconds * fps) if max_seconds > 0 else 0
    idx = read = 0
    t0 = time.perf_counter()
    confirmed_frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        read += 1
        if every > 1 and (read % every):
            continue
        if width and frame.shape[1] > width:      # misma reducción que hace la captura
            h = int(round(frame.shape[0] * width / frame.shape[1]))
            frame = cv2.resize(frame, (width, h), interpolation=cv2.INTER_AREA)
        try:
            conf = det.step(frame, idx)
        except Exception as e:  # noqa: BLE001 — un vídeo raro no debe cortar la tanda
            print(f"[CAL] {path.name}: fallo en el frame {idx}: {e}")
            break
        confirmed_frames += 1 if conf else 0
        idx += 1
        if frames_max and read >= frames_max:
            break
    cap.release()
    dt = time.perf_counter() - t0
    info = {"frames": idx, "read": read, "fps_proc": idx / max(dt, 1e-6),
            "yolo_ms": det.t_yolo_ms, "clip_ms": det.t_clip_ms,
            "mosaicos": len(collector.raws), "frames_confirmados": confirmed_frames,
            "seconds": dt}
    print(f"[CAL] {path.name}: {idx} frames analizados en {dt:.0f}s "
          f"({info['fps_proc']:.1f} fps) · {len(collector.raws)} mosaicos al juez · "
          f"YOLO {det.t_yolo_ms:.0f}ms · CLIP {det.t_clip_ms:.0f}ms")
    return collector.raws, info


def _stats(name: str, xs: List[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=np.float64)
    d = {"n": int(a.size), "min": float(a.min()), "max": float(a.max()),
         "media": float(a.mean()), "p50": float(np.percentile(a, 50)),
         "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)),
         "p99": float(np.percentile(a, 99))}
    print(f"[CAL] {name}: n={d['n']}  min={d['min']:.2f}  p50={d['p50']:.2f}  "
          f"p90={d['p90']:.2f}  p95={d['p95']:.2f}  p99={d['p99']:.2f}  max={d['max']:.2f}")
    return d


def _best_threshold(pos: List[float], neg: List[float]) -> Tuple[float, dict]:
    """Umbral que maximiza F1 separando peleas (pos) de normales (neg).

    Se prueba como candidato cada valor observado: son unos cientos, y así el corte
    sale de los datos y no de una fórmula."""
    if not pos or not neg:
        return 0.0, {}
    cands = sorted(set(round(v, 3) for v in (pos + neg)))
    p = np.asarray(pos)
    n = np.asarray(neg)
    best = (0.0, -1.0, {})
    for th in cands:
        tp = float((p >= th).sum())
        fn = float((p < th).sum())
        fp = float((n >= th).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        if f1 > best[1]:
            best = (th, f1, {"precision": prec, "recall": rec, "f1": f1,
                             "tp": tp, "fp": fp, "fn": fn})
    return best[0], best[2]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Calibra los umbrales del detector de peleas con tus vídeos.")
    ap.add_argument("--fight", action="append", default=[], metavar="RUTA",
                    help="vídeo (o carpeta) CON pelea. Repetible.")
    ap.add_argument("--normal", action="append", default=[], metavar="RUTA",
                    help="vídeo (o carpeta) SIN pelea. Repetible.")
    ap.add_argument("--clip-model", default=None,
                    help="backbone CLIP (por defecto, el de META: %(default)s)")
    ap.add_argument("--width", type=int, default=640,
                    help="ancho al que se reduce cada frame, como hace la captura "
                         "del nodo (0 = resolución nativa). Default: %(default)s")
    ap.add_argument("--every", type=int, default=1, metavar="N",
                    help="analizar 1 de cada N frames (default: %(default)s)")
    ap.add_argument("--max-seconds", type=float, default=0.0, metavar="S",
                    help="tope de segundos por vídeo (0 = completo)")
    ap.add_argument("--write", action="store_true",
                    help="guardar los umbrales en heuristicModels/fight_params.json")
    ap.add_argument("--out", default=None,
                    help="ruta alternativa del JSON de salida")
    args = ap.parse_args()

    fights = _collect(args.fight)
    normals = _collect(args.normal)
    if not fights and not normals:
        ap.error("hace falta al menos un --fight o un --normal")

    import heuristicModels.FightDetector_Production as fd
    clip_model = args.clip_model or fd.META.get("clip_model", "ViT-B/32")
    print(f"[CAL] Backbone CLIP: {clip_model} · ancho de análisis: "
          f"{args.width or 'nativo'} · 1 de cada {args.every} frame(s)")
    print(f"[CAL] Umbrales de partida: score={fd.F_SET_SCORE_TH} "
          f"fast_track={fd.F_FAST_TRACK_TH} mosaico={fd.MOSAIC_FIT}")

    pos: List[float] = []
    neg: List[float] = []
    detail: Dict[str, dict] = {}
    for path in fights:
        raws, info = _run_video(path, fd, clip_model, args.width, args.every,
                                args.max_seconds)
        pos += raws
        detail[path.name] = {"clase": "fight", **info, **_stats(f"  {path.name}", raws)}
    for path in normals:
        raws, info = _run_video(path, fd, clip_model, args.width, args.every,
                                args.max_seconds)
        neg += raws
        detail[path.name] = {"clase": "normal", **info, **_stats(f"  {path.name}", raws)}

    print("\n" + "=" * 62)
    print("  DISTRIBUCIÓN DEL JUEZ CLIP")
    print("=" * 62)
    s_pos = _stats("PELEA  ", pos)
    s_neg = _stats("NORMAL ", neg)

    if not pos or not neg:
        print("\n[CAL] Con una sola clase no hay umbral que proponer: hace falta al "
              "menos un vídeo de cada tipo (--fight y --normal). Los percentiles de "
              "arriba ya sirven para elegir a mano.")
        return 1

    th, metrics = _best_threshold(pos, neg)
    fast = float(np.percentile(np.asarray(neg), 99))
    fast = max(fast, th + 1.0)          # el fast-track siempre por encima del umbral
    print(f"\n[CAL] Propuesta:  F_SET_SCORE_TH={th:.2f}  F_FAST_TRACK_TH={fast:.2f}")
    print(f"[CAL] En los mosaicos medidos: precisión={metrics['precision']:.2f} "
          f"recall={metrics['recall']:.2f} F1={metrics['f1']:.2f} "
          f"(fp={metrics['fp']:.0f} fn={metrics['fn']:.0f})")
    solape = s_pos.get("p50", 0) - s_neg.get("p95", 0)
    if solape <= 0:
        print("[CAL] ⚠ Las dos poblaciones se solapan mucho (la mediana de pelea no "
              "supera el p95 de normal). Con estos vídeos el juez no separa: revisa "
              "que los recortes de pelea tengan movimiento y prueba ViT-B/16.")

    out = {
        "_generado_por": "tools/calibrate_fight.py",
        "_clip_model": clip_model,
        "_mosaic_fit": fd.MOSAIC_FIT,
        "_analisis": {"width": args.width, "every": args.every},
        "_muestras": {"fight": s_pos, "normal": s_neg},
        "_metricas_en_calibracion": metrics,
        "_videos": detail,
        "F_SET_SCORE_TH": round(th, 2),
        "F_FAST_TRACK_TH": round(fast, 2),
    }
    dest = Path(args.out) if args.out else fd.PARAMS_FILE
    if args.write or args.out:
        dest.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[CAL] Escrito {dest}. El nodo lo aplica al reiniciar.")
    else:
        print("\n[CAL] No se escribió nada (usa --write para guardarlo). JSON:")
        print(json.dumps({k: v for k, v in out.items() if not k.startswith("_videos")},
                         indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
