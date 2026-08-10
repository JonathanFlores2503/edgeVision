# Edge en RUBIK Pi 3 (Qualcomm QCS6490)

Variante para correr el nodo VAD en la **Thundercomm RUBIK Pi 3** (SoC Qualcomm,
CPU Cortex-A78/A55, GPU Adreno, **NPU Hexagon**) **sin tocar el camino Jetson**.

La diferencia con el Jetson es el **backend de inferencia**: en vez de torch +
TensorRT, esta variante usa **onnxruntime**. Se selecciona en `config.yaml`:

```yaml
model:
  backend: onnx          # torch (Jetson/PC, default) | onnx (RUBIK Pi 3)
  onnx_dir: ./data/onnx  # carpeta donde quedan los .onnx exportados
  name: x3d_l
  weights_dir: ...       # solo se usa para EXPORTAR; el runtime no lo necesita
  ...
```

onnxruntime elige el Execution Provider automáticamente:

1. **QNNExecutionProvider** → NPU **Hexagon** (rápido). Requiere QNN SDK + un
   onnxruntime compilado con `--use_qnn` (ver más abajo).
2. **CPUExecutionProvider** → fallback. **Funciona hoy** sin SDK, en los Cortex-A78
   (más lento; sirve para validar el pipeline real end-to-end).

El mismo código pasa de CPU a Hexagon en cuanto el QNN SDK esté presente.

---

## Puesta en marcha (CPU — funciona ya)

```bash
# 1. Dependencias de runtime (SIN torch)
pip install -r requirements.rubikpi.txt

# 2. Exportar los .onnx UNA vez (esto sí necesita torch; puede ser en un PC de dev)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.rubikpi-export.txt
python -m tools.export_onnx --config config.yaml --out-dir ./data/onnx

# 3. Ajustar config.yaml -> model.backend: onnx  (y model.onnx_dir)
# 4. Correr el nodo
python -m edge.main --config config.yaml
```

> Si solo quieres probar cámaras/eventos/interfaz sin modelo, sigue estando
> `python -m tools.desktop_demo --config config.yaml --source 0 --mock`.

---

## Activar la NPU Hexagon (acelerado)

`onnxruntime` de PyPI **no** trae el QNN EP en Linux aarch64. Para Hexagon:

1. Descarga el **Qualcomm AI Engine Direct (QNN) SDK** (requiere cuenta Qualcomm).
2. Obtén/compila un **onnxruntime con `--use_qnn`** contra ese SDK
   (o usa **Qualcomm AI Hub** para compilar el modelo a un binario Hexagon).
3. Exporta la ruta del backend HTP antes de correr:
   ```bash
   export QNN_BACKEND_PATH=/ruta/al/qnn/libQnnHtp.so
   python -m edge.main --config config.yaml
   ```
4. Al arrancar, el log debe mostrar `Providers efectivos: ['QNNExecutionProvider', ...]`.

### Notas / límites conocidos
- El extractor **X3D usa convoluciones 3D**; el soporte de Conv3D en el HTP de
  Hexagon es limitado. Puede requerir cuantización (QNN quantizer) o dejar el X3D
  en CPU y solo las cabezas en NPU. Las cabezas (`detector`, `classifier`) son
  `Linear` y mapean bien a la NPU.
- `tools/export_onnx.py` genera 3 ONNX independientes (X3D, detector, clasificador)
  precisamente para poder asignar provider por etapa si hiciera falta.

---

## Qué NO cambia del Jetson
- `model.py` (torch + TensorRT), `requirements-model.txt`, `Dockerfile.jetson`:
  intactos. Con `model.backend: torch` (default) el Jetson corre exactamente igual.
- Toda la fontanería (captura RTSP/USB, eventos, clips, SQLite, outbox, MQTT/HTTPS,
  monitor/desktop) es compartida: solo se intercambia la clase de inferencia.

---

## Notas específicas de la RUBIK Pi 3

- **Dashboard web** en `http://<IP-de-la-placa>:8090` (`monitor.port`). Es la interfaz
  principal (placa headless). Ver funciones en [`README.md`](README.md#-dashboard-web-monitor).
- **Panel de modelo (admin):** con `backend: onnx` los tres ONNX tienen **nombre fijo**
  (`x3d_features.onnx`, `detector.onnx`, `classifier.onnx`) en `onnx_dir/<name>/`, así que
  el selector de modelo del dashboard aparece **solo lectura**. Para cambiar de modelo en
  ONNX hay que **re-exportar** con `tools.export_onnx` (el cambio por dropdown es solo para
  `backend: torch`).
- **Cadencia de clips en CPU:** la inferencia en CPU (~2.6 s/clip, x3d_l) no alcanza a
  procesar clips contiguos de varias cámaras y la cola se satura (descarta clips). Ajusta
  `capture.clip_cooldown_s` (la RUBIK Pi usa `4.0`) para espaciar los clips inferidos sin
  pausar el vídeo en vivo. Regla: `clip_cooldown_s ≳ (Nº cámaras × seg/clip) − (80/fps)`.
- **GPS / mapa:** el mapa de México del dashboard funciona **sin internet** (GeoJSON
  embebido). Para probarlo sin hardware, pon `gps.simulate: true` (mueve un punto en CDMX);
  con el NEO-6M real, `gps.enabled: true` y el `gps.port` correcto.
