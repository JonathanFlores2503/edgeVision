# Edge — Nodo de detección

Nodo de borde que controla hasta **4 cámaras RTSP/USB**, ejecuta **el modelo que
elijas 100% on-device** (modelo integrado por clip o cualquier plugin de
[`heuristicModels/`](heuristicModels/)), genera **eventos + clips** y los comunica
al cloud (**MQTT** para metadatos + **HTTPS** para clips), con cola local
*offline-first*.

> El modelo es intercambiable. Para enchufar el tuyo, ver [`../docs/PLUGINS.md`](../docs/PLUGINS.md).

## Arquitectura del nodo

```
RTSP (4 cám) ─┐
              ├─ CameraCapture + RingBuffer ──► InferenceWorker (modelo seleccionado)
              │                                      │
              │                                      ├─► Telemetría  ──► MQTT (best-effort)
              │                                      └─► EventEngine (umbral + histéresis)
              │                                              ├─ on_open  ─► Store + Outbox(event)
              │                                              └─ on_close ─► ClipEncoder(.mp4)
              │                                                              + Store + Outbox(event, clip)
              └────────────────────────────────────────────  OutboxDispatcher ─► MQTT / HTTPS
```

## Estructura

```
edge/
├── edge/
│   ├── config.py            # carga/validación de config.yaml (pydantic)
│   ├── types.py             # Clip, ClipResult, Event
│   ├── main.py              # orquestador (python -m edge.main)
│   ├── capture/             # rtsp.py (captura) + ring_buffer.py
│   ├── inference/           # preprocess.py, networks.py, model.py, worker.py
│   ├── events/              # engine.py (histéresis) + clip.py (encoder .mp4)
│   ├── storage/             # db.py (SQLite: eventos + outbox)
│   ├── agent/               # mqtt_client.py, uploader.py, ota.py, dispatcher.py
│   ├── monitor/             # estado + interfaz web (state.py, server.py, index.html)
│   ├── desktop/             # app de escritorio Tkinter (app.py)
│   └── demo.py              # pipeline de demo reutilizable (fuente → scorer → estado)
├── tools/desktop_demo.py    # demo ventana nativa (programa local)
├── tools/live_demo.py       # demo web (navegador / acceso remoto)
├── tools/simulate_offline.py
├── config.example.yaml
├── requirements.txt
└── Dockerfile.jetson
```

## 🖥️ Interfaz local (programa de escritorio)

El nodo trae una **aplicación de escritorio nativa** (Tkinter, no navegador): ventana
con el vídeo anotado por el score, scores en vivo y registro de eventos. Es la interfaz
por defecto (`monitor.ui: desktop`) y será la base del futuro ejecutable `.exe`.

**Demo inmediata con tu webcam (scorer simulado, no requiere torch):**
```bash
python -m tools.desktop_demo --config config.yaml --source 0 --mock
```

Otras fuentes:
```bash
# Carpeta de frames de muestra
python -m tools.desktop_demo --config config.yaml \
  --source "/ruta/a/frames" --mock

# Archivo de vídeo
python -m tools.desktop_demo --config config.yaml --source video.mp4 --mock

# Con el modelo integrado real (requiere torch + pesos en config.yaml)
python -m tools.desktop_demo --config config.yaml --source 0
```

Cuando ejecutas el **nodo completo** (`python -m edge.main`), la ventana se abre
automáticamente (con `monitor.ui: desktop`).

### Alternativa web (acceso remoto / Jetson sin pantalla)
Pon `monitor.ui: web` en `config.yaml` —o usa la demo web— para ver la interfaz en el
navegador (MJPEG + SSE):
```bash
python -m tools.live_demo --config config.yaml --source 0 --mock
```

## 🌐 Dashboard web (monitor)

Interfaz principal en producción (RUBIK Pi 3 headless). Servida por el propio nodo
solo con la librería estándar (sin FastAPI), accesible en la LAN desde teléfono/PC:
**`http://<IP-del-nodo>:<monitor.port>`** (por defecto **`:8090`**; PWA instalable).

> El puerto es `monitor.port` en `config.yaml`. Si otra app ya usa ese puerto, cámbialo.

Funciones:

- **Acceso por sesión (cookie):** cuentas `admin` y `usuario` con contraseña
  *hasheada* (PBKDF2-HMAC-SHA256). El **admin** edita la Configuración; el
  **usuario** solo ve. Las contraseñas por defecto fuerzan cambio al primer ingreso.
- **Gestión de usuarios** (admin): crear y borrar cuentas, cambiar contraseñas
  (no permite borrarse a uno mismo ni quedar sin admin). *Si olvidas la contraseña:
  es irreversible (hash); resetea el `hash` en `data/settings.json` con
  `edge.monitor.settings._hash_pw` y reinicia el nodo.*
- **Cámaras desde la web:** alta/baja de cámaras RTSP, **autodetección USB**
  *plug-and-play*, umbral de anomalía y rotación **por cámara** (se aplican en
  caliente, sin reiniciar el nodo).
- **Ubicación + mapa de México OFFLINE:** lee GPS NEO-6M (NMEA) y dibuja la posición
  del vehículo sobre el contorno de los 32 estados (GeoJSON embebido en
  `edge/monitor/assets/mexico_states.json`, servido local, **sin internet**). Con
  `gps.simulate: true` mueve un punto simulado (centro CDMX) para probar el mapa.
- **Panel de modelo** (admin): muestra backend / modelo / dispositivo y permite
  **elegir detector y clasificador** entre los pesos de `weights_dir` (solo backend
  `torch`; con `onnx` los nombres son fijos). El cambio se aplica **al reiniciar**.
- **Alertas:** SMS/correo/WhatsApp/Telegram con enlace de ubicación (ver `alerts/`).
- **Guardado unificado:** el botón *Guardar configuración* también registra el
  usuario pendiente del formulario y guarda la selección de modelo.

> Nota: editar `index.html` **no** requiere reiniciar (se lee fresco en cada carga);
> cambios en `server.py`/`settings.py`/`main.py` **sí**. La selección de modelo y los
> parámetros de arranque (p. ej. `capture.clip_cooldown_s`) se aplican al reiniciar.

## Puesta en marcha

1. **Config**: `cp config.example.yaml config.yaml` y ajusta cámaras, `model.weights_dir`,
   umbrales y endpoints del cloud. Apunta `weights_dir` a tu carpeta de pesos del modelo integrado.
2. **Dependencias**: en Jetson usa la imagen `Dockerfile.jetson` (torch ya incluido).
   En PC de desarrollo: instala torch/torchvision desde pytorch.org y luego
   `pip install -r requirements.txt`.
3. **Ejecutar**: `python -m edge.main --config config.yaml`

## Verificación sin cámaras ni GPU

Prueba toda la fontanería (eventos → clip → SQLite/outbox) con frames de muestra y
scorer simulado:

```bash
# --repeat replica la secuencia si tienes pocos frames (un clip x3d_l necesita 80)
python -m tools.simulate_offline --config config.yaml \
  --frames-dir "/ruta/a/frames" --mock --repeat 80
```

Con el modelo real (requiere torch + pesos):

```bash
python -m tools.simulate_offline --config config.yaml --video ruta/al/video.mp4
```

## Notas de diseño

- **Inferencia 100% en la Jetson** (~2 GFLOPs). TensorRT/ONNX es optimización opcional
  (`model.use_tensorrt`); el fallback PyTorch siempre funciona.
- **Offline-first**: si el cloud no está disponible, eventos y clips quedan en el
  `outbox` (SQLite) y se reintegran al reconectar (`OutboxDispatcher`).
- `edge/inference/networks.py` define las cabezas del modelo integrado de ejemplo
  sin el `set_default_tensor_type` global (para correr también en CPU).
- **Modelo intercambiable**: el detector integrado es solo un ejemplo. Para enchufar el
  tuyo (caras, objetos, etc.) sin tocar la plataforma, ver [`../docs/PLUGINS.md`](../docs/PLUGINS.md).
```
