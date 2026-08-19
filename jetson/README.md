# Edge — Nodo de detección · **NVIDIA Jetson**

Nodo de borde que controla hasta **4 fuentes de vídeo** —cámaras **RTSP/USB** o
**vídeos que subas desde el propio dashboard**—, ejecuta **el modelo que
elijas 100% on-device** (modelo integrado por clip o cualquier plugin de
[`heuristicModels/`](heuristicModels/)), genera **eventos + clips** y los comunica
al cloud (**MQTT** para metadatos + **HTTPS** para clips), con cola local
*offline-first*.

> **Este árbol es solo para Jetson** (inferencia `torch` sobre CUDA). El nodo para la
> Thundercomm RUBIK Pi 3 (backend ONNX / NPU Hexagon) vive aparte, en
> [`../rubikpi/`](../rubikpi/), con su propio código: lo que cambies aquí **no** afecta
> a la otra placa. Ver [`../README.md`](../README.md).

> El modelo es intercambiable. Para enchufar el tuyo, ver [`../docs/PLUGINS.md`](../docs/PLUGINS.md).

## Arquitectura del nodo

```
RTSP / USB /  ─┐
vídeo subido   │
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
jetson/
├── edge/
│   ├── config.py            # carga/validación de config.yaml (pydantic)
│   ├── types.py             # Clip, ClipResult, Event
│   ├── main.py              # orquestador (python -m edge.main)
│   ├── capture/             # rtsp.py (captura RTSP/USB/archivo) + video_file.py
│   │                        #   (biblioteca de vídeos subidos) + ring_buffer.py + manager.py
│   ├── inference/           # preprocess.py, networks.py, model.py, worker.py, registry.py
│   ├── events/              # engine.py (histéresis) + clip.py (encoder .mp4)
│   ├── storage/             # db.py (SQLite: eventos + outbox)
│   ├── agent/               # mqtt_client.py, uploader.py, ota.py, dispatcher.py
│   ├── alerts/              # notifier.py, telegram.py, message.py
│   ├── gps/                 # neo6m.py (NMEA del NEO-6M)
│   ├── monitor/             # estado + dashboard web (state.py, server.py, index.html)
│   └── demo.py              # pipeline de demo reutilizable (fuente → scorer → estado)
├── heuristicModels/         # modelos enchufables (ver ../docs/PLUGINS.md)
├── Models/ContadorFlujo/    # herramientas del contador de flujo (zonas, web propia)
├── tools/live_demo.py       # demo web (navegador / acceso remoto)
├── tools/simulate_offline.py
├── tools/test_rtsp.py
├── config.example.yaml
├── pyproject.toml           # dependencias (uv) + extras `model` y `heuristic`
├── uv.lock                  # entorno reproducible; se versiona
├── .python-version          # 3.8 — impuesto por el torch con CUDA de JetPack
└── Dockerfile.jetson
```

## 🖥️ Interfaz del nodo — cómo abrirla

La interfaz es el **dashboard web** que sirve el propio nodo (ver la sección siguiente);
la Jetson corre headless y se ve desde el teléfono/PC en la LAN.

### ▶️ CON ESTE COMANDO SE CORRE LA PLATAFORMA CON EL MODELO

```bash
cd jetson
uv run python -m edge.main --config config.yaml --force
```

Eso es todo. Arranca el nodo **con el modelo que dejaste seleccionado** en el panel de
admin (guardado en `data/settings.json` → `model_selection`), con **las cámaras
guardadas** y con el panel **10 · Testigos** (galería de evidencia por evento) visible.

Luego abre en el navegador **`http://<IP-de-la-Jetson>:8080`** (o `http://localhost:8080`
si estás en la propia placa; el puerto es `monitor.port` de `config.yaml`). El servidor
escucha en `0.0.0.0`, así que se ve desde cualquier equipo de la LAN. Deja la terminal
abierta: el dashboard vive mientras corra ese proceso; se cierra con `Ctrl+C`.

Detalles que importan de ese comando:

- **Sin `--mock`.** `--mock` fuerza el scorer simulado y **descarta tu selección de
  modelo** (`edge/main.py`: con `mock=True` la selección pasa a `{"family": "vad",
  "key": "mock"}`). Si arrancas con `--mock`, no hay modelo real ni Testigos.
- **`--force`** detiene el nodo que ya tenga tomado el 8080 y arranca este en su lugar
  (solo mata procesos `edge.main`; nunca otros servicios). Sin él, si el puerto está
  ocupado el arranque aborta con `Address already in use`. A mano sería
  `pkill -f 'edge.main'`.
- **Lánzalo desde `jetson/`.** `storage.db_path` y `clips_dir` se resuelven contra el
  directorio actual, no contra el `config.yaml`: desde otra carpeta verás una base de
  datos vacía y sin eventos.
- El panel **Testigos** solo aparece si el modelo vivo guarda evidencia (p. ej. el
  Contador de flujo). Con el VAD integrado se oculta, porque ese modelo no las genera.

### 🩺 Si el dashboard no responde o el nodo no cierra

Tres cosas que conviene saber antes de perseguir un fallo que no existe.

**"Error de conexión" al iniciar sesión casi siempre es RAM, no el login.** La placa
tiene 7.4 GB compartidos con el escritorio y el Fight Detector se lleva ~4.5 GB con
**una sola cámara** (medido). Al agotarse la memoria, el bucle de `accept` del dashboard
se cae: el proceso sigue vivo, el puerto sigue en LISTEN y el navegador no recibe nada.
Diagnóstico en un segundo:

```bash
ss -ltn | grep :8080     # si Recv-Q sube y no baja, el servidor no está aceptando
free -m                  # si "available" baja de ~1 GB, es esto
```

Desde el 2026-08-19 el nodo **se recupera y lo dice** (`El bucle del dashboard se cayó;
reintento en …`) y avisa antes de llegar ahí (`RAM baja: … MB disponibles`). Si ves
`[FIGHT] Sin RAM para '<cámara>'`, no es un error: el modelo se negó a levantar esa
cámara para no tumbar el nodo, y sigue trabajando con las demás. Para usar dos cámaras
hay que darle memoria de verdad — arrancar la placa sin escritorio (`multi-user.target`)
libera ~1.5 GB — y subir `mem_budget_mb` en `heuristicModels/FightDetector_Production.py`.

**Ctrl+C ya cierra el nodo** (~2 s). Si alguna vez no lo hace, la segunda señal fuerza
la salida y deja en el log la pila de los hilos que no cerraron.

**Si algo se queda colgado, pide el volcado antes de matarlo.** El nodo imprime al
arrancar `Diagnóstico de cuelgues: kill -USR1 <pid> …`:

```bash
kill -USR1 <pid>     # pila de TODOS los hilos, al log del nodo
```

Funciona incluso con el intérprete congelado, porque escribe desde el manejador de señal
sin pedir el GIL. Es lo primero que hay que hacer: dice quién está bloqueado y en qué
línea. (`gdb` no sirve en esta placa: `ptrace_scope=1`.)

### 🎬 Sin cámara: sube un vídeo y analízalo como si lo fuera

A diferencia de `tools.live_demo` (más abajo), esto **sí es la plataforma completa**:
el modelo que tengas seleccionado, eventos, clips, alertas, mapa… Lo único que cambia
es de dónde salen los frames.

1. Entra al dashboard como **admin** → *Configuración* → sección **3 · Videos**.
2. Elige el archivo (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.ts`…) y pulsa
   **Subir vídeo**. Se guarda en `storage.videos_dir` (`data/videos/`) escribiendo a
   disco por trozos, nunca entero en RAM: la placa tiene poca y el archivo puede pesar
   cientos de MB. Tope por archivo: `storage.max_video_mb` (2 GB por defecto).
3. Pulsa **Usar como cámara** y luego **Guardar configuración**. La fuente arranca
   **en caliente**, sin reiniciar el nodo, y sale en el dashboard como una cámara más
   (vista en vivo, umbral y rotación propios, eventos, clips y alertas incluidos).

El vídeo se reproduce **en bucle** y **a su velocidad real** (los fps del contenedor),
no tan rápido como lo lea el disco: así las marcas de tiempo de los eventos tienen
sentido y la cola de inferencia no se satura. Para que se detenga al terminar en vez de
repetirse, pon `loop: false` en esa cámara del `config.yaml`.

También puedes escribir la fuente a mano en la sección *Cámaras*: `file://mi-video.mp4`
(se busca en `data/videos/`) o una ruta absoluta, `file:///ruta/al/video.mp4`.

> Los vídeos subidos viven en `data/`, que está fuera de git: **son de cada nodo**.

### 🧪 Demo web sin cámara — `tools.live_demo` (NO carga el modelo seleccionado)

⚠️ **Esta NO es la plataforma con tu modelo.** `tools.live_demo` solo sabe de dos
scorers: el simulado (`--mock`) y el VAD integrado (TransLowNet). **No lee
`model_selection`**, así que nunca arranca el Contador de flujo ni ningún otro plugin de
`heuristicModels/`, y por lo mismo **no muestra Testigos**. Úsalo solo para probar la
fontanería del dashboard con un vídeo cuando no hay cámara conectada:

```bash
cd jetson
uv run python -m tools.live_demo --config config.yaml --source data/sample.mp4 --no-browser
```

Prueba rápida **sin torch ni pesos** (scorer simulado, arranca en segundos):

```bash
uv run python -m tools.live_demo --config config.yaml \
  --source data/sample.mp4 --mock --clip-frames 12 --no-browser
```

> `data/sample.mp4` es un vídeo de muestra para estas pruebas. **No lo pongas dentro de
> `data/clips/`**: esa carpeta es la *salida* del nodo (`storage.clips_dir`) y se limpia;
> si tu copia se borra, regenérala desde cualquier clip grabado:
> `cp data/clips/<algún-clip>.mp4 data/sample.mp4`.

Fuentes válidas para `--source`:

| Valor | Qué es |
|---|---|
| `ruta/al/video.mp4` | archivo de vídeo, se repite en bucle — **la opción segura sin hardware** |
| `ruta/a/carpeta` | carpeta de frames (`.jpg`/`.png`), en bucle a 15 fps |
| `rtsp://user:pass@ip:554/stream` | cámara IP |
| `0`, `1`, … | webcam USB por índice — **solo si existe `/dev/video0`** |

Otras banderas útiles: `--port 8081` (si el 8080 está ocupado), `--no-browser` (no intenta
abrir un navegador, obligatorio en headless), `--no-second-cam` (no simula la segunda
cámara espejada), `--clip-frames N` (solo en `--mock`; con el modelo real el tamaño de clip
lo fija el backbone: 80 frames para `x3d_l`).

### ⚠️ Si `live_demo` se cierra solo al arrancar

Síntoma: verás `can't open camera by index` / `Camera index out of range` y acto seguido
`DemoPipeline detenido.` + `Demo finalizada.`

```
[ WARN:0@5.363] global cap_v4l.cpp:914 open VIDEOIO(V4L2:/dev/video0): can't open camera by index
```

**Causa:** `--source 0` pide una webcam USB que no está conectada. `frame_source()`
([`edge/demo.py`](edge/demo.py)) lanza `SystemExit`, muere el hilo del pipeline y
`tools/live_demo.py` —que corre mientras el hilo esté vivo— apaga el servidor web detrás.
No es un fallo del modelo: fíjate que en tu log el modelo **sí** cargó
(`TransLowNet listo en cuda`), y el dashboard alcanzó a anunciarse antes de caer.

**Solución:** pásale una fuente que exista. Usa el comando recomendado de arriba, o
comprueba primero si hay cámara:

```bash
ls /dev/video*            # si dice "No such file or directory" → no hay webcam, usa un vídeo
v4l2-ctl --list-devices   # detalle de los dispositivos de vídeo
```

*(En esta placa, hoy, no hay ninguna: `/dev/video*` no existe y solo aparece el
`/dev/media0` del ISP de Tegra, que no es una cámara utilizable por índice.)*

## 🌐 Dashboard web (monitor)

Interfaz principal en producción (Jetson headless). Servida por el propio nodo
solo con la librería estándar (sin FastAPI), accesible en la LAN desde teléfono/PC:
**`http://<IP-del-nodo>:<monitor.port>`** (por defecto **`:8080`**; PWA instalable).

> El puerto es `monitor.port` en `config.yaml`. Si otra app ya usa ese puerto, cámbialo.
> **Ojo:** eso vale para `python -m edge.main`. `tools.live_demo` **no lee** `monitor.port`:
> usa su propia bandera `--port` (8080 por defecto).

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
- **Videos como fuente** (admin): sube un archivo desde el navegador y úsalo como
  cámara (`file://nombre.mp4`), en bucle y a velocidad real. Para demos, pruebas de
  un modelo o revisar una grabación **sin ninguna cámara conectada**.
- **Ubicación + mapa de México OFFLINE:** lee GPS NEO-6M (NMEA) y dibuja la posición
  del vehículo sobre el contorno de los 32 estados (GeoJSON embebido en
  `edge/monitor/assets/mexico_states.json`, servido local, **sin internet**). Con
  `gps.simulate: true` mueve un punto simulado (centro CDMX) para probar el mapa.
- **Panel de modelo** (admin): muestra backend / modelo / dispositivo y permite
  **elegir detector y clasificador** entre los pesos `.pkl` de `weights_dir`.
  El cambio se aplica **al reiniciar**.
- **Alertas:** SMS/correo/WhatsApp/Telegram con enlace de ubicación (ver `alerts/`).
- **Guardado unificado:** el botón *Guardar configuración* también registra el
  usuario pendiente del formulario y guarda la selección de modelo.

> Nota: editar `index.html` **no** requiere reiniciar (se lee fresco en cada carga);
> cambios en `server.py`/`settings.py`/`main.py` **sí**. La selección de modelo y los
> parámetros de arranque (p. ej. `capture.clip_cooldown_s`) se aplican al reiniciar.

## Puesta en marcha — con `uv`

Las dependencias se manejan **solo con [uv](https://docs.astral.sh/uv/)**: nada de conda
ni de `pip install` a mano. La fuente de verdad es [`pyproject.toml`](pyproject.toml) y el
`uv.lock` versionado.

```bash
cd jetson
uv sync                                   # crea el entorno y lo deja igual al lock
cp config.example.yaml config.yaml        # ajusta cámaras, weights_dir, umbrales, cloud
uv run python -m edge.main --config config.yaml --force    # ← PRODUCCIÓN, con el modelo real
```

Ese último comando es el de producción — el mismo de la sección
*▶️ CON ESTE COMANDO SE CORRE LA PLATAFORMA CON EL MODELO* de más arriba. Cámbialo por
`--mock` **solo** si quieres arrancar sin GPU ni cámaras para probar la fontanería: ese modo sustituye el modelo por un scorer simulado y descarta tu selección
del panel de admin.

`uv sync` materializa el entorno en `.venv/` (es lo que hace uv por dentro), pero **no lo
actives nunca a mano**: `uv run` lo selecciona solo y, de paso, resincroniza si el lock
cambió. Todo comando del nodo va prefijado con `uv run`.

**Python 3.8 es obligatorio aquí**, y no por gusto: el único torch con CUDA de la placa es
el wheel de NVIDIA (`2.1.0a0+…nv23.06`, cp38) que trae JetPack. Con un Python más nuevo
solo hay torch de PyPI, que en aarch64 viene sin CUDA. Por eso `requires-python` está
clavado en `>=3.8,<3.9` y hay un `.python-version`.

### Extras

```bash
uv sync --extra cuda                    # torch con CUDA (wheel de NVIDIA, ~3 GB)
uv sync --extra cuda --extra model      # + VAD integrado: pytorchvideo + fvcore (X3D)
uv sync --extra heuristic               # plugins: solo lo que no depende de torch
```

**`ultralytics` y CLIP no se pueden meter en el lock.** No es pereza: dependen de
`torchvision`, y en PyPI no hay ningún `torchvision` compatible con el torch de JetPack
(los de PyPI se anclan a versiones de torch de PyPI). uv lo declara insatisfacible. El
rodeo —instalarlos con `--no-deps` y aportar un torchvision compilado para JetPack— está
detallado en [`pyproject.toml`](pyproject.toml). **En esta placa ese rodeo ya está hecho**
(`ultralytics`, `lap`, `clip` y el `torchvision` compilado están en el entorno), así que el
Contador de flujo y el Fight Detector aparecen **habilitados** en el panel de admin. En una
placa nueva hay que repetirlo o los dos saldrán deshabilitados.

**torch no viene de PyPI**: el `pyproject.toml` lo apunta por URL al wheel de JetPack en
`[tool.uv.sources]`. Se declara ahí y **nunca con `uv pip install` suelto**, porque
`uv sync` desinstala todo lo que no esté en el lock — un torch metido a mano
desaparecería en el siguiente sync. Verifica siempre después:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# tiene que imprimir True
```

## 🙂 Reconocimiento facial (`faces`) — rostros con nombre

`heuristicModels/faces.py`. **Es un adaptador, no una copia**: el modelo vive en
Full_CV (`models/faces/model.py` + el motor compartido `models/facecore/`) y aquí solo
se traduce al contrato del nodo. SCRFD det_10g detecta los rostros, un tracker por IoU
les da id estable, **AdaFace IR101** los embebe y el nombre se decide **por voto** a lo
largo del track, así que un frame malo no cambia la identidad. En la misma pasada
estima sexo y edad aproximada. Se envuelve en vez de copiarse para que un arreglo en
Full_CV llegue aquí sin portarlo a mano.

Lo que hace falta en la placa:

| Pieza | Dónde | Cómo se consigue |
|---|---|---|
| `insightface` + `onnxruntime-gpu` | entorno de uv | `uv pip install insightface` + `uv pip install vendor/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl` (extra `faces` del `pyproject.toml`) |
| SCRFD det_10g (`buffalo_l`) | `Full_CV/models/facecore/face_runner/weights/insightface/` | **se descarga solo** la 1ª vez (~275 MB) |
| AdaFace IR101 (~250 MB) | `weights/Adaface/adaface_ir101_webface12m.pt` | **manual**: no se autodescarga |
| `genderage.onnx` (opcional) | `…/face_runner/weights/genderage/` | viene dentro de `buffalo_l`; hay que **copiarlo** a esa carpeta o la demografía sale OFF |

> `uv pip install` y no `uv sync --extra faces` a propósito: este árbol tiene
> ultralytics/CLIP/torchvision metidos a mano fuera del lock y `uv sync` los borraría.
> Comprobado que estas dos deps **no** tocan numpy (el torch de JetPack exige <1.25).

### La galería: quién tiene nombre

Una subcarpeta por persona dentro de `data/rostros/`, con 2-4 fotos de frente:

```
data/rostros/
├── Juan Pérez/foto1.jpg, foto2.jpg
└── María López/foto1.jpg
```

Se construye al arrancar y se cachea en `data/rostros/face_gallery.npy`. Si añades gente,
borra ese `.npy` y reinicia. **Sin galería todos los rostros salen DESCONOCIDO** (el nodo
lo avisa en el log). Al ser `data/`, la galería es de cada dispositivo y está fuera de git.

### Qué dispara un evento

Se elige con `alerta` en el `META` del plugin:

- **`"reconocido"`** (por defecto): la galería es una **lista de búsqueda** — el evento
  salta cuando aparece alguien registrado. Es lo que suele querer transporte público.
- **`"desconocido"`**: la galería es la lista de **autorizados** — el evento salta con un
  rostro sin nombre que persiste. Es la semántica original del modelo (control de acceso).
  En un autobús esto dispararía con cada pasajero.

El `score` que se publica es la **confianza de identidad** (el coseno del voto, ~0.70-0.80
en pruebas), no la de detección de cara. Por eso el umbral por cámara del dashboard actúa
como un segundo filtro de identidad más estricto que el `thr` del plugin (0.45): puedes
apretarlo por cámara sin tocar el archivo.

### Rendimiento medido en esta placa (2026-08-19)

**SCRFD ya corre en GPU** (AdaFace también, en CUDA). Medido sobre frames de 848x480, con
`onnxruntime-gpu 1.16.0` y con el `onnxruntime` de PyPI (solo CPU) que había antes:

| `det_size` | CPU (antes) | **GPU (ahora)** | mejora |
|---|---|---|---|
| 640 | 570 ms · 1.8 FPS | **31.1 ms · 32 FPS** | 18× |
| **320** (el que usa el plugin) | 201 ms · 5.0 FPS | **13.8 ms · 73 FPS** | 14.6× |

Las detecciones en GPU salen **idénticas bit a bit** a las de CPU (bbox y `det_score`
comprobados sobre la galería), así que el cambio es de velocidad y nada más.

Con ese margen el plugin pasó a **`det_size: 640`**, y no por gusto: medido sobre la cámara
real de la oficina (plano cenital, 1920x1080, caras de 33-60 px), a 320 salían **6 caras en
6 frames y dos frames se quedaban a cero**; a 640 salen **14**. A 960 salen 21, pero las 7
nuevas miden 13-14 px — las tira `min_face` y un rostro de 14 px no da un embedding
utilizable, así que 960 solo tiene sentido en plano general.

> **Un plano cenital limita el reconocimiento, no el código.** SCRFD necesita ver la cara
> más o menos de frente: alguien mirando su portátil desde arriba no tiene rostro
> detectable a ninguna resolución. En esa escena el nodo cuenta a la persona (YOLO) y le
> pone nombre solo cuando levanta la vista. Si el objetivo es identificar a todo el que
> pasa, la cámara tiene que estar a la altura de la cara, no en el techo.

El wheel `onnxruntime-gpu` para JetPack 5.x/cp38 **no está en PyPI ni en el índice
`redist` de NVIDIA** (ese solo publica `pytorch/` y `tensorflow/`): sale del
[Jetson Zoo](https://elinux.org/Jetson_Zoo#ONNX_Runtime) y ya está guardado en
`vendor/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl`. Dos avisos:

- **Usar el build de JetPack 5.1.1 (1.16.0), no los de 5.1.2** (1.17.0 / 1.18.0): esos se
  compilaron con GCC 11 y piden `GLIBCXX_3.4.29`, que Ubuntu 20.04 no tiene (llega a
  3.4.28); fallan al importar con un `ImportError` de `libstdc++`.
- La demografía (`genderage.onnx`) se apagaba al instalar el runtime con GPU: `insightface`
  crea su `InferenceSession` sin `providers`, y ORT ≥ 1.9 lo rechaza en un build con GPU.
  Arreglado en `Full_CV/models/facecore/attrs.py`, que ahora pasa `CPUExecutionProvider`
  explícito (el modelo son 1.3 MB, no vale la pena en GPU).

Coste en RAM: el nodo con `faces` y una cámara queda en **4.5 GB de RSS** y deja ~1.4 GB
libres en la placa. El contexto CUDA de onnxruntime no es gratis — ver
`docs/architecture.md` y el presupuesto de RAM del plugin.

### Prueba de que funciona

Con la galería de ejemplo (4 personas), dejando **fuera** una foto de cada una y
preguntando por esas: **4/4 reconocidas**, coseno 0.70-0.79 contra un umbral de 0.45.

## 🥊 Detector de peleas (YOLO + CLIP) — afinado para la Jetson

`heuristicModels/FightDetector_Production.py`. Mismo pipeline que el modelo `fighting`
de Full_CV (pares de personas → gate de movimiento → zonas → mosaico temporal 3×2 →
juez CLIP → confirmación por consecutivos), pero con el envoltorio multi-cámara del
nodo y ajustado a lo que la placa aguanta.

**Activarlo:** *Configuración* → **9 · Modelo de detección** → *Fight Detector
(YOLO + CLIP)* → Guardar → **reiniciar el nodo** (el modelo se elige al arrancar).
Es uno **u** otro: si activas peleas, el contador de personas no corre.

### Qué se cambió respecto al pipeline original, y por qué

| Cambio | Motivo en la placa |
|---|---|
| **Un solo CLIP para todo el proceso** (`shared_judge`), no uno por cámara | Antes cada cámara cargaba su propia copia del mismo modelo: 368 MB de VRAM ×N y N hilos peleándose por la única GPU |
| **Juez con batching** (`_ClipDispatcher`, hasta 4 mosaicos por forward) | Medido en Orin con ViT-B/32 fp16: **34.2 ms** por mosaico suelto → **11.3 ms** en batch de 4 (**3.0×**) |
| **ViT-B/32 fp16** en vez de ViT-L/14 | ViT-L/14 son ~890 MB de pesos; no cabe junto al resto. `clip.load` ya entrega fp16 en CUDA (no hay que llamar a `.half()`: rompe los LayerNorm que CLIP deja en fp32) |
| **Preprocesado con OpenCV**, sin PIL | La conversión numpy→PIL→tensor costaba CPU por cada mosaico, y la CPU es lo disputado |
| **Mosaico completo** al juez (`MOSAIC_FIT="squash"`) | El `preprocess` de CLIP recorta el centro y **tiraba los bordes de las columnas laterales** del mosaico: el juez no veía las seis fases del gesto. Se puede volver al recorte con `MOSAIC_FIT="crop"` |
| **Umbrales espaciales escalados** al ancho real del frame | Los 300 px de separación se calibraron a 1280 de ancho; el nodo entrega ~640 px (`capture.proc_short_side`), donde 300 px son media pantalla y cualquier par formaba zona. Ahora se reescalan solos (se ve en el log: `Escala espacial para 476px de ancho (x0.37)`) |
| **Pesos YOLO resueltos en el árbol** (`jetson/yolo11n.pt`) | Ultralytics los resolvía contra el CWD y **volvía a descargarlos** dejando el `.pt` tirado en la carpeta desde donde se lanzó el nodo |
| **Sin `ALERTAS_FIGHT/` ni hilo de `imwrite`** en el nodo | La plataforma ya recorta y guarda el clip del evento; escribir JPEGs a la SD en el camino caliente era coste sin uso |

Coste medido con una cámara y un vídeo de 30 fps: **~3.3 GB de RSS** del proceso,
417 MB de VRAM, YOLO ~114 ms/frame y CLIP ~84 ms/mosaico *compitiendo con otro nodo
en la placa*. Si vas a correr varias cámaras, sube `infer_every_n` en el `META` del
plugin (analiza 1 de cada N frames) antes de tocar nada más.

### Calibrar los umbrales — obligatorio al cambiar de backbone

El juez devuelve una **diferencia de logits**, no una probabilidad, y su escala
depende del backbone: los `3.8` / `6.0` de fábrica se midieron con ViT-L/14. Con
ViT-B/32 hay que medir de nuevo, y para eso están tus propios vídeos (los mismos que
subes en *Videos*):

```bash
uv run python -m tools.calibrate_fight \
    --fight  data/videos/pelea1.mp4 --fight data/videos/pelea2.mp4 \
    --normal data/videos/pasillo.mp4 --normal data/videos/andén.mp4 \
    --width 640 --write
```

Corre el pipeline completo, recoge el `raw` de **cada** mosaico que llega al juez y
propone `F_SET_SCORE_TH` (el corte que maximiza F1 entre las dos poblaciones) y
`F_FAST_TRACK_TH` (percentil 99 de los mosaicos normales). Con `--write` deja
`heuristicModels/fight_params.json`, que el modelo aplica al arrancar — y avisa en el
log de qué umbrales cargó. Sin ese archivo se usan los de fábrica.

Como referencia de la escala: en un vídeo de interior de autobús **sin** peleas, el
juez ViT-B/32 dio `raw` entre **0.4 y 3.1**. Hacen falta vídeos **con** pelea para
fijar el corte; con una sola clase la herramienta te da los percentiles y no propone
nada, a propósito.

## Verificación sin cámaras ni GPU

Prueba toda la fontanería (eventos → clip → SQLite/outbox) con frames de muestra y
scorer simulado:

```bash
# --repeat replica la secuencia si tienes pocos frames (un clip x3d_l necesita 80)
uv run python -m tools.simulate_offline --config config.yaml \
  --frames-dir "/ruta/a/frames" --mock --repeat 80
```

Con el modelo real (requiere torch + pesos):

```bash
uv run python -m tools.simulate_offline --config config.yaml --video ruta/al/video.mp4
```

> **Ojo con las rutas de `storage`.** `db_path` y `clips_dir` del `config.yaml` se resuelven
> **relativas al directorio desde el que lanzas el nodo**, no al config. Lanzarlo desde otra
> carpeta te crea una base de datos distinta. (`data/settings.json` sí se ancla junto al
> `config.yaml`.) Usa rutas absolutas si vas a lanzarlo desde un servicio systemd.

## Notas de diseño

- **Inferencia 100% en la Jetson** (~2 GFLOPs), siempre con **torch sobre CUDA**
  (`model.device: cuda`). El flag `model.use_tensorrt` está reservado para la
  optimización TensorRT: **aún no está implementado** en `inference/model.py`, así que
  hoy no cambia nada. El camino ONNX no existe en este árbol (es el de `../rubikpi/`).
- **Offline-first**: si el cloud no está disponible, eventos y clips quedan en el
  `outbox` (SQLite) y se reintegran al reconectar (`OutboxDispatcher`).
- `edge/inference/networks.py` define las cabezas del modelo integrado de ejemplo
  sin el `set_default_tensor_type` global (para correr también en CPU).
- **Modelo intercambiable**: el detector integrado es solo un ejemplo. Para enchufar el
  tuyo (caras, objetos, etc.) sin tocar la plataforma, ver [`../docs/PLUGINS.md`](../docs/PLUGINS.md).
```
