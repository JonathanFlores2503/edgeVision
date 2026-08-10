# Arquitectura — Plataforma VAD para Transporte Público (PhD Final)

> Documento de diseño. **El nodo *edge* ya está implementado y operativo**; el *cloud*
> (backend FastAPI + frontend React) es la fase siguiente. Ver "Estado de implementación".

## Estado de implementación (act. 2026-06-29)

**✅ Edge — operativo en dos placas:**
- **NVIDIA Jetson** (backend `torch`, opción TensorRT) y **Thundercomm RUBIK Pi 3**
  (backend `onnx` / onnxruntime en CPU; Hexagon pendiente del QNN SDK). Ver
  [`edge/README.rubikpi.md`](../edge/README.rubikpi.md).
- Captura **RTSP + USB plug-and-play**, ring buffer, inferencia on-device, Event
  Engine (umbral + histéresis + confirmación temporal), recorte de clip `.mp4`,
  SQLite + outbox *offline-first*, MQTT/HTTPS, OTA.
- **Dashboard web** servido por el propio nodo (`:8090`, solo stdlib, PWA): auth por
  sesión (admin/usuario, PBKDF2), **gestión de usuarios**, gestión de cámaras +
  umbral/rotación por cámara, **GPS NEO-6M con mapa de México offline**, **panel de
  selección de modelo** y alertas (SMS/correo/WhatsApp/Telegram). Ver
  [`edge/README.md`](../edge/README.md).

**⏳ Pendiente:** cloud (backend FastAPI, ingest MQTT→Postgres, object storage de
clips, frontend React), MLflow/OTA de modelos, y empaquetado del modelo en
`models/translownet/`.

## 1. Contexto

Plataforma de **detección de anomalías en vídeo (VAD)** para transporte público.
El núcleo de investigación ya existe — **TransLowNet** (`D:\Codes\TransLowNet_Jetson\Propuesta2`) — y se
refinará ~2 meses. Este documento define **toda la plataforma alrededor del modelo**: un *edge* en **Jetson**
que controla **4 cámaras** y ejecuta el modelo localmente, y un *cloud* que recibe **metadatos + clips de
anomalía**, con **backend (FastAPI)**, **frontend (React)** y **MLOps**.

### Anclaje al modelo real (verificado en el código)
- Pipeline 3 etapas: **X3D** (PyTorchVideo, sin último bloque) → features **192-d** → **`Detector_VAD`**
  (MIL, score sigmoide [0,1]) → **`violenceOneCrop`** (softmax 9/13 clases).
- Entrada: clips de `num_frames`×`sampling_rate` (x3d_l: 16 frames, sampling 5, crop 320; mean/std `[0.45]/[0.225]`).
- Existen modo *stream* (`cv2.VideoCapture`) y *offline*, con medición de FPS/latencia. Pesos en `weightsEl_Salvador/`.
- **Contrato de salida por clip** que consume la plataforma: `{score: float, class_id: int, class_probs: float[], t_start, t_end, latency_ms}`.

## 2. Objetivos y requisitos

**Funcionales**
1. Edge (Jetson): ingesta de 4 cámaras RTSP → buffering de clips → inferencia TransLowNet → score+clase por clip.
2. Detección de evento: `score > umbral` con histéresis/ventana → genera **evento de anomalía**.
3. Edge→Cloud: **telemetría/metadatos** (scores, eventos, salud) + subida de **solo el clip de anomalía**.
4. Cloud: persistir eventos, almacenar clips, API y **dashboard** (estado de cámaras, timeline de scores,
   lista de eventos, reproducción de clip, gestión de umbrales).
5. MLOps: versionado de modelos/pesos y despliegue OTA al edge.

**Principio rector — inferencia 100% en el edge**
- **Todo el detector corre en la Jetson, sin excepción** (X3D → `Detector_VAD` → clasificador).
  El cloud **NUNCA ejecuta el modelo**; solo ingiere metadatos/eventos y almacena clips.
- Presupuesto de cómputo **~2 GFLOPs** → viabilidad en edge garantizada. **TensorRT/ONNX es optimización
  opcional** (FPS/energía), no requisito; fallback PyTorch siempre válido.

**No funcionales**
- *Offline-first*: si cae la red, cola local y reintento (store-and-forward).
- La inferencia no debe tener overhead bloqueante añadido por la plataforma.
- Seguridad: TLS; auth del edge por token/cert; auth de usuarios en el dashboard.
- Reproducibilidad (tesis): todo dockerizado, configs versionadas, experimentos trazables.

## 3. Arquitectura de alto nivel

```
┌─────────────────────────── EDGE (Jetson) — INFERENCIA 100% AQUÍ ───────────────┐
│  4× RTSP cams → Capture (GStreamer/DeepStream) → Ring buffer de clips           │
│       → Inference Worker (TransLowNet: X3D→Detector_VAD→Classifier)             │
│       → Event Engine (umbral+histéresis) → Clip Encoder (recorte H.264)         │
│       → Local Store (SQLite + disco) + Outbox (store-and-forward)               │
│       → Edge Agent: MQTT (telemetría/eventos)  +  HTTPS (subida de clips)       │
│       ← OTA: descarga de pesos/config nuevos                                    │
└────────────────────────────────────────────────────────────────────────────────┘
            │  MQTT (TLS) metadatos/eventos        │  HTTPS subida de clips
            ▼                                      ▼
┌─────────────────────────── CLOUD (SIN inferencia) ──────────────────────────────┐
│  MQTT Broker (Mosquitto) → Ingest Service → PostgreSQL (eventos/telemetría)     │
│  Object Storage (MinIO/S3) clips ── Backend FastAPI (REST + WebSocket) ──┐      │
│  Model Registry (MLflow) + OTA endpoint                                  │      │
│  Frontend React (dashboard) ◀── WebSocket (tiempo real) / REST ─────────┘      │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Comunicación (metadatos + clips de anomalía)**
- **MQTT (TLS)**: telemetría continua (scores por clip, heartbeat/salud) y eventos.
- **HTTPS multipart / presigned URL**: subida del clip de la anomalía a object storage (no por MQTT).
- **WebSocket** backend→frontend: empuje de eventos en vivo al dashboard.

## 4. Componentes

### 4.1 Edge (`edge/`) — Python (+ DeepStream/GStreamer opcional)
- **Capture**: 1 hilo/proceso por cámara (RTSP). Reusa el muestreo de `inferenceFinal_Offline.py`
  (`indexClipSampling`, `centralCrop`, normalización) extraído a `edge/inference/preprocess.py`.
- **Inference Worker (íntegro en el edge)**: carga X3D + `Detector_VAD` + `violenceOneCrop`.
  ~2 GFLOPs → cabe sin problema. **Backend seleccionable**: `torch` (Jetson, opción
  TensorRT) u **`onnx`** (RUBIK Pi 3, onnxruntime CPU/Hexagon). Un worker multiplexa las cámaras.
- **Captura RTSP + USB**: cámaras RTSP configurables desde la web y **autodetección USB**
  *plug-and-play* (sin reiniciar). Rotación y reescalado en captura; cadencia por cámara
  (`capture.clip_cooldown_s`) para no saturar la cola de inferencia en CPU.
- **Event Engine**: umbral configurable (global y por cámara) + histéresis + confirmación
  temporal (persistencia / ráfaga) → abre/cierra evento.
- **Clip Encoder**: recorta el buffer a clip `.mp4` con pre/post-roll al disparar evento.
- **Local Store + Outbox**: SQLite + disco + cola de subida con reintentos (offline-first).
- **Edge Agent**: cliente MQTT + uploader HTTPS + cliente OTA (poll de versión de pesos).
- **Monitor / Dashboard web** (`edge/monitor/`): servidor stdlib (`:8090`, PWA) con auth
  por sesión y roles, gestión de **usuarios**, cámaras, umbrales/rotación, **panel de
  modelo**, alertas, y **GPS NEO-6M + mapa de México offline** (GeoJSON embebido).
- **Config**: `config.yaml` por dispositivo (id, cámaras, umbrales, endpoints, modelo,
  backend, gps) + `data/settings.json` editable desde la web (vehículo, cámaras, alertas,
  usuarios, selección de modelo).

### 4.2 Backend (`cloud/backend/`) — FastAPI
- **Estructura**: `app/{api,core,models,schemas,services,workers}`.
- **Ingest Service**: suscriptor MQTT → valida → escribe en PostgreSQL → publica a WebSocket.
- **API REST**: dispositivos, cámaras, eventos (listar/filtrar/detalle), clips (presigned URL), umbrales/config, modelos.
- **WebSocket**: `/ws/events`, `/ws/telemetry/{device}`.
- **OTA**: endpoint que sirve la versión/artefacto de pesos activo por dispositivo (lee del Model Registry).
- **Auth**: JWT (usuarios, OAuth2 password flow); token/mTLS por dispositivo (edge).

### 4.3 Datos
- **PostgreSQL** (+ TimescaleDB opcional para serie temporal de scores): `devices`, `cameras`, `telemetry`,
  `events` (inicio/fin, score máx, clase, clip_uri, estado), `models`, `users`.
- **Object Storage** (MinIO/S3): clips en `device/camera/date/event_id.mp4`.
- **Model Registry** (MLflow): versiona pesos `Detector_VAD`/clasificador + métricas de cada iteración.

### 4.4 Frontend (`cloud/frontend/`) — React + TS
- **Stack**: Vite + React + TS, TanStack Query, cliente WebSocket, Tailwind/shadcn, Recharts.
- **Vistas**: Overview (estado de 4 cámaras/Jetson) · Live (timeline de scores en vivo) · Eventos (lista
  filtrable + reproductor) · Config (umbrales/cámaras) · Modelos (versiones + despliegue OTA).

### 4.5 Infraestructura (`infra/`)
- **Edge**: `docker-compose` / contenedor con base `nvcr.io/nvidia/l4t-pytorch`.
- **Cloud**: `docker-compose` (Mosquitto, Postgres, MinIO, backend, frontend, MLflow); Helm como opción.
- **CI**: lint+tests por paquete; `.env.example` y configs versionadas.

## 5. Estructura del monorepo (fase siguiente)

```
D:\Codes\phdFinal\
├── README.md
├── docs/                  # este documento, ADRs, diagramas, capítulo de tesis
├── edge/                  # capture/ inference/ events/ storage/ agent/ · config.example.yaml · Dockerfile.jetson
├── cloud/
│   ├── backend/           # FastAPI: app/{api,core,models,schemas,services,workers}
│   └── frontend/          # React + Vite + TS
├── models/translownet/    # wrapper que referencia el modelo de Propuesta2 (pip -e) + export ONNX/TensorRT
├── shared/                # contratos: esquema MQTT + modelo de evento (OpenAPI/JSON Schema)
├── infra/                 # docker-compose edge/cloud, CI, scripts
└── data/                  # gitignored: clips, sqlite, datasets
```

**Integración del modelo**: `models/translownet/` **referencia, no duplica** `Propuesta2`
(`models_Inference.py`, `transforms.py`, pesos). Recomendado **paquete `pip -e`** para que el modelo siga
evolucionando 2 meses sin romper la plataforma.

## 6. Roadmap (~8 semanas)

1. **Sem 1–2 — Cimientos**: monorepo, contratos en `shared/`, docker-compose cloud (broker+Postgres+MinIO), wrapper del modelo.
2. ✅ **Sem 3–4 — Edge MVP**: captura RTSP/USB → inferencia → Event Engine → clip → MQTT+upload (offline-first). **Hecho.**
3. **Sem 4–5 — Backend+datos**: ingest MQTT→Postgres, REST de eventos, presigned URLs, WebSocket.
4. **Sem 5–6 — Frontend**: Overview, Live timeline, Eventos + reproductor.
5. ✅ **Sem 6–7 — Multi-cámara + backend ONNX** (RUBIK Pi 3) + dashboard web (auth, mapa offline, panel de modelo). **Hecho.** (TensorRT en Jetson: opcional, pendiente de medir.)
6. **Sem 7–8 — MLOps/OTA + endurecimiento**: MLflow, OTA de pesos, auth, TLS, docs de tesis.

## 7. Verificación (end-to-end, fase de build)

- **Modelo**: cargar X3D+`Detector_VAD`+clasificador y producir `{score,class_id,...}` sobre `FrameStream/`.
- **Edge offline-first**: con cloud apagado, generar evento → queda en outbox y se sube al reconectar.
- **Comunicación**: publicar telemetría/evento por MQTT → verificar fila en Postgres y push por WebSocket.
- **Clip**: disparar anomalía → `.mp4` en MinIO y reproducción en dashboard vía presigned URL.
- **E2E**: docker-compose cloud + edge simulado (modo offline con `FrameStream`) → evento llega al dashboard.
- **Jetson**: medir FPS/latencia TensorRT vs PyTorch (reusar `measure_inference_time`).

## 8. Decisiones abiertas
- Modelo de Jetson (Orin Nano/AGX): solo afecta margen FPS/energía; con ~2 GFLOPs corre en cualquier caso.
- Broker MQTT: **Mosquitto** (recomendado, simple) vs EMQX (dashboard/escala).
- Integración del modelo: **paquete `pip -e`** (recomendado) vs submodule.
