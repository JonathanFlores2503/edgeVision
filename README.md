# edgeVision

Plataforma de **detección de anomalías en vídeo (VAD) para transporte público**: nodos
*edge* que capturan cámaras RTSP/USB, ejecutan el modelo **100% on-device**, generan
eventos + clips y los mandan al cloud (MQTT para metadatos, HTTPS para clips) con cola
local *offline-first*.

Diseño completo de la plataforma: [`docs/architecture.md`](docs/architecture.md).

## Un árbol de código por dispositivo

El nodo corre en **dos placas distintas** que no comparten ni acelerador ni runtime de
inferencia. Para poder trabajar en una **sin arriesgar la otra**, cada una tiene su
**propia copia completa del código**:

| Carpeta | Placa | Inferencia | Dashboard |
|---|---|---|---|
| [`jetson/`](jetson/) | NVIDIA Jetson | `torch` sobre **CUDA** | `:8080` |
| [`rubikpi/`](rubikpi/) | Thundercomm RUBIK Pi 3 (QCS6490) | `onnxruntime` → **NPU Hexagon** / CPU | `:8090` |

**Son independientes a propósito.** No hay código compartido entre ellas: lo que edites
en `jetson/` no toca `rubikpi/` y viceversa. El precio de ese aislamiento es que **una
corrección en la fontanería común (captura, eventos, storage, dashboard) hay que
aplicarla en las dos carpetas**; si solo la aplicas en una, las placas divergen.

Lo único común es:

```
docs/      # arquitectura y guía de plugins (aplica a las dos placas)
shared/    # contratos de mensajería edge ↔ cloud (esquema MQTT y de eventos)
```

## Estado

- **`jetson/`** — listo y limpio: solo camino `torch`/CUDA. El backend ONNX y los
  artefactos de Qualcomm ya no están en este árbol.
- **`rubikpi/`** — *copia en bruto, pendiente de limpieza*: todavía arrastra piezas de
  Jetson (`Dockerfile.jetson`, `requirements-model.txt`, el `model.py` de torch). Es el
  siguiente paso.
- **`edge/`** — árbol **original** intacto, de referencia. Se borra cuando confirmes que
  las dos carpetas nuevas arrancan bien en sus placas.
- **`cloud/`** — no existe todavía (backend FastAPI + frontend React); está diseñado en
  `docs/architecture.md`.

## Arrancar un nodo

Dependencias **solo con [uv](https://docs.astral.sh/uv/)** — nada de conda ni de `pip`
suelto. Cada árbol tiene su propio `pyproject.toml` + `uv.lock`, porque las dos placas no
comparten ni intérprete ni runtime de inferencia.

```bash
cd jetson          # o: cd rubikpi
uv sync                                # entorno reproducible desde el lock
cp config.example.yaml config.yaml     # ajusta cámaras, pesos, umbrales, endpoints
```

`uv sync` deja el entorno en `.venv/` (es como uv funciona por dentro), pero no se activa
a mano: todo se lanza con `uv run`.

### ▶️ ASÍ SE CORRE EL MODELO EN PRODUCCIÓN

Con el modelo real cargado, las cámaras guardadas, eventos, clips y alertas — o sea, la
plataforma de verdad:

```bash
cd jetson          # o: cd rubikpi
uv run python -m edge.main --config config.yaml --force
```

Y abre el dashboard en **`http://<IP-del-nodo>:8080`** — el puerto es `monitor.port` de
`config.yaml` (8080 por defecto en las dos placas; en `rubikpi/` está pendiente moverlo al
8090 de la tabla de arriba). El proceso se queda en primer plano: el nodo vive mientras
corra esa terminal y se para con `Ctrl+C`.

- **Sin `--mock`** — eso es lo que lo hace producción. `--mock` sustituye el modelo por un
  scorer simulado y **descarta la selección de modelo** del panel de admin: sirve para
  probar la fontanería (dashboard, eventos, cola cloud) sin GPU y sin cámaras, no para
  operar.
- **`--force`** libera el puerto matando un `edge.main` previo (solo esos procesos) en vez
  de abortar con `Address already in use`.
- **Lánzalo desde la carpeta de la placa** (`jetson/`, `rubikpi/`): `storage.db_path` y
  `clips_dir` se resuelven contra el directorio actual, así que desde otro sitio verás una
  base de datos vacía.

Para dejarlo corriendo sin terminal abierta, súbelo como servicio systemd con ese mismo
comando en `ExecStart` y `WorkingDirectory` en la carpeta de la placa.

El detalle de cada modelo, del panel de admin y de las pruebas sin cámara está en el
README de cada placa: [`jetson/README.md`](jetson/README.md).

**La versión de Python la impone el hardware, no el gusto.** En `jetson/` está clavada en
**3.8** porque el único torch con CUDA de la placa es el wheel cp38 de JetPack; con
cualquier Python más nuevo solo hay torch de PyPI, que en aarch64 no trae CUDA y tira la
inferencia a la CPU. Por lo mismo, **torch no se declara en el `pyproject.toml`**: se
instala aparte dentro del entorno de uv (instrucciones en el propio `pyproject.toml`).

`config.yaml` y `data/` están fuera de git: **son de cada dispositivo**, no del repo.
Dale a cada placa su propio `device.id` — ese id gobierna los topics MQTT
(`vad/<device_id>/…`) y los `event_id`; si las dos usan el mismo, sus datos se pisan en
el cloud.
