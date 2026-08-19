# CONTRATO — CONTADOR_PERSONAS / CONTADOR_V2 · v1.0 (2026-07-31)

**Repositorio (código fuente):** https://github.com/Celestial-Dynamics-AI/ArconteDetection_DebugTools/tree/main/Models/areaRest/ContadorFlujo

Contrato de integración de los contadores de **flujo** (entradas/salidas).
Define qué soy, qué consumo y qué produzco, para integrarme en otra
plataforma sin leer mi código.

---

## 1 · Qué soy

Dos detectores del mismo servicio HTTP (misma API, mismo framework):

| NAME | Archivo | Método |
|---|---|---|
| `CONTADOR_PERSONAS` | `PeopleCounter_Web.py` | V1 — línea de cruce simple |
| `CONTADOR_V2` | `PeopleCounter_V2_Web.py` | V2 — corredor con escalera de líneas (robusto a oclusiones) |

- **Función**: contar **cuánta gente cruza** una puerta/corredor y mantener
  `DENTRO = inicial + entradas − salidas`. (Para ocupación instantánea ver
  el contrato de `../ContadorAforo/`.)
- **Modelo de ejecución**: un proceso = un puerto = N fuentes; cada fuente
  necesita que un humano dibuje su línea (V1) o box (V2) una única vez en
  el editor web.
- **Persistencia**: el conteo sobrevive reinicios (`counters/<fuente>.json`).
- **Headless**: interfaz 100 % HTTP.

---

## 2 · Qué consumo

### Fuentes de video
RTSP (`rtsp://user:pass@IP:puerto/ruta`, reconexión automática), archivo de
video local, o webcam (`0`).

### Dependencias
| Qué | Detalle |
|---|---|
| Python + uv | `uv run --no-project --with flask,opencv-python,ultralytics,lap` (+`mediapipe` para nube de rostro) |
| Framework | `Models/Base/` en `../../Base` (mismo repo) |
| Pesos YOLO | default `../yolo11x.pt`; con `--model yolo11x.pt` ultralytics lo descarga solo |
| Tracker | `bytetrack_contador.yaml` (incluido en esta carpeta) |

### Configuración imprescindible por fuente (una vez, humano)
- **Dibujar la línea (V1) o el box de 4 puntos (V2)** en el editor web
  (botón Zonas) → se guarda en `zones_web/` y se recarga en caliente.
- Opcional: fijar el punto de partida (`POST /counter {"inside": N}`).

### Recursos
1 puerto TCP · GPU: 1 YOLO por fuente (YOLO ve solo un recorte alrededor de
la línea/box, no el frame completo) · disco: KB de estado + evidencia de
eventos.

---

## 3 · Qué produzco

### a) Conteos en JSON — salida principal
`GET /api/sources` → cada fuente incluye (además de `status/fps/alerts`):

```json
{
  "id": "e26d2e58", "name": "Puerta A", "type": "CONTADOR_V2",
  "status": "ONLINE",
  "inside": 12,     // ← dentro AHORA (= offset + in − out)
  "in": 40,         // ← entradas acumuladas
  "out": 28,        // ← salidas acumuladas
  "offset": 0,      // ← punto de partida fijado
  "counter": true,  // ← esta fuente soporta fijar 'inside'
  "fps": 25.1, "alerts": 0, "source": "rtsp://..."
}
```

También endpoint dedicado:
- `GET  /api/sources/<id>/counter` → `{"inside": N, "in": E, "out": S, "offset": O}`
- `POST /api/sources/<id>/counter` con `{"inside": N}` → fija el punto de
  partida (entradas/salidas arrancan de cero desde ese momento).

### b) Video anotado
`GET /api/sources/<id>/mjpeg` (stream en vivo con línea/corredor, tracks y
marcador ENTRADAS/SALIDAS/DENTRO) · `GET /api/sources/<id>/frame` (JPEG).

### c) Evidencia por evento de cruce (V2)
Por cada entrada/salida: foto del frame + crop de la persona + rostro +
registro `.jsonl` en `EVENTOS_CONTADOR/<fuente>/`.

### d) Estados especiales (V2)
Guardián de luz: con oscuridad severa el conteo se PAUSA (banner "sin luz
suficiente") para no acumular basura.

### e) Archivos locales (no versionar)
`counters/`, `zones_web/`, `params_web/`, `EVENTOS_CONTADOR/`, `uploads/`,
`sources.json`.

---

## 4 · API (idéntica base que los demás detectores del repo)

| Método y ruta | Qué hace |
|---|---|
| `GET /api/sources` | Lista fuentes **con conteos** (sección 3a) |
| `POST /api/sources` | Agregar fuente `{"source": "...", "name": "x"}` |
| `DELETE /api/sources/<id>` | Quitar fuente |
| `GET/POST /api/sources/<id>/counter` | Leer / fijar el conteo `inside` |
| `GET/POST /api/sources/<id>/zones` | Leer / definir línea o box (lo que dibuja el editor) |
| `GET/POST /api/sources/<id>/params` | Ajustes en caliente |
| `GET /api/sources/<id>/mjpeg` · `/frame` | Video anotado |
| `POST /api/upload` · `/record` · `GET /api/system` · `/api/logs` | Igual que el resto |

Sin autenticación propia: red interna o reverse-proxy.

---

## 5 · Cómo integrarme (caja negra HTTP)

```bash
# 1. Arrancar (elegir V1 o V2):
cd Models/areaRest/ContadorFlujo
uv run --no-project --with flask,opencv-python,ultralytics,lap \
    PeopleCounter_V2_Web.py --port 8092 --model yolo11x.pt

# 2. Registrar la cámara:
curl -X POST http://HOST:8092/api/sources -H "Content-Type: application/json" \
     -d '{"source": "rtsp://user:pass@IP:554/Streaming/Channels/101", "name": "Puerta A"}'

# 3. (Una vez, humano) dibujar el corredor en http://HOST:8092/ → botón Zonas
#    y opcionalmente fijar el punto de partida:
curl -X POST http://HOST:8092/api/sources/<id>/counter \
     -H "Content-Type: application/json" -d '{"inside": 12}'

# 4. Leer conteos cuando se quiera:
curl http://HOST:8092/api/sources
# → [{"name":"Puerta A", "inside":12, "in":40, "out":28, ...}]
```

---

## 6 · Garantías y limitaciones

- Conteo persistente entre reinicios; reconexión RTSP automática.
- V2 es el recomendado para producción (reglas R1–R5 contra oclusiones,
  cruces partidos en dos ids, y "se asoma y regresa" no cuenta).
- Requiere el paso manual de dibujar línea/box por cámara (una vez).
- No identifica personas únicas: si la misma persona entra 2 veces, son 2
  entradas.
- Si se cambia la resolución de la fuente, hay que redibujar la línea/box.

---

*Docs de funcionamiento: `README_ContadorFlujo.md` ·
`DOCUMENTACION_ContadorFlujo.html` · diseño del corredor: `DISENO_Contador.html`.*
