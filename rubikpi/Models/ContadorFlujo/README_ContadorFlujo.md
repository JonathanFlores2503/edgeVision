# ContadorFlujo — Conteo de personas ENTRADA/SALIDA

**Repositorio:** https://github.com/Celestial-Dynamics-AI/ArconteDetection_DebugTools/tree/main/Models/areaRest/ContadorFlujo

Dos detectores standalone montados sobre el framework `../../Base/`.
Cuentan **cuánta gente cruza** (flujo), no cuánta hay en el frame
(para ocupación/aforo ver `../ContadorAforo/`).

| Archivo | NAME | Método |
|---|---|---|
| `PeopleCounter_Web.py` | `CONTADOR_PERSONAS` | V1 — línea de cruce simple |
| `PeopleCounter_V2_Web.py` | `CONTADOR_V2` | V2 — corredor con escalera de líneas |

---

## V1 — `PeopleCounter_Web.py` (línea de cruce)

### Qué hace

Cuenta personas que cruzan **una línea** dibujada sobre la puerta,
bidireccional: un sentido suma (ENTRADA) y el otro resta (SALIDA).

### Cómo funciona

1. En el editor web (botón "Zonas") dibujas **una línea de 2 puntos**
   atravesando la puerta.
2. Cada persona se detecta con YOLO y se trackea con ByteTrack
   (tracker afinado en `bytetrack_contador.yaml`); su posición es el
   **centroide** del box.
3. Cuando el segmento *(centroide anterior → centroide actual)* cruza la
   línea, se cuenta según el lado hacia el que cruzó:
   - un lado = **ENTRADA** (sube)
   - el otro = **SALIDA** (baja)
   - `--invert` cambia el sentido sin redibujar.
4. `DENTRO = inicial + entradas − salidas`. El punto de partida se fija en
   la tarjeta ("Hay N personas dentro ahora → Fijar y contar") y todo se
   persiste en `counters/<fuente>.json` (sobrevive reinicios).

**Optimización:** YOLO NO ve el frame completo, solo un recorte generoso
alrededor de la línea (`--full-frame` lo desactiva).

### Limitación conocida

Una sola línea falla con oclusiones justo sobre la línea o cuando el
tracker pierde el id en el instante exacto del cruce → para eso está la V2.

---

## V2 — `PeopleCounter_V2_Web.py` (corredor con escalera)

### Qué hace

Refina la V1 con el método del **box + escalera de líneas**
(diseño en `DISENO_Contador.html`). No depende del
instante exacto del cruce.

### Cómo funciona

1. Dibujas **un box de 4 puntos** atravesando la puerta. El PRIMER lado
   que dibujas (clicks 1-2) = AFUERA (`--invert` lo voltea).
2. Dentro del box se genera una **escalera de N líneas internas**.
3. Cada track tiene un **progreso p (0–100 %)** a lo largo del eje
   AFUERA→ADENTRO. Las líneas son umbrales: brincarse una entre frames
   cuenta igual.

### Reglas de decisión (R1–R5)

| Regla | Situación | Resultado |
|---|---|---|
| R1 | Cruzó 50 %+1 de las líneas con dirección neta | Cuenta (mayoría) |
| R2 | Salió del box por el lado contrario al que entró sin haber contado | Cuenta por lado de salida |
| R3 | Track perdido dentro del box | Con mayoría → cuenta (inferido); sin mayoría → PENDIENTE (fantasma) |
| R4 | Id nuevo aparece dentro del box | Hereda el progreso de un fantasma compatible (cruce partido en 2 ids por oclusión) |
| R5 | Se asoma y regresa por donde vino | NO cuenta |

### Guardián de luz

Monitorea el brillo promedio del corredor (0–255, suavizado):

| Nivel | Estado | Acción |
|---|---|---|
| > LUZ_BAJA | NORMAL | — |
| LUZ_OSCURO…LUZ_BAJA | BAJA LUZ | Confianza reducida + CLAHE |
| < LUZ_OSCURO | OSCURO | Alerta "sin luz suficiente" + conteo EN PAUSA |

### Heredado de la V1

Contador persistente + "Fijar N dentro", evidencia por evento
(foto + crop + rostro con nombre + jsonl en `EVENTOS_CONTADOR/`),
nube de rostro (mediapipe, pesos en `weights_face/`), recorte de
análisis y ByteTrack afinado.

---

## Flags propias (además de las estándar de `Base/`)

| Flag | Default | Descripción |
|---|---|---|
| `--model` | `../yolo11x.pt` | Pesos YOLO (usa `--model yolo11x.pt` para autodescarga) |
| `--invert` | off | Voltea el sentido ENTRADA/SALIDA |
| `--full-frame` | off | YOLO ve el frame completo (sin recorte alrededor de la línea/box) |
| `--zone-expand` | 0 | Px extra de expansión del recorte de análisis |

## Ejecutar

```bash
# V1 — línea simple
uv run --no-project --with flask,opencv-python,ultralytics,lap,mediapipe \
    PeopleCounter_Web.py --port 8032 --model yolo11x.pt

# V2 — corredor
uv run --no-project --with flask,opencv-python,ultralytics,lap,mediapipe \
    PeopleCounter_V2_Web.py --port 8033 --model yolo11x.pt
```

Abrir `http://IP_SERVIDOR:PUERTO/`, agregar fuente (video o RTSP),
botón "Zonas" para dibujar la línea (V1) o el box (V2).

## Archivos de runtime (no se versionan)

`counters/` (estado persistente del conteo), `zones_web/` (líneas/boxes
por fuente), `params_web/` (parámetros ajustados desde la web),
`EVENTOS_CONTADOR/` (evidencia), `ALERTAS_*/`, `uploads/`, `sources.json`.
