# Contratos de mensajería Edge ↔ Cloud

Fuente única de verdad para los mensajes que intercambian el edge (Jetson) y el cloud.
El edge **produce**; el backend **consume**. Versionado con `schema_version`.

## Topics MQTT

```
vad/<device_id>/telemetry   # score por clip (continuo, throttled)
vad/<device_id>/events      # apertura/cierre de evento de anomalía
vad/<device_id>/health      # heartbeat / salud del nodo
```

## 1. Telemetry  (`vad/<device>/telemetry`)
Resultado de inferencia por clip y cámara.
```json
{
  "schema_version": 1,
  "device_id": "jetson-001",
  "camera_id": "cam-front",
  "ts": "2026-06-12T14:03:21.512Z",
  "t_start": "2026-06-12T14:03:18.000Z",
  "t_end": "2026-06-12T14:03:21.200Z",
  "score": 0.73,
  "class_id": 3,
  "class_probs": [0.02, 0.05, 0.10, 0.73, 0.10],
  "latency_ms": 41.8
}
```

## 2. Event  (`vad/<device>/events`)
```json
{
  "schema_version": 1,
  "event_id": "jetson-001:cam-front:1718200998",
  "device_id": "jetson-001",
  "camera_id": "cam-front",
  "state": "open",                 // "open" | "closed"
  "t_start": "2026-06-12T14:03:18.000Z",
  "t_end": null,                    // null si state=open
  "max_score": 0.81,
  "class_id": 3,
  "class_name": "Fighting",
  "clip_uri": null,                 // se rellena al subir el clip (cloud)
  "clip_object_key": "jetson-001/cam-front/2026-06-12/<event_id>.mp4"
}
```

## 3. Health  (`vad/<device>/health`)
```json
{
  "schema_version": 1,
  "device_id": "jetson-001",
  "ts": "2026-06-12T14:03:21.512Z",
  "uptime_s": 38211,
  "cameras": {"cam-front": "online", "cam-rear": "online"},
  "model": "x3d_l",
  "fps": {"cam-front": 7.9},
  "outbox_pending": 0,
  "cpu_pct": 44.0,
  "gpu_pct": 61.0,
  "temp_c": 58.0
}
```

## 4. Subida de clip (HTTPS, no MQTT)
El edge pide una *presigned URL* y sube el `.mp4`:
```
POST {http.base_url}/devices/{device_id}/clips:presign
  body: { "event_id", "object_key", "content_type": "video/mp4", "size_bytes" }
  resp: { "upload_url", "method": "PUT", "headers": {...} }
PUT <upload_url>  (binario del .mp4)
POST {http.base_url}/events/{event_id}/clip-uploaded   { "object_key" }
```
