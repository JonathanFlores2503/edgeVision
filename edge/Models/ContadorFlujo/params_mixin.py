# -*- coding: utf-8 -*-
"""
params_mixin.py
===============
Protocolo OPCIONAL de parametros ajustables en caliente desde el panel web
(web_server_zones.py). Un detector que quiera exponer sus parametros hereda
de ParamsMixin y declara PARAMS:

    class MiDetector(ParamsMixin, BaseDetector):
        PARAMS = [
            dict(key="conf", label="Confianza YOLO", type="float",
                 min=0.05, max=0.95, step=0.01,
                 help="Umbral minimo de deteccion"),
            dict(key="modo", label="Modo", type="choice",
                 choices=["a", "b"]),
            dict(key="invert", label="Invertir sentido", type="bool"),
        ]

Cada entrada:
  key      : nombre del parametro (y del atributo en self, salvo 'attr')
  attr     : (opcional) nombre del atributo si difiere de key
  label    : etiqueta que ve el usuario
  type     : "float" | "int" | "bool" | "choice"
  min/max  : clamp para float/int (opcional)
  step     : paso sugerido para la UI (opcional)
  choices  : lista de strings (solo type="choice")
  help     : descripcion corta (opcional)

El servidor llama:
  get_params()      → lista de specs + "value" actual (para pintar el form)
  set_params(dict)  → aplica con cast/clamp; retorna {"applied","errors"}
                      y dispara on_params_changed(set_de_keys) SOLO con las
                      keys cuyo valor realmente cambio (el form del panel
                      manda todos los campos en cada guardado).

Los valores guardados se persisten en params_web/<source_key>.json y se
re-aplican al agregar la fuente (tras configure(), antes de setup()).
"""

from typing import Dict, List


class ParamsMixin:
    PARAMS: List[dict] = []

    # ------------------------------------------------------------------ #
    def get_params(self) -> List[dict]:
        out = []
        for p in self.PARAMS:
            q = {k: v for k, v in p.items() if k != "attr"}
            q["value"] = getattr(self, p.get("attr", p["key"]), None)
            out.append(q)
        return out

    # ------------------------------------------------------------------ #
    def set_params(self, data: Dict) -> Dict:
        applied, errors = {}, {}
        changed = set()
        for p in self.PARAMS:
            key = p["key"]
            if key not in data:
                continue
            raw = data[key]
            typ = p.get("type", "float")
            try:
                if typ == "int":
                    val = int(float(raw))
                elif typ == "float":
                    val = float(raw)
                elif typ == "bool":
                    val = raw if isinstance(raw, bool) else \
                        str(raw).strip().lower() in ("1", "true", "on", "si", "sí")
                elif typ == "choice":
                    val = str(raw)
                    if val not in p.get("choices", []):
                        raise ValueError(f"opciones: {p.get('choices')}")
                else:
                    raise ValueError(f"tipo desconocido: {typ}")
            except (TypeError, ValueError) as e:
                errors[key] = f"valor invalido ({e})"
                continue

            if typ in ("int", "float"):
                if "min" in p and val < p["min"]:
                    val = p["min"]
                if "max" in p and val > p["max"]:
                    val = p["max"]

            attr = p.get("attr", key)
            old = getattr(self, attr, None)
            setattr(self, attr, val)
            applied[key] = val
            if old != val:          # solo lo que REALMENTE cambio de valor
                changed.add(key)    # (el form manda todos los campos juntos)

        if changed:
            try:
                self.on_params_changed(changed)
            except Exception as e:
                errors["_hook"] = str(e)

        return {"applied": applied, "errors": errors}

    # ------------------------------------------------------------------ #
    def on_params_changed(self, changed: set):
        """Hook opcional: recalcular estado derivado de los parametros."""
        pass
