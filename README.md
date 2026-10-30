# 3D Test

Visor 3D en Three.js para vajilla modular: orbitar las piezas, cambiar el color y ver el bowl apilado sobre el plato.

## Estructura

- `index.html` – visor (Three.js r147 desde CDN, sin build)
- `models/` – modelos GLB en metros, con el pivote en el centro de la base
- `scripts/generate.py` – genera los GLB a partir de perfiles 2D y calcula los offsets de apilado

## Modelos

| Pieza | Medidas | Triángulos |
|---|---|---|
| Plato chico | 17.5 × 2 cm | ~3,500 |
| Bowl mediano | 15 × 7 cm | ~4,200 |

Cada GLB guarda en `extras` sus medidas y `stackOffsets` (altura a la que se apoya sobre otra pieza).

Para cambiar formas o agregar piezas, edita los perfiles en `scripts/generate.py` y ejecuta:

```bash
pip install numpy
python3 scripts/generate.py
```

## Local

```bash
npx serve .
```

## Deploy

Proyecto estático: en Vercel no necesita comando de build ni directorio de salida.
