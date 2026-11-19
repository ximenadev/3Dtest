# 3D Test

Visor 3D en Three.js para vajilla modular: orbitar las piezas, cambiar el color y ver el bowl apilado sobre el plato.

## Estructura

- `index.html` – visor (Three.js r147 desde CDN, sin build)
- `models/` – modelos GLB en metros, con el pivote en el centro de la base
- `scripts/generate.py` – genera los GLB a partir de perfiles 2D y calcula los offsets de apilado
- `scripts/rebuild_round.py` – reconstruye una pieza redonda exportada de Blender como sólido de revolución limpio

## Modelos

| Pieza | Medidas | Triángulos |
|---|---|---|
| Plato chico | 17.5 × 2 cm | ~3,500 |
| Bowl mediano | 15 × 7 cm | ~4,200 |
| Vaso | 8.3 × 9.6 cm | ~8,300 |

Cada GLB guarda en `extras` sus medidas y `stackOffsets` (altura a la que se apoya sobre otra pieza).

Para cambiar formas o agregar piezas, edita los perfiles en `scripts/generate.py` y ejecuta:

```bash
pip install numpy
python3 scripts/generate.py
```

## Piezas desde Blender

Para piezas redondas (platos, bowls, vasos sin asa), después de exportar el GLB:

```bash
python3 scripts/rebuild_round.py vaso-blender.glb models/vaso.glb
```

El script corta la pieza por su eje para obtener el perfil exacto y lo gira 360° con anillos uniformes. Mantiene la silueta original, centra la base en el origen, conserva los bordes marcados (labio, pie) y elimina las franjas de luz que dejan los triángulos largos de los archivos CAD. No sirve para piezas con asa o pico.

## Local

```bash
npx serve .
```

## Deploy

Proyecto estático: en Vercel no necesita comando de build ni directorio de salida.
