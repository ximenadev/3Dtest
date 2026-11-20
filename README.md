# 3D Test

Visor 3D en Three.js para vajilla modular: orbitar las piezas y cambiar el color de cada una. Muestra las dos teteras, el vaso y la taza mocha.

## Estructura

- `index.html` – visor (Three.js r147 desde CDN, sin build)
- `models/` – modelos GLB en metros, con el pivote en el centro de la base (el plato y el bowl generados siguen aquí, aunque el visor ya no los muestra)
- `scripts/generate.py` – genera los GLB a partir de perfiles 2D y calcula los offsets de apilado
- `scripts/prepare_model.py` – script de Blender: escala, limpieza, origen en la base, reducción de triángulos, suavizado, material `glaze` y exportación GLB
- `scripts/rebuild_round.py` – reconstruye una pieza redonda exportada de Blender como sólido de revolución limpio

## Modelos

| Pieza | Medidas | Triángulos |
|---|---|---|
| Plato chico | 17.5 × 2 cm | ~3,500 |
| Bowl mediano | 15 × 7 cm | ~4,200 |
| Vaso | 8.3 × 9.6 cm | ~8,300 |
| Tetera asa superior | 15.2 × 13.5 × 22.5 cm | ~20,000 |
| Tetera asa lateral | 20.2 × 13.5 × 15 cm | ~20,000 |
| Taza mocha sin asa | 9.4 × 8.4 cm | ~6,900 |

Cada GLB guarda en `extras` sus medidas y `stackOffsets` (altura a la que se apoya sobre otra pieza).

Para cambiar formas o agregar piezas, edita los perfiles en `scripts/generate.py` y ejecuta:

```bash
pip install numpy
python3 scripts/generate.py
```

## Piezas desde Blender

Preparar una carpeta de modelos (OBJ, FBX, STL, GLB) desde la terminal:

```bash
/Applications/Blender.app/Contents/MacOS/Blender -b -P scripts/prepare_model.py -- \
  --in ~/Desktop/modelos --out ~/Desktop/glb --units mm
```

O dentro de Blender: seleccionar la pieza, abrir el script en la pestaña Scripting y ejecutarlo.

Para piezas redondas (platos, bowls, vasos sin asa), después de exportar el GLB:

```bash
python3 scripts/rebuild_round.py vaso-blender.glb models/vaso.glb
```

El script corta la pieza por su eje para obtener el perfil exacto y lo gira 360° con anillos uniformes. Mantiene la silueta original, centra la base en el origen, conserva los bordes marcados (labio, pie) y elimina las franjas de luz que dejan los triángulos largos de los archivos CAD. No sirve para piezas con asa o pico.

También repara defectos comunes de los archivos CAD: cierra huecos en la superficie (hasta 10 mm), cierra agujeros en el centro de la base o del fondo, quita tapas o caras sueltas que tapan la abertura e ignora caras sueltas. Al terminar, imprime qué reparó.

## Local

```bash
npx serve .
```

## Deploy

Proyecto estático: en Vercel no necesita comando de build ni directorio de salida.
