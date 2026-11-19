"""
Prepare tableware models for the web viewer, inside Blender (4.1 or newer).

For each piece it:
  1. joins all its parts into one object and applies rotation/scale
  2. converts units to meters (auto-detects mm / cm, or set UNITS)
  3. merges duplicate vertices and points all normals outward
  4. moves the origin to the center of the base, sitting on the floor
  5. reduces triangles to TARGET_TRIS if it has more
  6. smooths shading, keeping corners sharper than SMOOTH_ANGLE crisp
  7. leaves a single material named "glaze"
  8. exports a .glb ready for Three.js

Two ways to run it:

A) Inside Blender, on what you have open
   Save your .blend first. Select the piece (all its parts), open the Scripting tab,
   open this file and press Run Script. The .glb goes to a "glb" folder next to the
   .blend file (or your Desktop if the .blend isn't saved).

B) A whole folder of files, from the terminal (Mac):
   /Applications/Blender.app/Contents/MacOS/Blender -b -P prepare_model.py -- \
       --in ~/Desktop/modelos --out ~/Desktop/glb --units mm
   Reads .obj .fbx .stl .glb .gltf files; each file is one piece.

Options for B (all optional): --units auto|mm|cm|m   --tris 20000   --angle 30
"""
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

# Settings for running inside Blender (A). Option B can override them from the terminal.
UNITS = "auto"          # "auto", "mm", "cm" or "m": the units the model was made in
TARGET_TRIS = 20000     # reduce to about this many triangles; pieces below it are left alone
SMOOTH_ANGLE = 30       # degrees: edges sharper than this stay crisp (rims, handle corners)
MERGE_DISTANCE = 0.000001  # meters (0.001 mm): only true duplicates are merged
BASE_BAND = 0.002       # meters: vertices this close to the floor count as the base

IMPORTERS = {
    ".obj": lambda p: bpy.ops.wm.obj_import(filepath=p),
    ".stl": lambda p: bpy.ops.wm.stl_import(filepath=p),
    ".fbx": lambda p: bpy.ops.import_scene.fbx(filepath=p),
    ".glb": lambda p: bpy.ops.import_scene.gltf(filepath=p),
    ".gltf": lambda p: bpy.ops.import_scene.gltf(filepath=p),
}


def triangle_count(mesh):
    mesh.calc_loop_triangles()
    return len(mesh.loop_triangles)


def unit_scale(size, units):
    """Scale factor to meters. Tableware is roughly 4-50 cm, which keeps mm and cm apart."""
    fixed = {"mm": 0.001, "cm": 0.01, "m": 1.0}
    if units in fixed:
        return fixed[units], units
    if size >= 60:
        return 0.001, "mm (auto)"
    if size >= 1.5:
        return 0.01, "cm (auto)"
    return 1.0, "m (auto)"


def join_and_bake(objects, name):
    """One new object with every part's rotation, scale and position baked into the mesh."""
    bm = bmesh.new()
    material = next((m for o in objects for m in o.data.materials if m), None)
    for obj in objects:
        temp = obj.data.copy()
        temp.transform(obj.matrix_world)
        bm.from_mesh(temp)  # appends to what's already in bm
        bpy.data.meshes.remove(temp)
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for obj in objects:
        bpy.data.objects.remove(obj)
    piece = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(piece)
    return piece, material


def vertex_coords(mesh):
    return [v.co.copy() for v in mesh.vertices]


def clean(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=MERGE_DISTANCE)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    merged = before - len(bm.verts)
    bm.to_mesh(mesh)
    bm.free()
    return merged


def center_on_base(mesh):
    co = vertex_coords(mesh)
    min_z = min(v.z for v in co)
    base = [v for v in co if v.z <= min_z + BASE_BAND] or co
    cx = (min(v.x for v in base) + max(v.x for v in base)) / 2
    cy = (min(v.y for v in base) + max(v.y for v in base)) / 2
    mesh.transform(Matrix.Translation(Vector((-cx, -cy, -min_z))))


def apply_modifier(obj, modifier):
    """Bake one modifier into the mesh without relying on the UI."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    baked = bpy.data.meshes.new_from_object(obj.evaluated_get(depsgraph))
    obj.modifiers.remove(modifier)
    old = obj.data
    obj.data = baked
    bpy.data.meshes.remove(old)


def reduce_triangles(obj, target):
    tris = triangle_count(obj.data)
    if tris <= target:
        return tris
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    mod.decimate_type = "COLLAPSE"
    mod.ratio = target / tris
    apply_modifier(obj, mod)
    return triangle_count(obj.data)


def select_only(obj):
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def smooth(obj, angle_deg):
    angle = math.radians(angle_deg)
    mesh = obj.data
    if hasattr(mesh, "set_sharp_from_angle") and hasattr(mesh, "shade_smooth"):
        mesh.shade_smooth()
        mesh.set_sharp_from_angle(angle=angle)
    else:
        select_only(obj)
        bpy.ops.object.shade_smooth_by_angle(angle=angle)
    wn = obj.modifiers.new("WeightedNormal", "WEIGHTED_NORMAL")
    wn.keep_sharp = True


def single_glaze(mesh, material):
    existing = bpy.data.materials.get("glaze")
    if material is None:
        material = existing or bpy.data.materials.new("glaze")
    if existing and existing is not material:
        existing.name = "glaze-old"
    material.name = "glaze"
    mesh.materials.clear()
    mesh.materials.append(material)
    mesh.polygons.foreach_set("material_index", [0] * len(mesh.polygons))


def export_glb(obj, folder, name):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name + ".glb")
    select_only(obj)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_apply=True,   # bakes the smoothing
        export_yup=True,     # Three.js is Y-up
        export_cameras=False,
        export_lights=False,
    )
    return path


def prepare(objects, name, out_dir, units=UNITS, target=TARGET_TRIS, angle=SMOOTH_ANGLE):
    parts = [o for o in objects if o.type == "MESH"]
    if not parts:
        print(f"  {name}: no mesh found, skipped")
        return None
    obj, material = join_and_bake(parts, name)
    mesh = obj.data

    co = vertex_coords(mesh)
    size = max(max(v[i] for v in co) - min(v[i] for v in co) for i in range(3))
    scale, unit_label = unit_scale(size, units)
    mesh.transform(Matrix.Scale(scale, 4))

    tris_before = triangle_count(mesh)
    merged = clean(mesh)
    center_on_base(mesh)
    tris_after = reduce_triangles(obj, target)
    smooth(obj, angle)
    single_glaze(obj.data, material)
    path = export_glb(obj, out_dir, name)

    bpy.context.view_layer.update()
    dims = obj.dimensions
    print(f"  {name}: units {unit_label}, {dims.x*100:.1f} x {dims.y*100:.1f} x {dims.z*100:.1f} cm "
          f"(width x depth x height), {tris_before:,} -> {tris_after:,} triangles, "
          f"{merged} duplicate vertices merged, {os.path.getsize(path)/1024:.0f} KB -> {path}")
    if size * scale > 0.6 or size * scale < 0.02:
        print(f"  {name}: WARNING size looks wrong for tableware; set the units explicitly and run again")
    return obj


def slug(text):
    return "".join(c if c.isalnum() else "-" for c in text.lower()).strip("-") or "pieza"


def run_in_blender():
    selected = list(bpy.context.selected_objects)
    if not selected:
        print("Select the piece first (all its parts), then run the script.")
        return
    active = bpy.context.view_layer.objects.active or selected[0]
    folder = bpy.path.abspath("//glb") if bpy.data.filepath else os.path.expanduser("~/Desktop/glb")
    print("Preparing selection:")
    obj = prepare(selected, slug(active.name), folder)
    if obj:
        select_only(obj)


def run_batch(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="prepare_model.py")
    ap.add_argument("--in", dest="src", required=True, help="folder with model files")
    ap.add_argument("--out", dest="out", required=True, help="folder for the .glb files")
    ap.add_argument("--units", default=UNITS, choices=["auto", "mm", "cm", "m"])
    ap.add_argument("--tris", type=int, default=TARGET_TRIS)
    ap.add_argument("--angle", type=float, default=SMOOTH_ANGLE)
    args = ap.parse_args(argv)
    src = os.path.expanduser(args.src)
    out = os.path.expanduser(args.out)
    files = sorted(f for f in os.listdir(src) if os.path.splitext(f)[1].lower() in IMPORTERS)
    if not files:
        print(f"No .obj .fbx .stl .glb or .gltf files in {src}")
        return
    print(f"Preparing {len(files)} file(s) from {src}:")
    for f in files:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        IMPORTERS[os.path.splitext(f)[1].lower()](os.path.join(src, f))
        prepare(list(bpy.context.scene.objects), slug(os.path.splitext(f)[0]), out,
                args.units, args.tris, args.angle)


if "--" in sys.argv:
    run_batch(sys.argv[sys.argv.index("--") + 1:])
else:
    run_in_blender()
