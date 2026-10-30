"""
Generate 3D tableware models as GLB files.

Each piece is a lathe (revolved) solid built from a 2D profile in meters:
real-world scale, pivot at the center of the base, one PBR material,
smooth normals and UVs. Stacking offsets are computed from the profiles
and stored in each file's extras so the configurator can read them.

Usage: python3 generate.py
"""
import json
import os
import struct

import numpy as np

SEGMENTS = 96  # radial segments; keeps each piece well under 10k triangles


# ---------------------------------------------------------------------------
# Profiles: (radius, height) in meters, from the bottom axis, along the
# outside, over the rim, down the inside, back to the axis.
# ---------------------------------------------------------------------------

# Plato chico: 17.5 cm diameter x 2 cm
PLATE_OUTER = [
    (0.0000, 0.0020), (0.0570, 0.0020), (0.0590, 0.0004), (0.0605, 0.0000),
    (0.0640, 0.0000), (0.0660, 0.0008), (0.0680, 0.0025), (0.0760, 0.0060),
    (0.0820, 0.0110), (0.0855, 0.0160), (0.0872, 0.0190),
]
PLATE_RIM = [(0.0870, 0.0198), (0.0862, 0.0200), (0.0852, 0.0195)]
PLATE_INNER = [
    (0.0830, 0.0160), (0.0790, 0.0110), (0.0730, 0.0080), (0.0650, 0.0068),
    (0.0000, 0.0068),
]

# Bowl mediano: 15 cm diameter x 7 cm (MVP estimate; replace with real size)
BOWL_OUTER = [
    (0.0000, 0.0030), (0.0300, 0.0030), (0.0320, 0.0006), (0.0335, 0.0000),
    (0.0370, 0.0000), (0.0390, 0.0010), (0.0410, 0.0040), (0.0500, 0.0130),
    (0.0590, 0.0260), (0.0660, 0.0410), (0.0715, 0.0570), (0.0745, 0.0680),
]
BOWL_RIM = [(0.0748, 0.0692), (0.0742, 0.0700), (0.0730, 0.0698)]
BOWL_INNER = [
    (0.0700, 0.0660), (0.0665, 0.0540), (0.0610, 0.0400), (0.0540, 0.0270),
    (0.0450, 0.0170), (0.0340, 0.0110), (0.0200, 0.0090), (0.0000, 0.0085),
]

PIECES = {
    "plato-chico-17cm": {
        "profile": PLATE_OUTER + PLATE_RIM + PLATE_INNER,
        "outer": PLATE_OUTER, "inner": PLATE_INNER, "rim": PLATE_RIM,
        "color": (0.02, 0.10, 0.55),  # blue (linear RGB)
        "glaze": "Azul",
    },
    "bowl-mediano-15cm": {
        "profile": BOWL_OUTER + BOWL_RIM + BOWL_INNER,
        "outer": BOWL_OUTER, "inner": BOWL_INNER, "rim": BOWL_RIM,
        "color": (0.60, 0.16, 0.06),  # Terracota-like (linear RGB)
        "glaze": "Terracota",
    },
}


# ---------------------------------------------------------------------------
# Lathe geometry
# ---------------------------------------------------------------------------

def profile_normals(profile):
    """Smooth 2D normals (dr, dy) for each profile point, pointing out of the material."""
    p = np.array(profile, dtype=np.float64)
    tangents = np.zeros_like(p)
    tangents[1:-1] = p[2:] - p[:-2]
    tangents[0] = p[1] - p[0]
    tangents[-1] = p[-1] - p[-2]
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)
    # Points on the axis face straight down (bottom) or up (inside floor).
    normals[0] = (0.0, -1.0)
    normals[-1] = (0.0, 1.0)
    return normals


def lathe(profile, segments=SEGMENTS):
    p = np.array(profile, dtype=np.float64)
    n2 = profile_normals(profile)
    lengths = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    v_coord = lengths / lengths[-1]

    positions, normals, uvs = [], [], []
    for s in range(segments + 1):  # duplicate seam column for clean UVs
        phi = 2.0 * np.pi * s / segments
        c, sn = np.cos(phi), np.sin(phi)
        for i, (r, y) in enumerate(p):
            positions.append((r * sn, y, r * c))
            nr, ny = n2[i]
            normals.append((nr * sn, ny, nr * c))
            uvs.append((s / segments, v_coord[i]))

    rows = len(p)
    indices = []
    for s in range(segments):
        for i in range(rows - 1):
            a = s * rows + i
            b = (s + 1) * rows + i
            indices += [a, a + 1, b, b, a + 1, b + 1]

    positions = np.array(positions, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    indices = np.array(indices, dtype=np.uint32).reshape(-1, 3)

    # Make triangle winding agree with the outward normals (counter-clockwise front faces).
    tri = positions[indices]
    face_n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    vert_n = normals[indices].sum(axis=1)
    flip = (face_n * vert_n).sum(axis=1) < 0
    indices[flip] = indices[flip][:, [0, 2, 1]]

    return positions, normals, np.array(uvs, dtype=np.float32), indices.astype(np.uint32)


# ---------------------------------------------------------------------------
# Stacking: how high piece A sits when placed centered on piece B.
# ---------------------------------------------------------------------------

def lower_surface(outer):
    """Height of the piece's underside as a function of radius."""
    p = np.array(outer)
    return lambda r: np.interp(r, p[:, 0], p[:, 1], left=np.nan, right=np.nan)


def upper_surface(piece):
    """Height of the piece's top surface (inside floor, walls and rim) as a function of radius."""
    p = np.array(piece["inner"] + piece["rim"])
    p = p[np.argsort(p[:, 0])]
    # Where the rim folds back, keep the highest point at each radius.
    rs = np.unique(p[:, 0])
    ys = np.array([p[p[:, 0] == r, 1].max() for r in rs])
    return lambda r: np.interp(r, rs, ys, left=np.nan, right=np.nan)


def stack_offset(top, below):
    """Vertical offset (m) for `top` resting on `below`: the highest contact point."""
    r_max = min(np.array(top["outer"])[:, 0].max(), np.array(below["inner"] + below["rim"])[:, 0].max())
    rs = np.linspace(0.0, r_max, 4000)
    gap = upper_surface(below)(rs) - lower_surface(top["outer"])(rs)
    return float(np.nanmax(gap))


# ---------------------------------------------------------------------------
# GLB writer (glTF 2.0, single mesh, single PBR material)
# ---------------------------------------------------------------------------

def pad4(b, fill=b"\x00"):
    return b + fill * ((4 - len(b) % 4) % 4)


def write_glb(path, name, positions, normals, uvs, indices, color, extras):
    blobs, views, accessors = [], [], []
    offset = 0

    def add(arr, target, comp, typ, minmax=False):
        nonlocal offset
        data = pad4(arr.tobytes())
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": arr.nbytes, "target": target})
        acc = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            acc["min"] = arr.min(axis=0).tolist()
            acc["max"] = arr.max(axis=0).tolist()
        accessors.append(acc)
        blobs.append(data)
        offset += len(data)
        return len(accessors) - 1

    pos_i = add(positions, 34962, 5126, "VEC3", minmax=True)
    nrm_i = add(normals, 34962, 5126, "VEC3")
    uv_i = add(uvs, 34962, 5126, "VEC2")
    idx_i = add(indices.reshape(-1), 34963, 5125, "SCALAR")
    buffer = b"".join(blobs)

    gltf = {
        "asset": {"version": "2.0", "generator": "lathe-glb"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name, "extras": extras}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": pos_i, "NORMAL": nrm_i, "TEXCOORD_0": uv_i},
            "indices": idx_i, "material": 0,
        }]}],
        "materials": [{
            "name": "glaze",
            "pbrMetallicRoughness": {"baseColorFactor": [*color, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.85},
        }],
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(buffer)}],
    }
    json_chunk = pad4(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    total = 12 + 8 + len(json_chunk) + 8 + len(buffer)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_chunk), 0x4E4F534A) + json_chunk)
        f.write(struct.pack("<II", len(buffer), 0x004E4942) + buffer)


def main():
    plate, bowl = PIECES["plato-chico-17cm"], PIECES["bowl-mediano-15cm"]
    stacking = {
        "plato-chico-17cm": {"on_plato-chico-17cm": stack_offset(plate, plate),
                              "on_bowl-mediano-15cm": stack_offset(plate, bowl)},
        "bowl-mediano-15cm": {"on_plato-chico-17cm": stack_offset(bowl, plate),
                               "on_bowl-mediano-15cm": stack_offset(bowl, bowl)},
    }

    report = {}
    for name, piece in PIECES.items():
        pos, nrm, uv, idx = lathe(piece["profile"])
        size = (pos.max(axis=0) - pos.min(axis=0)).tolist()
        extras = {
            "piece": name,
            "units": "meters",
            "pivot": "center of base",
            "height": round(float(pos[:, 1].max()), 4),
            "diameter": round(float(size[0]), 4),
            "stackOffsets": {k: round(v, 4) for k, v in stacking[name].items()},
            "previewGlaze": piece["glaze"],
        }
        path = os.path.join(os.path.dirname(__file__), "..", "models", f"{name}.glb")
        write_glb(path, name, pos, nrm, uv, idx, piece["color"], extras)
        report[path] = {"triangles": len(idx), "vertices": len(pos), **extras}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
