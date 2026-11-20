"""
Rebuild a round piece (plate, bowl, glass...) as a clean turned shape.

Takes a GLB exported from Blender, slices it with a vertical plane through its axis to get
the exact cross-section, and spins that profile 360° to build new geometry with even rings.
The silhouette matches the original; the lighting comes out smooth because every ring of
vertices sits at one height. Sharp corners (rim, foot) stay sharp.

Only for pieces that are round around the vertical axis. Handles, spouts or other
asymmetric parts would be lost; don't use it on those.

Usage: python3 scripts/rebuild_round.py input.glb output.glb [segments]
"""
import json
import struct
import sys

import numpy as np

SEGMENTS = 96            # slices around the piece
SIMPLIFY_TOL = 0.0001    # meters (0.1 mm): profile points closer than this to a straight line are dropped
SHARP_ANGLE = 35         # degrees: profile corners sharper than this get a hard edge
SLICE_ANGLES = [0.0, 0.0123, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6]  # radians tried for the slice
MAX_GAP = 0.01           # meters (10 mm): holes in the mesh up to this size get closed
AXIS_TOL = 0.0002        # meters (0.2 mm): outline points this close to the center count as on the axis


# --------------------------------------------------------------------------- reading

def read_mesh(path):
    b = open(path, "rb").read()
    if struct.unpack("<I", b[:4])[0] != 0x46546C67:
        sys.exit(f"{path} is not a GLB file")
    jl = struct.unpack("<I", b[12:16])[0]
    g = json.loads(b[20:20 + jl])
    bo = 20 + jl + 8
    types = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}

    def acc(i, n):
        a = g["accessors"][i]
        v = g["bufferViews"][a["bufferView"]]
        dt = np.dtype(types[a["componentType"]])
        if v.get("byteStride", dt.itemsize * n) != dt.itemsize * n:
            sys.exit("Unsupported layout: re-export from Blender with default glTF settings")
        s = bo + v.get("byteOffset", 0) + a.get("byteOffset", 0)
        return np.frombuffer(b, dt, count=a["count"] * n, offset=s).reshape(-1, n)

    positions, triangles, base = [], [], 0
    for mesh in g["meshes"]:
        for prim in mesh["primitives"]:
            pos = acc(prim["attributes"]["POSITION"], 3).astype(np.float64)
            idx = acc(prim["indices"], 1).reshape(-1, 3).astype(np.int64)
            positions.append(pos)
            triangles.append(idx + base)
            base += len(pos)
    name = g["nodes"][0].get("name", "piece") if g.get("nodes") else "piece"
    return np.concatenate(positions), np.concatenate(triangles), name


# --------------------------------------------------------------------------- profile

def cross_section(pos, tri):
    """Segments where the mesh crosses the half-plane z = 0, x >= 0, as (r, y) pairs."""
    # Skip zero-area triangles: CAD exports often contain long, flat slivers that draw
    # nothing but would add fake lines to the cross-section.
    t = pos[tri]
    area2 = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    tri = tri[area2 > 1e-12]
    segs = []
    for a, b, c in tri:
        pts = []
        for i, j in ((a, b), (b, c), (c, a)):
            zi, zj = pos[i, 2], pos[j, 2]
            if (zi > 0) != (zj > 0):
                t = zi / (zi - zj)
                pts.append(pos[i] + t * (pos[j] - pos[i]))
        if len(pts) == 2:
            p, q = pts
            if p[0] < 0 and q[0] < 0:
                continue
            if p[0] < 0 or q[0] < 0:  # clip at the axis
                if p[0] < 0:
                    p, q = q, p
                t = p[0] / (p[0] - q[0])
                q = p + t * (q - p)
            segs.append(((max(p[0], 0.0), p[1]), (max(q[0], 0.0), q[1])))
    return segs


def chain(segs):
    """Join segments into one polyline running from axis to axis."""
    key = lambda p: (round(p[0] * 1e7), round(p[1] * 1e7))
    adj, pts = {}, {}
    for p, q in segs:
        kp, kq = key(p), key(q)
        if kp == kq:
            continue
        pts[kp], pts[kq] = p, q
        adj.setdefault(kp, []).append(kq)
        adj.setdefault(kq, []).append(kp)
    if any(len(v) > 2 for v in adj.values()):
        sys.exit("Cross-section branches (faces stacked on top of each other): clean the mesh in Blender first")

    # Split the cross-section into separate pieces; holes in the mesh leave gaps between them.
    ends = [k for k, v in adj.items() if len(v) == 1]
    pieces, seen = [], set()
    for e in ends:
        if e in seen:
            continue
        line, prev, cur = [e], None, e
        while True:
            nxt = [k for k in adj[cur] if k != prev]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            line.append(cur)
        seen.update(line)
        pieces.append(line)
    if len(seen) != len(adj):
        sys.exit("Cross-section has a separate closed part: this piece isn't a single round shape")

    on_axis = lambda k: pts[k][0] <= AXIS_TOL

    # The main outline is the piece that reaches the outer wall (largest radius). Start at its
    # end on the center axis (or its lower end) and follow it; whenever it stops at a hole,
    # bridge to the nearest loose end of another piece, until the outline reaches the axis again.
    main = max(pieces, key=lambda p: max(pts[k][0] for k in p))
    a0, a1 = on_axis(main[0]), on_axis(main[-1])
    if (a1 and not a0) or (a0 == a1 and pts[main[0]][1] > pts[main[-1]][1]):
        main = main[::-1]
    line = [pts[k] for k in main]
    rest, gaps, centers = [p for p in pieces if p is not main], [], []
    if line[0][0] > AXIS_TOL:  # hole in the middle of the base
        centers.append(line[0][0])
        line.insert(0, (0.0, line[0][1]))
    while line[-1][0] > AXIS_TOL:
        here = np.array(line[-1])
        best, flip, dist = None, False, np.inf
        for p in rest:
            for idx, rev in ((0, False), (-1, True)):
                d = np.linalg.norm(np.array(pts[p[idx]]) - here)
                if d < dist:
                    best, flip, dist = p, rev, d
        if best is None or dist > MAX_GAP:
            # Nothing near enough to connect to: close the opening straight across to the center.
            centers.append(line[-1][0])
            line.append((0.0, line[-1][1]))
            break
        gaps.append(dist)
        line += [pts[k] for k in (best[::-1] if flip else best)]
        rest = [p for p in rest if p is not best]
    notes = []
    if gaps:
        notes.append(f"Closed {len(gaps)} gap(s) in the mesh: " + ", ".join(f"{g*1000:.1f} mm" for g in gaps))
    if centers:
        notes.append(f"Closed {len(centers)} hole(s) at the center: " + ", ".join(f"{2*c*1000:.1f} mm across" for c in centers))
    if rest:
        notes.append(f"Ignored {len(rest)} loose piece(s) not connected to the outline (stray faces)")
    # Fewer breaks = cleaner slice; used to pick the best slice angle.
    score = (len(gaps) + len(centers) + len(rest), sum(gaps))

    prof, fold_note = remove_folds(np.array(line))
    notes += fold_note
    if prof[0, 1] > prof[-1, 1]:  # the outline must start under the base, not on the inside floor
        prof = prof[::-1]
    prof[0, 0] = prof[-1, 0] = 0.0
    return prof, notes, score


def remove_folds(prof):
    """Drop parts of the outline that run out and straight back over themselves.

    A zero-thickness surface, like a lid left across the opening of a cup, shows up in the
    cross-section as a line that goes to the center and returns along the same path.
    """
    out = []
    for p in prof:
        out.append(p)
        while len(out) >= 3:
            a, b = out[-2] - out[-3], out[-1] - out[-2]
            la, lb = np.linalg.norm(a), np.linalg.norm(b)
            if la == 0 or lb == 0:
                out.pop(-2)
                continue
            straight_back = np.dot(a, b) / (la * lb) < -0.999 and abs(a[0] * b[1] - a[1] * b[0]) < 1e-9
            if not straight_back:
                break
            out.pop(-2)  # the turning point; the overlap shortens until the path no longer folds
    removed = len(prof) - len(out)
    note = [f"Removed a surface that folds back on itself ({removed} outline points), e.g. a lid over an opening"] if removed else []
    return np.array(out), note


def simplify(p, tol):
    """Douglas-Peucker on the profile."""
    if len(p) < 3:
        return p
    a, b = p[0], p[-1]
    ab = b - a
    n = np.linalg.norm(ab)
    rel = p - a
    d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / n if n > 0 else np.linalg.norm(rel, axis=1)
    i = int(np.argmax(d))
    if d[i] > tol:
        return np.vstack([simplify(p[:i + 1], tol)[:-1], simplify(p[i:], tol)])
    return np.array([a, b])


def split_runs(prof, sharp_deg):
    """Split the profile at sharp corners so each run gets its own smooth normals."""
    runs, cur = [], [0]
    for i in range(1, len(prof) - 1):
        u, v = prof[i] - prof[i - 1], prof[i + 1] - prof[i]
        ang = np.degrees(np.arccos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
        cur.append(i)
        if ang > sharp_deg:
            runs.append(cur)
            cur = [i]
    cur.append(len(prof) - 1)
    runs.append(cur)
    return [prof[r] for r in runs]


# --------------------------------------------------------------------------- lathe

def lathe(runs, segments, total_len):
    positions, normals, uvs, indices = [], [], [], []
    walked = 0.0
    for run in runs:
        tan = np.zeros_like(run)
        tan[1:-1] = run[2:] - run[:-2]
        tan[0], tan[-1] = run[1] - run[0], run[-1] - run[-2]
        tan /= np.linalg.norm(tan, axis=1, keepdims=True)
        n2 = np.stack([tan[:, 1], -tan[:, 0]], axis=1)  # (out, up), outward for this winding
        lens = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(run, axis=0), axis=1))])
        v = (walked + lens) / total_len
        walked += lens[-1]
        base, rows = len(positions), len(run)
        for s in range(segments + 1):
            phi = 2 * np.pi * s / segments
            c, sn = np.cos(phi), np.sin(phi)
            for i, (r, y) in enumerate(run):
                positions.append((r * c, y, -r * sn))
                normals.append((n2[i, 0] * c, n2[i, 1], -n2[i, 0] * sn))
                uvs.append((s / segments, v[i]))
        for s in range(segments):
            for i in range(rows - 1):
                a, b = base + s * rows + i, base + (s + 1) * rows + i
                indices += [(a, a + 1, b), (b, a + 1, b + 1)]
    pos = np.array(positions, np.float32)
    nrm = np.array(normals, np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    idx = np.array(indices, np.uint32)
    tri = pos[idx]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    flip = (fn * nrm[idx].sum(1)).sum(1) < 0
    idx[flip] = idx[flip][:, [0, 2, 1]]
    # The profile runs outside-then-inside; if the normals came out inward, turn them around.
    outer = pos[:, 1] < pos[:, 1].max() * 0.5
    if (nrm[outer] * pos[outer] * [1, 0, 1]).sum() < 0:
        nrm = -nrm
        idx = idx[:, [0, 2, 1]]
    # Drop zero-area triangles at the axis.
    tri = pos[idx]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = idx[area > 1e-14]
    return pos, nrm, np.array(uvs, np.float32), idx


# --------------------------------------------------------------------------- writing

def write_glb(path, name, pos, nrm, uv, idx):
    blobs, views, accessors, offset = [], [], [], 0

    def add(arr, target, comp, typ, minmax=False):
        nonlocal offset
        raw = arr.tobytes()
        raw += b"\0" * ((4 - len(raw) % 4) % 4)
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": arr.nbytes, "target": target})
        a = {"bufferView": len(views) - 1, "componentType": comp, "count": len(arr), "type": typ}
        if minmax:
            a["min"], a["max"] = arr.min(0).tolist(), arr.max(0).tolist()
        accessors.append(a)
        blobs.append(raw)
        offset += len(raw)
        return len(accessors) - 1

    p = add(pos, 34962, 5126, "VEC3", True)
    n = add(nrm, 34962, 5126, "VEC3")
    t = add(uv, 34962, 5126, "VEC2")
    i = add(idx.reshape(-1), 34963, 5125, "SCALAR")
    buf = b"".join(blobs)
    size = pos.max(0) - pos.min(0)
    g = {
        "asset": {"version": "2.0", "generator": "lathe-glb"},
        "scene": 0, "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": name, "extras": {
            "units": "meters", "pivot": "center of base",
            "height": round(float(size[1]), 4), "diameter": round(float(size[0]), 4)}}],
        "meshes": [{"name": name, "primitives": [{
            "attributes": {"POSITION": p, "NORMAL": n, "TEXCOORD_0": t}, "indices": i, "material": 0}]}],
        "materials": [{"name": "glaze", "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.8, 0.8, 1], "metallicFactor": 0, "roughnessFactor": 0.85}}],
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(buf)}],
    }
    js = json.dumps(g, separators=(",", ":")).encode()
    js += b" " * ((4 - len(js) % 4) % 4)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, 28 + len(js) + len(buf)))
        f.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        f.write(struct.pack("<II", len(buf), 0x004E4942) + buf)


def main(src, dst, segments=SEGMENTS):
    pos, tri, name = read_mesh(src)
    lo, hi = pos.min(0), pos.max(0)
    pos = pos - [(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2]  # base center to origin
    # Turn the slice plane slightly so it doesn't pass exactly through vertices on the axis.
    # Slice at several angles and keep the cleanest one: a slice through a flawed spot of the
    # mesh breaks the outline into pieces, while the shape itself is the same all the way around.
    best = None
    for a in SLICE_ANGLES:
        rot = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])
        try:
            result = chain(cross_section(pos @ rot, tri))
        except SystemExit as err:
            if best is None:
                best_error = err
            continue
        if best is None or result[2] < best[2]:
            best = result
        if best[2][0] == 0:
            break
    if best is None:
        raise best_error
    prof, notes, _ = best
    for n in notes:
        print(n)
    prof = simplify(prof, SIMPLIFY_TOL)
    total = np.linalg.norm(np.diff(prof, axis=0), axis=1).sum()
    out = lathe(split_runs(prof, SHARP_ANGLE), segments, total)
    write_glb(dst, name, *out)
    size = out[0].max(0) - out[0].min(0)
    print(f"{dst}: profile {len(prof)} points, {len(out[3])} triangles "
          f"(was {len(tri)}), size {size[0]*100:.2f} x {size[1]*100:.2f} cm")
    return prof


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) == 4 else SEGMENTS)
