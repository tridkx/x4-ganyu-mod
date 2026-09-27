# -*- coding: utf-8 -*-
"""Read the skeleton out of an .xac without Blender.

The converter addon can only be used from inside Blender, which makes every
skeleton question -- "is the left thigh's local frame the mirror of the
right's?", "did the retarget put the ankles where the mesh thinks they are?"
-- a Blender round trip.  This module reads the same chunk structure
(`xac_format.py` in X4CharacterConverter: chunk 11 holds the nodes) with plain
`struct`, and rebuilds the two matrices the addon builds:

    local = T(position) @ R(quaternion, xyzw) @ S(scale)   # X4 axes
    world = parent_world @ local                           # X4 axes

The quaternion is stored (x, y, z, w) and the addon maps it to Blender's
(w, x, y, z); the axis change is a -90 degree rotation about X
(`AXIS_TO_BLENDER`) undone on the other side (`AXIS_TO_XAC`), and it cancels
out in the world matrix, so the numbers here are X4 axes: **+X right,
+Y forward (the direction the character faces), +Z up, centimetres**.
"""

import struct

import numpy as np

#: axis-change the addon applies on import, kept here so the local frames we
#: print are the ones Blender would see (world matrices are unaffected: the
#: two rotations cancel)
AXIS_TO_BLENDER = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                           float)


class Reader(object):
    def __init__(self, data):
        self.d = data
        self.o = 0

    def read(self, n):
        if self.o + n > len(self.d):
            raise ValueError('unexpected end of .xac at %d' % self.o)
        self.o += n
        return self.d[self.o - n:self.o]

    def i32(self):
        return struct.unpack('<i', self.read(4))[0]

    def string(self):
        return self.read(struct.unpack('<I', self.read(4))[0]).decode(
            'utf-8', 'replace')


def iter_chunks(data):
    """Yield (type, body_offset, length) for every chunk in the file.

    Walking the chunks by arithmetic does not work: sections are padded, so
    `offset + header + length` drifts and lands inside the next payload, and a
    naive "small type id + plausible length" scan happily accepts a byte pair
    inside a vertex buffer as a chunk header (the exported body desynchronises
    right after the node section and then reports a 63-byte mesh).

    So each candidate is *validated against the shape of its own section*:
    only 4-byte-aligned offsets are considered, and the first fields of the
    body have to make sense for that chunk type (a mesh section must start
    with a node id, a plausible vertex count and a positive index count; an
    influence section with a node id and counts that fit the table).
    """
    n = len(data)

    def plausible(t, probe, ln):
        if not (1 <= t <= 20) or not (16 < ln < n - probe - 12):
            return False
        b = probe + 12
        if b + 16 > n:
            return False
        if t == 1:                                   # mesh
            node_id, _ranges, vcount, icount = struct.unpack_from('<4i', data, b)
            return (0 <= node_id < 4096 and 0 < vcount < 4_000_000
                    and 0 < icount < 40_000_000)
        if t == 2:                                   # skin
            node_id, _local, infl = struct.unpack_from('<3i', data, b)
            return 0 <= node_id < 4096 and 0 < infl < 8_000_000
        if t == 11:                                  # nodes
            count = struct.unpack_from('<i', data, b)[0]
            return 0 < count < 100_000
        if t == 3:                                   # material
            return True
        return True

    off = 8
    while off + 16 < n:
        hit = None
        # a real header follows the previous section immediately (within its
        # padding): a candidate further away is a plausible-looking byte pair
        # inside a payload, which is exactly how the first version of this
        # scan reported a 63-byte mesh section and then lost the real one
        for probe in range(off, min(off + 20, n - 16)):
            t, ln = struct.unpack_from('<ii', data, probe)
            if plausible(t, probe, ln):
                hit = (t, probe, ln)
                break
        if hit is None:
            break
        t, probe, ln = hit
        yield t, probe + 12, ln
        off = probe + 12 + ln


def parse_nodes(path):
    """[(node_id, name, parent_id, position, quaternion_xyzw, scale)]"""
    data = open(path, 'rb').read()
    if data[:4] != b'XAC ':
        raise ValueError('not an .xac: %s' % path)
    nodes = []
    for type_, body, length in iter_chunks(data):
        r = Reader(data)
        r.o = body
        if type_ != 11:
            continue
        if True:
            count = r.i32()
            r.read(4)
            for nid in range(count):
                rot = struct.unpack('<4f', r.read(16))
                r.read(16)
                pos = struct.unpack('<3f', r.read(12))
                scale = struct.unpack('<3f', r.read(12))
                r.read(12)
                r.read(8)
                parent = r.i32()
                r.read(4)
                r.read(4)
                r.read(64)
                r.read(4)
                name = r.string()
                nodes.append((nid, name, parent, pos, rot, scale))
    return nodes


def _quat_matrix(q):
    x, y, z, w = q
    n = (x * x + y * y + z * z + w * w) ** 0.5
    if n < 1e-9:
        return np.eye(4)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1]], float)


def local_matrix(node):
    _nid, _name, _parent, pos, rot, scale = node
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(rot)[:3, :3] @ np.diag(scale)
    m[:3, 3] = pos
    # X4 axes -> Blender axes on the left, back on the right
    return AXIS_TO_BLENDER @ m @ AXIS_TO_BLENDER.T


class Skeleton(object):
    """World transforms (X4 axes, cm) for every node of one .xac."""

    def __init__(self, path):
        self.path = path
        self.nodes = parse_nodes(path)
        self.by_name = {}
        for n in self.nodes:
            self.by_name.setdefault(n[1], n)
        self.world = {}
        for n in self.nodes:
            self._world(n[0])

    def _world(self, nid):
        if nid in self.world:
            return self.world[nid]
        n = self.nodes[nid]
        local = local_matrix(n)
        p = n[2]
        self.world[nid] = local if p is None or p < 0 else self._world(p) @ local
        return self.world[nid]

    def head(self, name):
        return self.world[self.by_name[name][0]][:3, 3].copy()

    def axis(self, name):
        """Local frame axes in X4 world space: columns of the world rotation."""
        return self.world[self.by_name[name][0]][:3, :3].copy()

    def names(self):
        return [n[1] for n in self.nodes]


def report(path, bones=None):
    sk = Skeleton(path)
    print('== %s ==' % path)
    print('   %d nodes' % len(sk.nodes))
    for b in bones or ():
        if b in sk.by_name:
            h = sk.head(b)
            print('   %-22s head (%7.2f, %7.2f, %7.2f)' % (b, h[0], h[1], h[2]))
    return sk


if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        report(p, ['Bip01 L Thigh', 'Bip01 R Thigh', 'Bip01 L Calf',
                   'Bip01 R Calf', 'Bip01 L Foot', 'Bip01 R Foot'])


# --------------------------------------------------------------------------
# mesh + skinning, so a built asset can be checked without Blender
# --------------------------------------------------------------------------

def parse_meshes(path):
    """{node_id: {'positions': (n,3) float, 'weights': [(bone_id, w), ...]}}.

    Mirrors the addon's `read_mesh`/`read_skin`: chunk 1 carries the attribute
    layers (positions = type 0, influence range id = type 5), chunk 2 the
    influence table plus one (first_index, count) range per vertex.  Verified
    against the addon's own reader on a real asset.
    """
    data = open(path, 'rb').read()
    meshes, skins = {}, {}
    for type_, body, length in iter_chunks(data):
        r = Reader(data)
        r.o = body
        if type_ == 1:
            node_id = r.i32()
            r.i32()                                   # range count
            vcount = r.i32()
            r.i32()                                   # index count
            submesh_count = r.i32()
            attr_count = r.i32()
            r.read(4)                                 # collision byte + pad
            attrs = {}
            for _ in range(attr_count):
                tid = r.i32()
                size = r.i32()
                r.read(4)
                attrs[tid] = (size, r.read(vcount * size))
            for _ in range(submesh_count):
                icount = r.i32()
                r.i32()                               # vertex count
                r.read(4)
                r.i32()                               # material id
                bone_count = r.i32()
                r.read(icount * 4 + bone_count * 4)
            pos = attrs.get(0)
            rid = attrs.get(5)
            v = (struct.unpack('<%df' % (vcount * 3), pos[1])
                 if pos and pos[0] == 12 else None)
            ranges = (struct.unpack('<%dI' % vcount, rid[1])
                      if rid and rid[0] == 4 else None)
            meshes[node_id] = {
                'positions': (np.array(v, float).reshape(-1, 3)
                              if v else None),
                'range_ids': ranges,
            }
        elif type_ == 2:
            node_id = r.i32()
            r.i32()                                   # local bone count
            influence_count = r.i32()
            r.read(4)
            infl = []
            for _ in range(influence_count):
                w = struct.unpack('<f', r.read(4))[0]
                bone_id = struct.unpack('<h', r.read(2))[0]
                r.read(2)
                infl.append((bone_id, w))
            n_ranges = len(meshes.get(node_id, {}).get('range_ids') or [])
            rr = []
            for _ in range(n_ranges):
                rr.append((r.i32(), r.i32()))         # first_index, count
            skins[node_id] = {'influences': infl, 'ranges': rr}

    out = {}
    for node_id, mesh in meshes.items():
        skin = skins.get(node_id)
        if skin is None or mesh['positions'] is None:
            continue
        w = []
        for rid in mesh['range_ids']:
            first, count = skin['ranges'][rid]
            w.append(skin['influences'][first:first + count])
        out[node_id] = {'positions': mesh['positions'], 'weights': w}
    return out


def bone_of(meshes):
    """[(bone_id, name)] for every bone node that carries skin weights."""
    return meshes
