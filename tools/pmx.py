# -*- coding: utf-8 -*-
"""Minimal PMX (MikuMikuDance) reader.

Reads geometry with skin weights, materials and the bone tree.  PMX stores
everything in one binary blob; the text encoding and the width of every index
type are declared in the header, so nothing can be assumed.

Coordinate space: MMD is Y-up.  Which horizontal axis is "forward" is NOT
assumed here -- it is decided by anatomy later (see ganyu_mmd_to_x4.py), because
guessing it is exactly the mistake that cost the previous project a dozen
rounds.
"""

import struct

MAGIC = b'PMX '


class PmxError(Exception):
    pass


class _Reader:
    def __init__(self, data, encoding):
        self.d = data
        self.p = 0
        self.enc = encoding

    def raw(self, n):
        if self.p + n > len(self.d):
            raise PmxError('read past end of file')
        v = self.d[self.p:self.p + n]
        self.p += n
        return v

    def u8(self):
        return self.raw(1)[0]

    def i8(self):
        return struct.unpack('<b', self.raw(1))[0]

    def u16(self):
        return struct.unpack('<H', self.raw(2))[0]

    def i32(self):
        return struct.unpack('<i', self.raw(4))[0]

    def f32(self):
        return struct.unpack('<f', self.raw(4))[0]

    def vec(self, n):
        return struct.unpack('<%df' % n, self.raw(4 * n))

    def text(self):
        n = self.i32()
        return self.raw(n).decode(self.enc, errors='replace')

    def idx(self, size, signed=True):
        if size == 1:
            return self.i8() if signed else self.u8()
        if size == 2:
            return struct.unpack('<h' if signed else '<H', self.raw(2))[0]
        if size == 4:
            return self.i32() if signed else struct.unpack('<I', self.raw(4))[0]
        raise PmxError('bad index size %d' % size)


def read_pmx(path):
    data = open(path, 'rb').read()
    if data[:4] != MAGIC:
        raise PmxError('not a PMX file: %r' % data[:4])
    r = _Reader(data, 'utf-16-le')          # placeholder, replaced below

    r.p = 4
    version = r.f32()
    n_globals = r.u8()
    gl = [r.u8() for _ in range(n_globals)]
    enc = 'utf-16-le' if gl[0] == 0 else 'utf-8'
    add_uv = gl[1]
    sz_vertex, sz_texture, sz_material = gl[2], gl[3], gl[4]
    sz_bone, sz_morph, sz_rigid = gl[5], gl[6], gl[7]

    r = _Reader(data, enc)
    r.p = 4 + 4 + 1 + n_globals          # magic + version + count + globals
    model = {
        'version': version,
        'encoding': enc,
        'additional_uv': add_uv,
        'name': r.text(),
        'name_en': r.text(),
        'comment': r.text(),
        'comment_en': r.text(),
    }

    # ---------------------------------------------------------------- vertices
    nv = r.i32()
    positions, normals, uvs, weights, edges = [], [], [], [], []
    for _ in range(nv):
        positions.append(r.vec(3))
        normals.append(r.vec(3))
        uvs.append(r.vec(2))
        for _ in range(add_uv):
            r.vec(4)
        wt = r.u8()
        if wt == 0:                                   # BDEF1
            b0 = r.idx(sz_bone)
            w = [(b0, 1.0)]
        elif wt == 1:                                 # BDEF2
            b0, b1, w0 = r.idx(sz_bone), r.idx(sz_bone), r.f32()
            w = [(b0, w0), (b1, 1.0 - w0)]
        elif wt == 2:                                 # BDEF4
            bs = [r.idx(sz_bone) for _ in range(4)]
            ws = [r.f32() for _ in range(4)]
            w = list(zip(bs, ws))
        elif wt == 3:                                 # SDEF
            b0, b1, w0 = r.idx(sz_bone), r.idx(sz_bone), r.f32()
            r.vec(3); r.vec(3); r.vec(3)              # C, R0, R1 unused here
            w = [(b0, w0), (b1, 1.0 - w0)]
        elif wt == 4:                                 # QDEF (2.1)
            bs = [r.idx(sz_bone) for _ in range(4)]
            ws = [r.f32() for _ in range(4)]
            w = list(zip(bs, ws))
        else:
            raise PmxError('unknown weight type %d' % wt)
        edges.append(r.f32())
        weights.append([(b, x) for b, x in w if x > 1e-6])
    model['positions'] = positions
    model['normals'] = normals
    model['uvs'] = uvs
    model['weights'] = weights

    # ------------------------------------------------------------------- faces
    n_idx = r.i32()
    faces = []
    for _ in range(n_idx // 3):
        faces.append(tuple(r.idx(sz_vertex, signed=False) for _ in range(3)))
    model['faces'] = faces

    # ---------------------------------------------------------------- textures
    model['textures'] = [r.text() for _ in range(r.i32())]

    # --------------------------------------------------------------- materials
    mats = []
    for _ in range(r.i32()):
        m = {
            'name': r.text(),
            'name_en': r.text(),
            'diffuse': r.vec(4),
            'specular': r.vec(3),
            'specularity': r.f32(),
            'ambient': r.vec(3),
            'flag': r.u8(),
            'edge_color': r.vec(4),
            'edge_size': r.f32(),
            'texture': r.idx(sz_texture),
            'sphere': r.idx(sz_texture),
            'sphere_mode': r.u8(),
            'toon_flag': r.u8(),
            'toon': 0,
        }
        # toon reference is a byte (built-in ramp) when toon_flag == 1,
        # otherwise a texture index
        m['toon'] = (r.u8() if m['toon_flag'] else r.idx(sz_texture))
        m['memo'] = r.text()
        m['face_count'] = r.i32()          # number of *indices*, not faces
        mats.append(m)
    model['materials'] = mats

    # ------------------------------------------------------------------- bones
    bones = []
    for _ in range(r.i32()):
        b = {
            'name': r.text(),
            'name_en': r.text(),
            'position': r.vec(3),
            'parent': r.idx(sz_bone),
            'layer': r.i32(),
            'flags': r.u16(),
        }
        f = b['flags']
        if f & 0x0001:                              # tail is another bone
            b['tail_bone'] = r.idx(sz_bone)
            b['tail'] = None
        else:
            b['tail'] = r.vec(3)
            b['tail_bone'] = -1
        if f & (0x0100 | 0x0200):                   # inherit rotation/translation
            b['inherit_parent'] = r.idx(sz_bone)
            b['inherit_weight'] = r.f32()
        else:
            b['inherit_parent'] = -1
            b['inherit_weight'] = 0.0
        if f & 0x0400:                              # fixed axis
            b['fixed_axis'] = r.vec(3)
        if f & 0x0800:                              # local axis
            b['local_x'] = r.vec(3)
            b['local_z'] = r.vec(3)
        if f & 0x2000:                              # external parent deform
            b['key'] = r.i32()
        if f & 0x0020:                              # IK
            b['ik_target'] = r.idx(sz_bone)
            b['ik_loops'] = r.i32()
            b['ik_limit'] = r.f32()
            links = []
            for _ in range(r.i32()):
                li = {'bone': r.idx(sz_bone), 'limit': r.u8()}
                if li['limit']:
                    li['lower'] = r.vec(3)
                    li['upper'] = r.vec(3)
                links.append(li)
            b['ik_links'] = links
        bones.append(b)
    model['bones'] = bones

    # ------------------------------------------------------------------ morphs
    morphs = []
    for _ in range(r.i32()):
        m = {'name': r.text(), 'name_en': r.text(), 'panel': r.u8(),
             'type': r.u8(), 'offset_count': r.i32()}
        m['offset_start'] = r.p
        for _ in range(m['offset_count']):
            if m['type'] == 1:                      # vertex morph
                r.idx(sz_vertex, signed=False); r.vec(3)
            elif m['type'] == 2:                    # UV morph
                r.idx(sz_vertex, signed=False); r.vec(4)
            elif m['type'] in (3, 4, 5, 6, 7):      # bone morphs
                r.idx(sz_bone); r.vec(3)
                if m['type'] in (3, 4):
                    r.vec(4)
            elif m['type'] == 8:                    # material morph
                # CalcMode, diffuse[4], specular[3], specularity, ambient[3],
                # edgeColor[4], edgeSize, then the texture / sphere / toon
                # *offsets*, which are all float[4] (RGBA multipliers).  This
                # is easy to misread as mixed widths; getting it wrong desyncs
                # every later block (frames, rigid bodies) without any error
                # until something reads past the end.
                r.idx(sz_material); r.u8()
                r.vec(4); r.vec(3); r.f32(); r.vec(3)
                r.vec(4); r.f32()
                r.vec(4); r.vec(4); r.vec(4)
            elif m['type'] == 9:                    # flip morph
                r.idx(sz_morph); r.f32()
            elif m['type'] == 10:                   # impulse morph
                r.idx(sz_rigid); r.u8()
            else:
                raise PmxError('unknown morph type %d' % m['type'])
        morphs.append(m)
    model['morphs'] = morphs

    # ----------------------------------------------------------- display frames
    frames = []
    for _ in range(r.i32()):
        fr = {'name': r.text(), 'name_en': r.text(), 'special': r.u8(),
              'items': []}
        for _ in range(r.i32()):
            kind = r.u8()
            index = r.idx(sz_morph) if kind else r.idx(sz_bone)
            fr['items'].append((kind, index))
        frames.append(fr)
    model['frames'] = frames

    model['bytes_read'] = r.p
    model['bytes_total'] = len(data)
    return model


def bone_children(bones):
    """{bone name -> [child names]} in declaration order."""
    out = {}
    for i, b in enumerate(bones):
        p = b['parent']
        if p >= 0:
            out.setdefault(bones[p]['name'], []).append(b['name'])
    return out


def summary(model):
    lines = []
    lines.append('PMX %.1f  "%s" (%s)' % (model['version'], model['name'],
                                          model['encoding']))
    lines.append('  vertices %d   faces %d   materials %d   bones %d   '
                 'textures %d   morphs %d'
                 % (len(model['positions']), len(model['faces']),
                    len(model['materials']), len(model['bones']),
                    len(model['textures']), len(model['morphs'])))
    lines.append('  bytes read %d / %d' % (model['bytes_read'],
                                           model['bytes_total']))
    return '\n'.join(lines)
