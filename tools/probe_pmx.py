# -*- coding: utf-8 -*-
"""Segment-by-segment PMX probe: print the reader position after each block so
a desync can be localised instead of guessed at."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import _Reader                              # noqa: E402


def probe(path):
    data = open(path, 'rb').read()
    r = _Reader(data, 'utf-16-le')
    r.p = 4
    print('version', r.f32())
    ng = r.u8()
    gl = [r.u8() for _ in range(ng)]
    enc = 'utf-16-le' if gl[0] == 0 else 'utf-8'
    add_uv = gl[1]
    sz_vertex, sz_texture, sz_material = gl[2], gl[3], gl[4]
    sz_bone, sz_morph, sz_rigid = gl[5], gl[6], gl[7]
    print('globals', gl)
    print('  add_uv=%d sz_vertex=%d sz_tex=%d sz_mat=%d sz_bone=%d '
          'sz_morph=%d sz_rigid=%d' % (add_uv, sz_vertex, sz_texture,
                                       sz_material, sz_bone, sz_morph,
                                       sz_rigid))
    r = _Reader(data, enc)
    r.p = 4 + 4 + 1 + ng
    for k in ('name', 'name_en', 'comment', 'comment_en'):
        print('  %-11s = %r' % (k, r.text()[:40]))
    print('after header        p=%d' % r.p)

    # ------------------------------------------------------------- vertices
    nv = r.i32()
    print('n_vertices          = %d' % nv)
    wt_hist = {}
    for i in range(nv):
        r.vec(3); r.vec(3); r.vec(2)
        for _ in range(add_uv):
            r.vec(4)
        wt = r.u8()
        wt_hist[wt] = wt_hist.get(wt, 0) + 1
        if wt == 0:
            r.idx(sz_bone)
        elif wt == 1:
            r.idx(sz_bone); r.idx(sz_bone); r.f32()
        elif wt == 2:
            for _ in range(4):
                r.idx(sz_bone)
            for _ in range(4):
                r.f32()
        elif wt == 3:
            r.idx(sz_bone); r.idx(sz_bone); r.f32()
            r.vec(3); r.vec(3); r.vec(3)
        elif wt == 4:
            for _ in range(4):
                r.idx(sz_bone)
            for _ in range(4):
                r.f32()
        else:
            raise SystemExit('!! unknown weight type %d at vertex %d'
                             % (wt, i))
        r.f32()                                        # edge scale
    print('weight types        = %s' % wt_hist)
    print('after vertices      p=%d' % r.p)

    # ---------------------------------------------------------------- faces
    n_idx = r.i32()
    print('n_indices           = %d  (faces %d)' % (n_idx, n_idx // 3))
    for _ in range(n_idx):
        r.idx(sz_vertex, signed=False)
    print('after faces         p=%d' % r.p)

    # ------------------------------------------------------------- textures
    nt = r.i32()
    print('n_textures          = %d' % nt)
    for i in range(nt):
        print('   [%2d] %s' % (i, r.text()))
    print('after textures      p=%d' % r.p)

    # ------------------------------------------------------------ materials
    nm = r.i32()
    print('n_materials         = %d' % nm)
    for i in range(nm):
        name = r.text()
        r.text()
        r.vec(4); r.vec(3); r.f32(); r.vec(3)
        flag = r.u8()
        r.vec(4); r.f32()
        tex = r.idx(sz_texture)
        r.idx(sz_texture)
        r.u8()
        toon_flag = r.u8()
        r.u8() if toon_flag else r.idx(sz_texture)
        r.text()
        fc = r.i32()
        print('   [%2d] %-20s tex=%-3d flag=0x%02x faces=%d'
              % (i, name, tex, flag, fc // 3))
    print('after materials     p=%d' % r.p)

    # ---------------------------------------------------------------- bones
    nb = r.i32()
    print('n_bones             = %d' % nb)
    for i in range(nb):
        name = r.text()
        r.text()
        r.vec(3)
        r.idx(sz_bone); r.i32()
        f = r.u16()
        if f & 0x0001:
            r.idx(sz_bone)
        else:
            r.vec(3)
        if f & (0x0100 | 0x0200):
            r.idx(sz_bone); r.f32()
        if f & 0x0400:
            r.vec(3)
        if f & 0x0800:
            r.vec(3); r.vec(3)
        if f & 0x2000:
            r.i32()
        if f & 0x0020:
            r.idx(sz_bone); r.i32(); r.f32()
            for _ in range(r.i32()):
                r.idx(sz_bone)
                if r.u8():
                    r.vec(3); r.vec(3)
    print('after bones         p=%d' % r.p)

    # --------------------------------------------------------------- morphs
    nmo = r.i32()
    print('n_morphs            = %d' % nmo)
    for i in range(nmo):
        name = r.text()
        r.text(); r.u8()
        ty = r.u8()
        cnt = r.i32()
        print('   [%3d] %-24s type=%d count=%d' % (i, name, ty, cnt))
        for _ in range(cnt):
            if ty == 1:
                r.idx(sz_vertex, signed=False); r.vec(3)
            elif ty == 2:
                r.idx(sz_vertex, signed=False); r.vec(4)
            elif ty in (3, 4, 5, 6, 7):
                r.idx(sz_bone); r.vec(3)
                if ty in (3, 4):
                    r.vec(4)
            elif ty == 8:
                r.idx(sz_material); r.u8(); r.vec(4); r.vec(3); r.f32()
                r.vec(3); r.vec(4); r.f32(); r.vec(4); r.vec(3); r.f32()
            elif ty == 9:
                r.idx(sz_morph); r.f32()
            elif ty == 10:
                r.idx(sz_rigid); r.u8()
            else:
                raise SystemExit('!! unknown morph type %d at morph %d'
                                 % (ty, i))
    print('after morphs        p=%d' % r.p)

    nf = r.i32()
    print('n_frames            = %d' % nf)
    for _ in range(nf):
        r.text(); r.text(); r.u8()
        for _ in range(r.i32()):
            kind = r.u8()
            r.idx(sz_morph if kind else sz_bone)
    print('after frames        p=%d' % r.p)

    nr = r.i32()
    print('n_rigid_bodies      = %d' % nr)
    print('file size           = %d' % len(data))
    print('remaining           = %d' % (len(data) - r.p))


if __name__ == '__main__':
    probe(sys.argv[1])
