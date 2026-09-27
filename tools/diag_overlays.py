# -*- coding: utf-8 -*-
"""Where do the overlay materials actually sit, and how much do they cover?

`elem` (元素1.png) and `hairspec` (髪+/sp.png) are additive-ish layers whose
textures are 88-98% transparent, and both are *coincident* with a base layer:

    hair <-> hairspec   3380 vertex pairs closer than 4 mm (min 0.00)
    cloth <-> elem       292
    ribbon <-> elem      216

Coincident shells z-fight in the engine's depth buffer, which reads in game as
flickering -- the previous project's "black stockings flicker" was exactly this.
Before dropping a layer, though, it is worth seeing what it draws: this renders
each overlay alone, and measures how much of it is hidden inside a base shell.
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                                  # noqa: E402
from ganyu_src import SRC_DIR, SRC_PMX, stem_of          # noqa: E402
import soft_render                                        # noqa: E402

WORK = os.path.join(os.path.dirname(HERE), 'work')
OUT = os.path.join(WORK, 'preview')
OVERLAYS = ('elem', 'hairspec')


def main():
    os.makedirs(OUT, exist_ok=True)
    m = read_pmx(SRC_PMX)
    P = np.array(m['positions'], float)
    UV = np.array(m['uvs'], float)

    # per-material face ranges and uv-sampled alpha
    cache = {}

    def alpha_of(rel):
        if rel not in cache:
            p = os.path.join(SRC_DIR, rel.replace('\\', os.sep)
                             .replace('/', os.sep))
            cache[rel] = np.asarray(Image.open(p).convert('RGBA'))[..., 3]
        return cache[rel]

    def sample(img, uv):
        h, w = img.shape[:2]
        u = np.clip(uv[..., 0] % 1.0, 0, 1)
        v = np.clip(uv[..., 1] % 1.0, 0, 1)
        x = np.clip((u * (w - 1)).astype(int), 0, w - 1)
        y = np.clip((v * (h - 1)).astype(int), 0, h - 1)
        return img[y, x].astype(float)

    verts_of = {}
    fi = 0
    for mt in m['materials']:
        n = mt['face_count'] // 3
        fi += n
        stem = stem_of(mt['name'])
        if not stem:
            continue
        sel = m['faces'][fi - n:fi]
        verts_of.setdefault(stem, set()).update(
            i for f in sel if len(set(f)) == 3 for i in f)

    # how far is each overlay vertex from the nearest vertex of its base layer?
    print('-- overlay coverage --')
    print('%-9s %6s %6s  %-34s %s'
          % ('stem', 'verts', 'faces', 'alpha through own UVs', 'verdict'))
    for stem in OVERLAYS:
        if stem not in verts_of:
            continue
        idx = sorted(verts_of[stem])
        ti = None
        for mt in m['materials']:
            if stem_of(mt['name']) == stem:
                ti = mt['texture']
                break
        a = alpha_of(m['textures'][ti])
        al = sample(a, UV[idx])
        print('%-9s %6d %6s  visible %5.1f%%  alpha mean %5.1f  %s'
              % (stem, len(idx), '-',
                 float((al > 128).mean()) * 100, al.mean(), ''))

    # distance from each overlay vertex to the base layer it sits on
    base_of = {'elem': ('cloth', 'ribbon'), 'hairspec': ('hair',)}
    for stem, bases in base_of.items():
        if stem not in verts_of:
            continue
        A = P[sorted(verts_of[stem])]
        B = np.vstack([P[sorted(verts_of[b])] for b in bases if b in verts_of])
        if not len(B):
            continue
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
        near = d.min(axis=1) * 9.77462 * 10          # mm
        print('%-9s -> %-16s %5.1f%% of its vertices are within 1 mm of a base '
              'vertex (median %.1f mm)'
              % (stem, '+'.join(bases), float((near < 1.0).mean()) * 100,
                 float(np.median(near))))

    # ------------------------------------------------------------- renders
    print('\n-- rendering each overlay alone --')
    faces = np.array([f for f in m['faces'] if len(set(f)) == 3], np.int64)
    for stem in OVERLAYS + ('hair', 'cloth', 'ribbon', 'skin'):
        if stem not in verts_of:
            continue
        keep = np.array(sorted(verts_of[stem]))
        remap = {v: i for i, v in enumerate(keep)}
        sub = []
        for f in faces:
            if all(int(v) in remap for v in f):
                sub.append([remap[int(v)] for v in f])
        if not sub:
            continue
        Vs = P[keep]
        src = np.stack([Vs[:, 0], -Vs[:, 2], Vs[:, 1]], 1)
        cols = np.tile(np.array([0.85, 0.80, 0.55]), (len(sub), 1))
        soft_render.orbit_views(src, np.array(sub), cols, OUT, 'only_' + stem,
                                angles=(0, 90))
        print('  %-9s %5d faces -> only_%s_000.png' % (stem, len(sub), stem))

    print('\npreviews in %s' % OUT)


if __name__ == '__main__':
    main()
