# -*- coding: utf-8 -*-
"""Is a material transparent in the source, and are any two layers coincident?

MMD puts transparency in the **texture alpha**, not in the material's diffuse
alpha (this model's are all 1.0 except 髪+).  Dropping the alpha channel makes
a see-through layer a solid shell, and coincident shells z-fight in the game's
depth buffer -- both of those were real bugs in the previous project, so they
are measured before anything is built.

Sampling is done per material through that material's *own* UVs: a shared atlas
can be opaque where one material samples it and a mask where another does.
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                       # noqa: E402
from ganyu_src import SRC_DIR, SRC_PMX, stem_of   # noqa: E402

#: rough MMD-units -> cm factor, only used to state distances in cm; the exact
#: value comes out of the global fit and does not change any verdict here
SCALE_HINT = 9.6


def sample(img, uv):
    h, w = img.shape[:2]
    u = np.clip(uv[:, 0] % 1.0, 0, 1)
    v = np.clip(uv[:, 1] % 1.0, 0, 1)
    x = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    y = np.clip((v * (h - 1)).astype(int), 0, h - 1)
    return img[y, x]


def main():
    m = read_pmx(SRC_PMX)
    P = np.array(m['positions'], np.float64)
    UV = np.array(m['uvs'], np.float64)
    faces = np.array([f for f in m['faces'] if len(set(f)) == 3], np.int64)

    cache = {}

    def imgs(rel, mode):
        key = (rel, mode)
        if key not in cache:
            p = os.path.join(SRC_DIR, rel.replace('\\', os.sep)
                             .replace('/', os.sep))
            im = Image.open(p)
            cache[key] = (np.asarray(im.convert(mode)), im.mode, im.size)
        return cache[key]

    print('%-9s %-14s %6s  %-30s %s'
          % ('stem', 'texture', 'faces', 'alpha through own UVs', 'verdict'))
    fi = 0
    matverts = {}
    for mt in m['materials']:
        n = mt['face_count'] // 3
        sel = faces[fi:fi + n]
        fi += n
        stem = stem_of(mt['name'])
        ti = mt['texture']
        if not stem or not sel.size or not (0 <= ti < len(m['textures'])):
            continue
        rel = m['textures'][ti]
        idx = np.unique(sel.ravel())
        matverts[stem] = idx
        a, mode, size = imgs(rel, 'RGBA')
        al = sample(a, UV[idx]).astype(float)
        frac_hole = float((al < 200).mean())
        if frac_hole > 0.15:
            verdict = '**NEEDS ALPHA** %.0f%% of area is a hole' % (
                frac_hole * 100)
        elif frac_hole > 0.02:
            verdict = 'some alpha (%.0f%%)' % (frac_hole * 100)
        else:
            verdict = 'opaque'
        print('%-9s %-14s %6d  mean %6.1f  <200 %5.1f%%  %-12s %s'
              % (stem, '%s(%s)' % (os.path.basename(rel), mode), n, al.mean(),
                 frac_hole * 100, '%dx%d' % size, verdict))

    print('\n-- diffuse alpha as authored (material level) --')
    for mt in m['materials']:
        stem = stem_of(mt['name'])
        if stem and abs(mt['diffuse'][3] - 1.0) > 1e-6:
            print('  %-9s diffuse.a = %.3f  sphere_mode=%d'
                  % (stem, mt['diffuse'][3], mt['sphere_mode']))
    print('  (all others are 1.0)')

    print('\n-- coincident vertices from different materials --')
    order = sorted(matverts, key=lambda s: -len(matverts[s]))
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            A = P[matverts[order[i]]] * SCALE_HINT
            B = P[matverts[order[j]]] * SCALE_HINT
            if len(A) * len(B) > 4_000_000:
                A = A[::max(1, len(A) // 2000)]
                B = B[::max(1, len(B) // 2000)]
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
            close = (d < 0.4).sum()
            if close > 20:
                print('  %-9s <-> %-9s  %6d pairs closer than 4 mm (min %.2f mm)'
                      % (order[i], order[j], close, d.min() * 10))


if __name__ == '__main__':
    main()
