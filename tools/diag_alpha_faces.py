# -*- coding: utf-8 -*-
"""Per-material alpha histogram, measured per *face* rather than per vertex.

`diag_alpha.py` samples the atlas through every vertex a material uses, which
counts a vertex that merely *sits* in an unused corner of the sheet as a hole.
What decides the blend mode is how much of the material's actual *surface* is
see-through, so this bins the triangles instead:

    opaque   mean alpha > 200   -- renders fine as NONE
    cutout   0 < a <= 200       -- an edge/feather: ALPHA1 keeps the shape
    clear    mean alpha == 0    -- a real hole; NONE would paint it solid

A material is "mostly opaque with a feathered rim" when most faces are opaque
and only a minority are cutout -- that is the vanilla clothes case.  A material
whose faces are mostly clear or mostly cutout is a genuine alpha surface.
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                       # noqa: E402
from ganyu_src import SRC_DIR, SRC_PMX, stem_of   # noqa: E402


def sample(img, uv):
    # `uv` may be (N, 2) or (F, 3, 2); index the *last* axis.  Using `uv[:, 0]`
    # here silently takes column 0 of a per-corner array, which lands on
    # unrelated pixels and reported every material as opaque.
    h, w = img.shape[:2]
    u = np.clip(uv[..., 0] % 1.0, 0, 1)
    v = np.clip(uv[..., 1] % 1.0, 0, 1)
    x = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    y = np.clip((v * (h - 1)).astype(int), 0, h - 1)
    return img[y, x].astype(float)


def main():
    m = read_pmx(SRC_PMX)
    UV = np.array(m['uvs'], np.float64)
    faces = np.array([f for f in m['faces'] if len(set(f)) == 3], np.int64)

    cache = {}

    def alpha_of(rel):
        if rel not in cache:
            p = os.path.join(SRC_DIR, rel.replace('\\', os.sep)
                             .replace('/', os.sep))
            im = Image.open(p)
            cache[rel] = np.asarray(im.convert('RGBA'))[..., 3]
        return cache[rel]

    print('%-9s %-11s %6s | %7s %7s %7s | %s'
          % ('stem', 'texture', 'faces', 'opaque', 'cutout', 'clear',
             'verdict'))
    print('-' * 88)
    fi = 0
    verdicts = {}
    for mt in m['materials']:
        n = mt['face_count'] // 3
        sel = faces[fi:fi + n]
        fi += n
        stem = stem_of(mt['name'])
        ti = mt['texture']
        if not stem or not sel.size or not (0 <= ti < len(m['textures'])):
            continue
        a = alpha_of(m['textures'][ti])
        fa = sample(a, UV[sel]).mean(axis=1)      # per-face mean alpha
        opaque = float((fa > 200).mean())
        clear = float((fa <= 1).mean())
        cutout = 1.0 - opaque - clear
        if clear > 0.25:
            v = '** real holes (clear %.0f%%)' % (clear * 100)
        elif cutout > 0.5:
            v = '** mostly soft alpha (cutout %.0f%%)' % (cutout * 100)
        elif cutout > 0.10:
            v = 'feathered rim (cutout %.0f%%)' % (cutout * 100)
        elif cutout > 0.01:
            v = 'few soft faces (%.0f%%)' % (cutout * 100)
        else:
            v = 'opaque'
        verdicts[stem] = (opaque, cutout, clear, v)
        print('%-9s %-11s %6d | %6.1f%% %6.1f%% %6.1f%% | %s'
              % (stem, os.path.basename(m['textures'][ti]), n,
                 opaque * 100, cutout * 100, clear * 100, v))

    print()
    print('-- materials that are NOT safely opaque --')
    for stem, (o, c, cl, v) in sorted(verdicts.items()):
        if o < 0.99:
            print('  %-9s opaque %5.1f%%  cutout %5.1f%%  clear %5.1f%%'
                  % (stem, o * 100, c * 100, cl * 100))


if __name__ == '__main__':
    main()
