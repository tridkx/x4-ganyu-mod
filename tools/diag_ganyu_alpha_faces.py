# -*- coding: utf-8 -*-
"""Where are the transparent triangles, and should they be deleted or kept?

    python tools/diag_ganyu_alpha_faces.py

`diag_alpha_faces.py` counts them; this renders them, because "clear 9%" is
only actionable once you know *which* faces they are:

* a whole hidden panel (the author painting over faces they did not want) --
  deleting the triangles is exactly equivalent and saves the geometry;
* a feathered rim on a real silhouette (a sleeve opening, a lace edge) --
  deleting it eats the edge and the material needs a real alpha mode.

The three groups are rendered separately, in the same camera, so the answer is
a picture rather than a guess.  Group boundaries are the ones the builder uses
(`ganyu_src.ALPHA_KEEP`): a face survives when its mean sampled alpha is at
least that.
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                                  # noqa: E402
from ganyu_src import (SRC_DIR, SRC_PMX, stem_of,         # noqa: E402
                       ALPHA_KEEP, sample_face_alpha)
import soft_render                                        # noqa: E402

WORK = os.path.join(os.path.dirname(HERE), 'work')
OUT = os.path.join(WORK, 'preview')

#: the materials whose atlas carries an alpha mask at all
WATCH = ('cloth', 'skin', 'blush', 'face', 'hair', 'horn', 'iris')


def main():
    os.makedirs(OUT, exist_ok=True)
    m = read_pmx(SRC_PMX)
    P = np.array(m['positions'], float)
    UV = np.array(m['uvs'], float)
    cache = {}

    def alpha_of(rel):
        if rel not in cache:
            p = os.path.join(SRC_DIR, rel.replace('\\', os.sep)
                             .replace('/', os.sep))
            cache[rel] = np.asarray(Image.open(p).convert('RGBA'))[..., 3]
        return cache[rel]

    groups = {}
    fi = 0
    for mt in m['materials']:
        n = mt['face_count'] // 3
        sel = [f for f in m['faces'][fi:fi + n] if len(set(f)) == 3]
        fi += n
        stem = stem_of(mt['name'])
        ti = mt['texture']
        if not stem or not sel or not (0 <= ti < len(m['textures'])):
            continue
        sel = np.array(sel, np.int64)
        fa = sample_face_alpha(alpha_of(m['textures'][ti]), UV[sel])
        if stem in WATCH:
            print('%-9s %6d faces | opaque(>=0.97) %5.1f%%  feather %5.1f%%  '
                  'invisible(<%.2f) %5.1f%%'
                  % (stem, len(sel), (fa >= 0.97).mean() * 100,
                     ((fa >= ALPHA_KEEP) & (fa < 0.97)).mean() * 100,
                     ALPHA_KEEP, (fa < ALPHA_KEEP).mean() * 100))
        groups.setdefault(stem, []).append((sel, fa))

    for stem in WATCH:
        if stem not in groups:
            continue
        for label, keep in (('kept', lambda fa: fa >= ALPHA_KEEP),
                            ('dropped', lambda fa: fa < ALPHA_KEEP)):
            sub = [f for sel, fa in groups[stem] for f in sel[keep(fa)]]
            if not sub:
                continue
            used = sorted({int(i) for f in sub for i in f})
            remap = {v: i for i, v in enumerate(used)}
            V = P[np.array(used)]
            src = np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1)
            faces = np.array([[remap[int(i)] for i in f] for f in sub])
            cols = np.tile(np.array([0.85, 0.80, 0.55]), (len(faces), 1))
            soft_render.orbit_views(src, faces, cols, OUT,
                                    '%s_%s' % (stem, label), angles=(0, 90))
            print('  %-9s %-8s %5d faces -> %s_%s_000.png'
                  % (stem, label, len(faces), stem, label))


if __name__ == '__main__':
    main()
