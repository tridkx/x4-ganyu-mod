# -*- coding: utf-8 -*-
"""Side-by-side preview: the source rig, and the retargeted result.

    python tools/render_compare.py

Renders the PMX exactly as authored (mapped into Blender axes but with no
retargeting) next to the retargeted geometry, so a coordinate or pose mistake
is visible as a *difference* rather than as an opinion about one picture.

Colours are each material's mean texture colour -- enough to tell skin from
cloth from hair without a full texture pipeline.
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "work", "vanilla"))
from pmx import read_pmx                                  # noqa: E402
from ganyu_src import SRC_DIR, SRC_PMX, stem_of          # noqa: E402
import soft_render                                        # noqa: E402

WORK = os.path.join(os.path.dirname(HERE), 'work')
OUT = os.path.join(WORK, 'preview')

#: fallback colours for materials whose texture cannot be sampled
FALLBACK = (0.55, 0.55, 0.58)


def material_colours(model):
    """(F, 3) per-face colour from each material's mean texture colour."""
    cache = {}

    def mean_colour(rel):
        if rel not in cache:
            p = os.path.join(SRC_DIR, rel.replace('\\', os.sep)
                             .replace('/', os.sep))
            try:
                a = np.asarray(Image.open(p).convert('RGB'), float) / 255.0
                cache[rel] = tuple(a.reshape(-1, 3).mean(0))
            except Exception:
                cache[rel] = FALLBACK
        return cache[rel]

    faces = [f for f in model['faces'] if len(set(f)) == 3]
    cols = []
    fi = 0
    allf = model['faces']
    for mt in model['materials']:
        n = mt['face_count'] // 3
        ti = mt['texture']
        rel = (model['textures'][ti]
               if 0 <= ti < len(model['textures']) else None)
        col = mean_colour(rel) if rel else FALLBACK
        d = mt['diffuse']
        col = tuple(c * x for c, x in zip(col, d[:3]))
        cols.extend([col] * n)
        fi += n
    # the face list used for rendering drops degenerate triangles, so colour
    # has to be filtered the same way or every material shifts
    out = [c for f, c in zip(allf, cols) if len(set(f)) == 3]
    return np.array(out, float)


def main():
    os.makedirs(OUT, exist_ok=True)
    m = read_pmx(SRC_PMX)
    cols = material_colours(m)
    faces = np.array([f for f in m['faces'] if len(set(f)) == 3], np.int64)
    print('faces %d, colour rows %d' % (len(faces), len(cols)))
    assert len(faces) == len(cols), 'face/colour mismatch'

    # ------------------------------------------------ source, as authored
    P = np.array(m['positions'], float)
    src = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)     # -> Blender axes
    print('source bbox: X[%.1f %.1f] Y[%.1f %.1f] Z[%.1f %.1f]'
          % (src[:, 0].min(), src[:, 0].max(), src[:, 1].min(),
             src[:, 1].max(), src[:, 2].min(), src[:, 2].max()))
    soft_render.orbit_views(src, faces, cols, OUT, 'src')

    # ------------------------------------------------------- retargeted
    npz = os.path.join(WORK, 'retarget_check.npz')
    if os.path.exists(npz):
        Vx = np.load(npz)['Vx']
        soft_render.orbit_views(Vx, faces, cols, OUT, 'x4')
        print('rendered retargeted result from %s' % npz)
    else:
        print('(run diag_retarget.py first for the x4_* views)')

    print('previews in %s' % OUT)


if __name__ == '__main__':
    main()
