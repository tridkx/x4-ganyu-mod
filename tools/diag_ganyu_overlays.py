# -*- coding: utf-8 -*-
"""Which of Ganyu's 19 materials sit on top of which, and how visible they are.

    python tools/diag_ganyu_overlays.py

The Lumine project's `diag_overlays.py` hard-coded the two overlays its model
had (`elem`, `hairspec`).  This model has a different set, so the question is
asked of *every* material pair instead of a guessed list:

* **coincident shells z-fight** in the engine's depth buffer -- the symptom is
  a surface that flickers between two colours (the previous project shipped
  black stockings that flickered for exactly this reason).  A layer is only
  safe to keep if it is offset from its base, or safe to drop if it draws
  nothing;
* the deciding number is therefore **how much of the overlay is actually
  visible** (alpha > 128 through its own UVs) *and* how close it sits to the
  base layer's vertices.

Distances are reported in **source units** (MMD) and in **millimetres after
retargeting**, using the global fit scale measured in
`docs/甘雨移植进展.md`.  Units are converted in one place so the two cannot
disagree.
"""

import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                                  # noqa: E402
from ganyu_src import SRC_DIR, SRC_PMX, stem_of, MAT_ASCII  # noqa: E402
import soft_render                                        # noqa: E402

WORK = os.path.join(os.path.dirname(HERE), 'work')
OUT = os.path.join(WORK, 'preview')

#: cm per source unit.  Filled in from the stage-1 fit; the default is close
#: enough to report useful millimetres before the fit has run (Ganyu is
#: 18.6 units to the crown, an X4 woman is ~170 cm).
SCALE_CM = 9.14

#: "coincident" in source units: 0.01 unit ~ 0.9 mm, well under the engine's
#: depth resolution at NPC range and far under anything visible.
NEAR = 0.01


def main():
    os.makedirs(OUT, exist_ok=True)
    m = read_pmx(SRC_PMX)
    P = np.array(m['positions'], float)
    UV = np.array(m['uvs'], float)

    verts_of, tex_of, faces_of = {}, {}, {}
    fi = 0
    for mt in m['materials']:
        n = mt['face_count'] // 3
        sel = m['faces'][fi:fi + n]
        fi += n
        stem = stem_of(mt['name'])
        if not stem:
            continue
        vs = set()
        for f in sel:
            if len(set(f)) == 3:
                vs.update(int(i) for i in f)
        verts_of.setdefault(stem, set()).update(vs)
        tex_of.setdefault(stem, m['textures'][mt['texture']]
                          if 0 <= mt['texture'] < len(m['textures']) else None)

    alpha_cache = {}

    def alpha_of(rel):
        if rel not in alpha_cache:
            p = os.path.join(SRC_DIR, rel.replace('\\', os.sep)
                             .replace('/', os.sep))
            alpha_cache[rel] = np.asarray(Image.open(p).convert('RGBA'))[..., 3]
        return alpha_cache[rel]

    def sample_alpha(stem):
        rel = tex_of.get(stem)
        if not rel:
            return None
        img = alpha_of(rel)
        idx = np.array(sorted(verts_of[stem]))
        h, w = img.shape[:2]
        uv = UV[idx]
        x = np.clip((np.clip(uv[:, 0] % 1.0, 0, 1) * (w - 1)).astype(int), 0, w - 1)
        y = np.clip((np.clip(uv[:, 1] % 1.0, 0, 1) * (h - 1)).astype(int), 0, h - 1)
        return img[y, x].astype(float)

    stems = sorted(verts_of)
    print('-- material inventory --')
    print('%-9s %6s %7s  %-14s %s' % ('stem', 'verts', 'faces', 'texture',
                                      'visible (alpha>128 through own UVs)'))
    for s in stems:
        nf = sum((mt['face_count'] // 3) for mt in m['materials']
                 if stem_of(mt['name']) == s)
        a = sample_alpha(s)
        vis = '-' if a is None else '%5.1f%%' % (float((a > 128).mean()) * 100)
        print('%-9s %6d %7d  %-14s %s'
              % (s, len(verts_of[s]), nf, (tex_of.get(s) or '-').split('/')[-1],
                 vis))

    # ------------------------------------------------ coincidence matrix
    from scipy.spatial import cKDTree
    pts = {s: P[np.array(sorted(verts_of[s]))] for s in stems}
    trees = {s: cKDTree(pts[s]) for s in stems}

    print('\n-- how close is each material to each other material --')
    print('(a pair counts as coincident when >50%% of the smaller layer\'s')
    print(' vertices have a neighbour within %.0f mm)'
          % (NEAR * SCALE_CM * 10))
    print('%-9s %-9s %8s %8s %8s  %s'
          % ('A', 'B', 'pairs', '% near', 'min mm', 'verdict'))
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            d, _ = trees[b].query(pts[a], k=1)
            near = float((d < NEAR).mean())
            d2, _ = trees[a].query(pts[b], k=1)
            near2 = float((d2 < NEAR).mean())
            frac = max(near, near2)
            if frac < 0.02:
                continue
            small = a if len(pts[a]) <= len(pts[b]) else b
            mm = float(min(d.min(), d2.min())) * SCALE_CM * 10
            verdict = ('COINCIDENT' if frac > 0.5 else 'partial overlap')
            print('%-9s %-9s %8d %7.1f%% %8.2f  %s (%s is the smaller layer)'
                  % (a, b, int(round(frac * min(len(pts[a]), len(pts[b])))),
                     frac * 100, mm, verdict, small))

    # ------------------------------------------------------------- renders
    print('\n-- rendering every material on its own --')
    faces = np.array([f for f in m['faces'] if len(set(f)) == 3], np.int64)
    allv = {}
    fi = 0
    for mt in m['materials']:
        n = mt['face_count'] // 3
        sel = m['faces'][fi:fi + n]
        fi += n
        stem = stem_of(mt['name'])
        if stem:
            allv.setdefault(stem, []).extend(sel)
    for stem, sel in sorted(allv.items()):
        keep = np.array(sorted(verts_of[stem]))
        remap = {int(v): i for i, v in enumerate(keep)}
        sub = [[remap[int(v)] for v in f] for f in sel if len(set(f)) == 3]
        Vs = P[keep]
        src = np.stack([Vs[:, 0], -Vs[:, 2], Vs[:, 1]], 1)
        cols = np.tile(np.array([0.85, 0.80, 0.55]), (len(sub), 1))
        soft_render.orbit_views(src, np.array(sub), cols, OUT,
                                'only_' + stem, angles=(0, 90))
    print('  %d previews in %s' % (len(allv), OUT))


if __name__ == '__main__':
    main()
