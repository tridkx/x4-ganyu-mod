# -*- coding: utf-8 -*-
"""Where will the feet land with the *new* damping?  (no Blender needed)

The build applies one transform per source bone and skins the source mesh with
the source weights, so the geometry a given rebuild produces is fully
determined by (source vertices, source weights, new per-bone targets, the
per-bone rotation).  That means the fix can be checked *before* spending a
Blender round trip on it -- and checked exactly, because it is the same
arithmetic the pipeline performs.

Reports, per bone: the mean X of the vertices that bone drives, in the same
terms as `diag_weights.py` did for the old build:

  * feet must end up well apart, not on one line (the catwalk report)
  * each foot must stay on its own side of x = 0 (no crossing)
  * every joint should stay close to the geometry it drives, or animation
    swings that geometry about the wrong pivot

    python tools/check_damp.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools"))
from pmx import read_pmx                                     # noqa: E402
from ganyu_src import SRC_PMX                               # noqa: E402
from ganyu_mmd_to_x4 import MmdAdapter, build_bone_map, mmd_to_blender  # noqa: E402
from retarget_core import BindPoseRetarget, kabsch           # noqa: E402

LEG = ('Bip01 L Thigh', 'Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L Toe0',
       'Bip01 R Thigh', 'Bip01 R Calf', 'Bip01 R Foot', 'Bip01 R Toe0')


def main():
    m = read_pmx(SRC_PMX)
    bones = m['bones']
    pos = {b['name']: b['position'] for b in bones}
    par = {b['name']: (bones[b['parent']]['name'] if b['parent'] >= 0 else None)
           for b in bones}
    ad = MmdAdapter(build_bone_map(bones))

    vd = np.load(os.path.join(HERE, '..', 'work', 'b_vanilla_body.npz'),
                 allow_pickle=True)
    names = [str(x) for x in vd['bone_names']]
    x4 = {n: {'head': vd['bone_heads'][i], 'tail': vd['bone_tails'][i]}
          for i, n in enumerate(names)}
    t = BindPoseRetarget(pos, par, x4, verbose=False, adapter=ad)
    # the source's bone positions in the fitted frame -- `delta` stores its
    # (src, dst) in this space, so the source mesh has to be lifted into it
    # before the per-bone transforms can be applied
    sfit = {b: t.src[b] for b in t.src}

    # skin the source mesh with the source weights: vertex -> sum w * (R (v - s) + d)
    V = np.array(m['positions'], float)
    W = m['weights']
    name_of = {i: b['name'] for i, b in enumerate(bones)}
    # transform of each X4 bone, keyed by the *source* bone that drives it
    tr = {}
    for sb, (src, dst, R) in t.delta.items():
        tr[sb] = (np.asarray(src, float), np.asarray(dst, float), R)
    src_of_x4 = {ad.map_bone(sb): sb for sb in t.direct}

    out = np.zeros_like(V)
    for i, ws in enumerate(W):
        v = V[i]
        acc = np.zeros(3)
        for bi, w in ws:
            bn = name_of[bi]
            rec = tr.get(bn)
            if rec is None:
                acc += w * v
                continue
            s, d, R = rec
            acc += w * (R @ (v - sfit[bn]) + d)
        out[i] = acc

    print('== 重建后每根腿骨的几何位置（x=0 是中线，+x 是角色左侧）==')
    report = {}
    for B in LEG:
        sb = src_of_x4.get(B)
        if sb is None:
            continue
        bi = bones.index([b for b in bones if b['name'] == sb][0])
        idx = [i for i, ws in enumerate(W)
               if any(j == bi and w > 0.5 for j, w in ws)]
        if not idx:
            continue
        c = out[idx].mean(axis=0)
        bone = np.asarray(t.delta[B][1], float)
        report[B] = c
        print('   %-16s 骨 x=%+7.2f   几何中心 x=%+7.2f  (%4d 顶点)  偏差 %4.2f cm'
              % (B, bone[0], c[0], len(idx),
                 float(np.hypot(*(bone[:2] - c[:2])))))

    print()
    for side, sign, zh in (('左', +1, 'L'), ('右', -1, 'R')):
        f = report.get('Bip01 %s Foot' % zh)
        t2 = report.get('Bip01 %s Toe0' % zh)
        if f is None:
            continue
        ok = '✓' if (f[0] * sign > 0 and t2[0] * sign > 0) else '✗ 越到对侧'
        print('   %s脚: 脚 %+6.2f  趾 %+6.2f   %s' % (side, f[0], t2[0], ok))
    dist = float(np.linalg.norm(report['Bip01 L Foot'][:2]
                                - report['Bip01 R Foot'][:2]))
    print()
    print('   两脚间距（几何中心）: %.2f cm' % dist)
    print('   参照: 修复前 18.0 / 源码几何 13.6 / vanilla 39.8')
    return 0


if __name__ == '__main__':
    sys.exit(main())
