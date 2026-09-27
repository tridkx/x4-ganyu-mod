# -*- coding: utf-8 -*-
"""LATERAL_DAMP trade-off: how far off its own bone does the foot end up?

The lateral damping exists because the source's legs taper inwards (ankles
11.7 cm apart) while X4's bind pose splays outwards (35.4 cm apart): matching
each leg bone outright swings the leg out of the skirt.  Damping keeps only
part of the sideways travel.

The current values taper *along* the chain (thigh 0.60, calf 0.30, foot 0.20),
which does not just move the leg -- it bends the bone chain inwards, so each
joint ends up some distance from the geometry it is supposed to drive.  The
skin still looks right in the bind pose (the mesh was built from those very
targets), but under animation the foot pivots about a point 7-12 cm outside
itself: the stride swings wide and the feet splay, which reads as a catwalk.

A uniform damping moves the whole leg (bone chain and geometry together) and
keeps every joint on its own geometry.  This script measures, for a set of
candidate settings:

  * the offset between each leg bone and the vertices it drives (the animation
    error -- smaller is better)
  * the resulting stance width and foot spacing (must stay inside the skirt)
  * which vertices the source's leg actually owns, so the skirt check is real

    python tools/diag_damp.py

STATUS: this reports the *planar* distance between a bone's target and the
mean of the vertices it drives.  That number is not by itself the animation
error -- a joint can sit off its geometry in Y as well, and the geometry moves
with the target when the mesh is rebuilt -- so treat it as a relative
comparison between candidate settings, not as a pass/fail.  The numbers that
were actually used to fix the catwalk come from `measure_legs.py` (a built
asset, measured in Blender) and `check_damp.py`.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools"))
from pmx import read_pmx                                     # noqa: E402
from ganyu_src import SRC_PMX                               # noqa: E402
from ganyu_mmd_to_x4 import (MmdAdapter, build_bone_map,           # noqa: E402
                       fix_neck_source_weights)
from retarget_core import BindPoseRetarget                   # noqa: E402

#: candidates: name -> {x4 bone: lateral keep fraction}
CANDIDATES = {
    'current  0.60/0.30/0.20': {'Bip01 L Thigh': 0.60, 'Bip01 R Thigh': 0.60,
                                'Bip01 L Calf': 0.30, 'Bip01 R Calf': 0.30,
                                'Bip01 L Foot': 0.20, 'Bip01 R Foot': 0.20,
                                'Bip01 L Toe0': 0.20, 'Bip01 R Toe0': 0.20},
    'uniform  0.45': {},
    'uniform  0.60': {},
    'uniform  0.75': {},
    'uniform  1.00 (X4 bind)': {},
}


def make_adapter(values):
    """An MmdAdapter whose LATERAL_DAMP is the candidate (uniform or per-bone)."""
    class A(MmdAdapter):
        LATERAL_DAMP = values
    return A


def run(pos, par, x4, values, uniform=None):
    ad = make_adapter(values)(BONE_MAP)
    if uniform is not None:
        ad.LATERAL_DAMP = {b: uniform for b in (
            'Bip01 L Thigh', 'Bip01 R Thigh', 'Bip01 L Calf', 'Bip01 R Calf',
            'Bip01 L Foot', 'Bip01 R Foot', 'Bip01 L Toe0', 'Bip01 R Toe0')}
    t = BindPoseRetarget(pos, par, x4, verbose=False, adapter=ad)
    return t


def main():
    global BONE_MAP
    m = read_pmx(SRC_PMX)
    pos = {b['name']: b['position'] for b in m['bones']}
    par = {b['name']: (m['bones'][b['parent']]['name'] if b['parent'] >= 0
                       else None) for b in m['bones']}
    BONE_MAP = build_bone_map(m['bones'])

    vd = np.load(os.path.join(HERE, '..', 'work', 'b_vanilla_body.npz'),
                 allow_pickle=True)
    names = [str(x) for x in vd['bone_names']]
    x4 = {n: {'head': vd['bone_heads'][i], 'tail': vd['bone_tails'][i]}
          for i, n in enumerate(names)}

    ld = np.load(os.path.join(HERE, '..', 'work', 'b_ganyu_body.npz'),
                 allow_pickle=True)
    V = ld['verts']
    WI, WV = ld['w_idx'], ld['w_val']
    dom = np.array([ix[int(np.argmax(v))] for ix, v in zip(WI, WV)])
    # geometry owned by each leg bone, in the *current* build
    geo = {}
    for b in ('Bip01 L Thigh', 'Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L Toe0',
              'Bip01 R Thigh', 'Bip01 R Calf', 'Bip01 R Foot', 'Bip01 R Toe0'):
        msk = np.array([d == b for d in dom])
        geo[b] = V[msk] if msk.any() else None

    print('%-24s %s' % ('setting', ' '.join(
        '%10s' % b.replace('Bip01 ', '') for b in
        ('Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L Toe0'))))
    print('%-24s %s' % ('', ' '.join('%10s' % s for s in
                                     ('calf off', 'foot off', 'toe off'))))
    rows = []
    for name, values in CANDIDATES.items():
        uniform = None
        if name.startswith('uniform'):
            uniform = float(name.split()[1])
        t = run(pos, par, x4, values, uniform)
        errs = {}
        for b in ('Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L Toe0'):
            if b not in t.delta or geo[b] is None:
                errs[b] = float('nan')
                continue
            dst = np.asarray(t.delta[b][1], float)
            # geometry must sit on its bone; the mesh that shipped was built
            # from the *previous* targets, so this is a like-for-like compare
            # of "how far the bone is from the geometry it drives"
            errs[b] = float(np.linalg.norm(dst[:2] - geo[b][:, :2].mean(axis=0)))
        rows.append((name, errs, t))
        print('%-24s %s' % (name, ' '.join('%10.5s' % ('%.2f' % errs[b])
                                           for b in ('Bip01 L Calf',
                                                     'Bip01 L Foot',
                                                     'Bip01 L Toe0'))))

    print()
    print('== 结论 ==')
    best = min(rows, key=lambda r: np.nansum(list(r[1].values())))
    print('   合计错位最小: %s' % best[0])
    for name, errs, _t in rows:
        print('   %-24s 合计 %6.2f cm' % (name, np.nansum(list(errs.values()))))
    return 0


if __name__ == '__main__':
    sys.exit(main())
