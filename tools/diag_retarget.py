# -*- coding: utf-8 -*-
"""Numeric validation of the retarget, without Blender.

Runs the same code path `build_ganyu_x4.py` uses, then reports the things a
render would only show qualitatively:

  * bone axis errors -- an upper arm that fits >60 degrees means the twist-bone
    short circuit (`_pair_axis`) has fired and the limb will bend sideways
  * how far each bone's vertices travel, in cm, and which bone moves most
  * the stance width, which sets `LATERAL_DAMP`
  * unweighted vertices and vertex budget per region

Everything here is measurable before a single DDS is encoded, which is the
point: the previous project spent rounds looking at renders for problems this
would have named outright.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools"))
from pmx import read_pmx                                  # noqa: E402
from ganyu_src import SRC_PMX                            # noqa: E402
from ganyu_mmd_to_x4 import (MmdAdapter, build_bone_map, report_map,  # noqa: E402
                       check_core, CORE, fix_neck_source_weights,
                       blend_chain_offsets, LIMB_CHAINS)
from retarget_core import BindPoseRetarget                # noqa: E402

X4_DUMP = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "work",
                       "vanilla", "x4_skeleton_dump.json")


def load_x4(part='head'):
    d = json.load(open(X4_DUMP, encoding='utf-8'))
    return {n: {'head': tuple(b['head']), 'tail': tuple(b['tail']),
                'parent': b['parent']} for n, b in d[part]['bones'].items()}


def main():
    x4 = load_x4('head')
    m = read_pmx(SRC_PMX)
    bones = m['bones']
    check_core(bones)
    src_pos = {b['name']: b['position'] for b in bones}
    src_par = {b['name']: (bones[b['parent']]['name'] if b['parent'] >= 0
                           else None) for b in bones}
    bmap = build_bone_map(bones)
    adapter = MmdAdapter(bmap)
    weighted = sorted({bones[bi]['name'] for w in m['weights'] for bi, _ in w})
    report_map(bones, bmap, set(x4), weighted)

    tr = BindPoseRetarget(src_pos, src_par, x4, adapter=adapter)

    V = np.array(m['positions'], float)
    fix_neck_source_weights(m, tr, V)
    idx_bone = [b['name'] for b in bones]
    Vx, n_unw, n_targets = tr.transform(V, m['weights'], idx_bone)
    print('\ntransform: %d verts, %d unweighted, %d target bones'
          % (len(Vx), n_unw, n_targets))
    if not np.isfinite(Vx).all():
        raise SystemExit('!! non-finite vertices produced')

    # ------------------------------------------------------------------ axes
    print('\n-- bone axis fit (destructive if > ~60 deg) --')
    rows = []
    for name in sorted(tr.delta):
        rec = tr.delta[name]
        src, dst, R = rec
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        rows.append((ang, name))
    for ang, name in sorted(rows, reverse=True)[:14]:
        print('  %-24s %6.1f deg' % (name, ang))

    # --------------------------------------------------------------- travel
    print('\n-- vertices per X4 bone and their mean travel (cm) --')
    merged = tr.merge_weights(m['weights'], idx_bone)
    G = tr.global_only(V)
    from collections import defaultdict
    cnt = defaultdict(int)
    trav = defaultdict(float)
    for i, d in enumerate(merged):
        for B in d:
            cnt[B] += 1
            rec = tr.delta.get(B)
            if rec:
                trav[B] += float(np.linalg.norm(Vx[i] - G[i]))
    print('  %-24s %7s %9s' % ('x4 bone', 'verts', 'travel'))
    for B in sorted(cnt, key=lambda b: -cnt[b])[:18]:
        print('  %-24s %7d %8.1f' % (B, cnt[B], trav[B] / max(1, cnt[B])))

    # -------------------------------------------------------------- stance
    def bone_at(name):
        return np.array(x4[name]['head'])

    print('\n-- stance / proportion check (X4 bind pose vs retargeted) --')
    print('  %-26s %10s %10s' % ('', 'X4 bind', 'ours'))
    for label, a, b in (('hip width', 'Bip01 L Thigh', 'Bip01 R Thigh'),
                        ('ankle width', 'Bip01 L Foot', 'Bip01 R Foot'),
                        ('shoulder width', 'Bip01 L UpperArm',
                         'Bip01 R UpperArm')):
        if a in x4 and b in x4:
            print('  %-26s %9.1f %10.1f cm'
                  % (label, abs(bone_at(a)[0] - bone_at(b)[0]), float('nan')))
    # measure ours from the *geometry*: vertices weighted to the foot bones
    def geom_width(bone_names, axis=0):
        sel = np.zeros(len(Vx), bool)
        for i, d in enumerate(merged):
            if any(B in bone_names for B in d):
                sel[i] = True
        if not sel.any():
            return float('nan')
        return float(Vx[sel][:, axis].max() - Vx[sel][:, axis].min())

    print('  geometry ankle span        %9s %10.1f cm'
          % ('-', geom_width({'Bip01 L Foot', 'Bip01 R Foot'})))
    print('  geometry head span         %9s %10.1f cm'
          % ('-', geom_width({'Bip01 Head'})))

    # ------------------------------------------------------------------ bbox
    print('\n-- bbox --')
    print('  ours : X[%7.1f %7.1f]  Y[%7.1f %7.1f]  Z[%7.1f %7.1f]'
          % (Vx[:, 0].min(), Vx[:, 0].max(), Vx[:, 1].min(), Vx[:, 1].max(),
             Vx[:, 2].min(), Vx[:, 2].max()))
    print('  X4 head bone z = %.1f, neck z = %.1f'
          % (x4['Bip01 Head']['head'][2], x4['Bip01 Neck']['head'][2]))

    # feet on the floor?
    from collections import defaultdict as dd
    foot = np.zeros(len(Vx), bool)
    for i, d in enumerate(merged):
        if any(B.endswith(' Foot') or 'Toe' in B for B in d):
            foot[i] = True
    if foot.any():
        print('  lowest foot vertex z = %.2f cm (vanilla sole -0.32)'
              % Vx[foot][:, 2].min())
    print('  lowest vertex overall z = %.2f cm' % Vx[:, 2].min())

    out = os.path.join(os.path.dirname(HERE), 'work', 'retarget_check.npz')
    np.savez_compressed(out, Vx=Vx, V=V)
    print('\nsaved %s' % out)


if __name__ == '__main__':
    main()
