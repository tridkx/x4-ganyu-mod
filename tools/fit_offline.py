# -*- coding: utf-8 -*-
"""Run the global Kabsch frame fit without Blender, and report residuals.

Blender is only needed to *build*; the fit itself is pure numpy, so keeping it
outside the DCC makes the mapping table testable in seconds instead of minutes.

The X4 side comes from `work/vanilla/x4_skeleton_dump.json`, which is the same
armature `build_ganyu_x4.py` imports (`char_arg_f_dyn_blend_head.xac`) -- if
these two ever disagree the numbers here are meaningless, so the bone count is
printed and compared.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# the retarget core is shared verbatim with the two earlier X4 projects: it is
# source-agnostic and already carries the fixes this rig needs (axis-chain
# walking, chain-root folding, feet sharing one offset)
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools"))
from pmx import read_pmx                                  # noqa: E402
from ganyu_src import SRC_PMX                            # noqa: E402
from ganyu_mmd_to_x4 import MmdAdapter, build_bone_map, CORE, ALIGN_PAIRS  # noqa: E402

WORK = os.path.join(os.path.dirname(HERE), 'work')
X4_DUMP = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "work",
                       "vanilla", "x4_skeleton_dump.json")


def load_x4(part='head'):
    d = json.load(open(X4_DUMP, encoding='utf-8'))
    bones = {}
    for name, b in d[part]['bones'].items():
        bones[name] = {'head': tuple(b['head']), 'tail': tuple(b['tail']),
                       'parent': b['parent']}
    return bones


def main():
    m = read_pmx(SRC_PMX)
    bones = m['bones']
    src_pos = {b['name']: b['position'] for b in bones}
    src_par = {b['name']: (bones[b['parent']]['name'] if b['parent'] >= 0
                           else None) for b in bones}
    x4 = load_x4('head')
    print('X4 armature (%s): %d bones' % ('dyn_blend_head', len(x4)))
    missing = [b for _, b in ALIGN_PAIRS if b not in x4]
    if missing:
        raise SystemExit('!! align targets missing from the X4 dump: %s'
                         % missing)

    # the X4 dump stores the reference rig; confirm the core names all resolve
    print('core map: %d direct bones' % len(CORE))
    for src, dst in sorted(CORE.items()):
        if dst not in x4:
            print('   !! %-12s -> %-22s NOT IN X4' % (src, dst))

    from retarget_core import kabsch
    P, Q = [], []
    for rb, xb in ALIGN_PAIRS:
        if rb in src_pos and xb in x4:
            P.append(np.array([src_pos[rb][0], -src_pos[rb][2], src_pos[rb][1]]))
            Q.append(np.array(x4[xb]['head']))
    P, Q = np.array(P), np.array(Q)
    scale, R, t = kabsch(P, Q)
    pred = (scale * (R @ P.T)).T + t
    res = np.linalg.norm(pred - Q, axis=1)
    print('\nglobal fit: scale = %.4f cm/unit   det(R) = %+.4f' %
          (scale, float(np.linalg.det(R))))
    print('  (previous project: 9.625)')
    print('\n%-12s %-22s %8s' % ('source', 'x4 target', 'residual'))
    for (rb, xb), d in sorted(zip(ALIGN_PAIRS, res), key=lambda z: -z[1]):
        print('  %-12s -> %-22s %6.1f cm' % (rb, xb, d))
    print('  mean %.1f cm   max %.1f cm' % (res.mean(), res.max()))

    # ------------------------------------------------------- per-bone travel
    # what each mapped bone's vertices actually move when the chain is
    # retargeted numerically (no local rotation solved here -- this is the pure
    # translation of the direct-match step, which is what "travel" measured in
    # the previous project)
    print('\n-- where the mapped bones land (translation only) --')
    print('%-12s %-22s %9s %9s %8s' % ('source', 'x4', 'src z', 'dst z', 'travel'))
    bmap = build_bone_map(bones)
    rows = []
    for rb, xb in CORE.items():
        if rb not in src_pos or xb not in x4:
            continue
        sp = np.array([src_pos[rb][0], -src_pos[rb][2], src_pos[rb][1]]) * scale
        # rotate/translate using the fitted frame
        sp = scale * (R @ np.array([src_pos[rb][0], -src_pos[rb][2],
                                    src_pos[rb][1]])) + t
        dp = np.array(x4[xb]['head'])
        rows.append((float(np.linalg.norm(sp - dp)), rb, xb, sp[2], dp[2]))
    for d, rb, xb, sz, dz in sorted(rows, reverse=True)[:18]:
        print('  %-12s -> %-22s %8.1f %9.1f %7.1f' % (rb, xb, sz, dz, d))

    # ---------------------------------------------------- weight distribution
    from collections import Counter
    c = Counter()
    for w in m['weights']:
        for bi, wv in w:
            c[bones[bi]['name']] += 1
    weighted = sorted(c, key=lambda n: -c[n])
    direct = [b for b in weighted if b in CORE]
    folded = [b for b in weighted if b not in CORE and bmap.get(b) in x4]
    lost = [b for b in weighted if not bmap.get(b) or bmap[b] not in x4]
    print('\nweighted bones: %d -> %d direct, %d folded, %d unmapped'
          % (len(weighted), len(direct), len(folded), len(lost)))
    if lost:
        print('  !! UNMAPPED (vertices would fall back to global frame):')
        for n in lost[:30]:
            print('     %-16s %6d verts' % (n, c[n]))
    print('\n  top folded bones (chain -> landing bone):')
    for n in [b for b in weighted if b not in CORE][:16]:
        print('     %-16s %6d verts -> %s' % (n, c[n], bmap.get(n)))


if __name__ == '__main__':
    main()
