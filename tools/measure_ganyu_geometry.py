# -*- coding: utf-8 -*-
"""Measure the numbers the retarget parameters depend on, before building.

    python tools/measure_ganyu_geometry.py

Three questions, all answered from the *source* rig plus the X4 bind pose --
no Blender needed, so the numbers can be re-taken in seconds:

1. **How far inwards are the legs?**  `LATERAL_DAMP` decides how much of the
   sideways travel to keep.  Get it wrong and the character walks on one line
   (the "catwalk"); the criterion is the **foot spacing measured on the
   geometry**, compared with vanilla's 39.8 cm.  Both the source bone chain
   and the geometry each bone actually drives are printed, because damping
   that bends the chain leaves the joint inside the mesh it carries.
2. **Where is the neck?**  `NECK_FIX_Z` rebinds the lower neck from the chest
   bone to the neck bone *before* retargeting.  It has to bracket the band
   where `上半身2` and `首` blend, in retargeted centimetres.
3. **Where does the sole sit?**  `GROUND_Z` lifts the feet onto the floor.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'x4-character-retarget', 'tools'))

from pmx import read_pmx                                     # noqa: E402
from ganyu_src import SRC_PMX                                # noqa: E402
from ganyu_mmd_to_x4 import (MmdAdapter, build_bone_map,     # noqa: E402
                             check_core)
from retarget_core import BindPoseRetarget                   # noqa: E402
from fit_offline import load_x4                              # noqa: E402

LEG_X4 = ['Bip01 L Thigh', 'Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L Toe0',
          'Bip01 R Thigh', 'Bip01 R Calf', 'Bip01 R Foot', 'Bip01 R Toe0']


def main():
    m = read_pmx(SRC_PMX)
    bones = m['bones']
    check_core(bones)
    src_pos = {b['name']: b['position'] for b in bones}
    src_par = {b['name']: (bones[b['parent']]['name'] if b['parent'] >= 0
                           else None) for b in bones}
    x4 = load_x4('head')
    adapter = MmdAdapter(build_bone_map(bones))
    tr = BindPoseRetarget(src_pos, src_par, x4, adapter=adapter, verbose=False)

    V = np.array(m['positions'], float)
    idx_bone = [b['name'] for b in bones]
    G = tr.global_only(V)                     # after the global frame only
    Vx = tr.transform(V, m['weights'], idx_bone)[0]
    W = tr.merge_weights(m['weights'], idx_bone)

    print('\n== leg bones: source fit / X4 bind / geometry each bone drives ==')
    print('(+X is the character\'s left; all values cm)')
    print('%-18s %9s %9s %9s %9s' % ('x4 bone', 'src fit', 'x4 bind',
                                     'geom fit', 'geom retgt'))
    for b in LEG_X4:
        own = [i for i, w in enumerate(W) if w.get(b, 0.0) > 0.5]
        src_x = tr.src_signed_x(b) if hasattr(tr, 'src_signed_x') else None
        print('%-18s %9s %9.2f %9s %9s' % (
            b, '-', float(x4[b]['head'][0]),
            ('%9.2f' % np.mean(G[own][:, 0])) if own else '-',
            ('%9.2f' % np.mean(Vx[own][:, 0])) if own else '-'))

    print('\n-- what the damped targets would be --')
    print('%-18s %9s %9s %9s' % ('x4 bone', 'src fit x', 'own fit x',
                                 'damped x'))
    for b in LEG_X4:
        own = [i for i, w in enumerate(W) if w.get(b, 0.0) > 0.5]
        srcb = [n for n in adapter.bone_map if adapter.bone_map[n] == b]
        sx = np.mean([tr.src[n][0] for n in srcb]) if srcb else float('nan')
        ownx = float(np.mean(G[own][:, 0])) if own else float('nan')
        tgt = adapter.adjust_target(b, np.array([sx, 0, 0]),
                                   np.array(x4[b]['head'], float))
        print('%-18s %9.2f %9.2f %9.2f' % (b, sx, ownx, tgt[0]))

    def gap(VV):
        l = [i for i, w in enumerate(W) if w.get('Bip01 L Foot', 0) > 0.5]
        r = [i for i, w in enumerate(W) if w.get('Bip01 R Foot', 0) > 0.5]
        if not l or not r:
            return float('nan')
        return float(np.mean(VV[l][:, 0]) - np.mean(VV[r][:, 0]))

    print('\nfoot spacing (geometry): source fit %.1f cm -> retargeted %.1f cm'
          % (gap(G), gap(Vx)))
    print('   X4 bind bone spacing  : %.1f cm  (vanilla women measure 39.8 in '
          'game)' % (x4['Bip01 L Foot']['head'][0]
                     - x4['Bip01 R Foot']['head'][0]))

    # ------------------------------------------------- fitted rotations
    # The neck fix exists because two bones that blend across a joint can
    # carry very different fit rotations: geometry bound to both is then swept
    # sideways by the difference.  Print the angles so the decision is data.
    print('\n== fitted rotation per core bone (the ones that disagree most) ==')
    rows = []
    for src, dst in adapter.bone_map.items():
        d = tr.delta.get(dst)
        if not d or src not in adapter.bone_map:
            continue
        R = d[2]
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        rows.append((ang, src, dst, d[1] - d[0]))
    seen = set()
    uniq = []
    for ang, src, dst, off in sorted(rows, reverse=True):
        if dst in seen:
            continue
        seen.add(dst)
        uniq.append((ang, src, dst, off))
    for ang, src, dst, off in uniq[:12]:
        print('  %-10s -> %-20s %6.1f deg   offset (%5.1f,%5.1f,%5.1f) cm'
              % (src, dst, ang, off[0], off[1], off[2]))

    # ------------------------------------------------------------- the neck
    print('\n== neck ==')
    idx = {b['name']: i for i, b in enumerate(bones)}
    i_spine, i_neck, i_head = idx['上半身2'], idx['首'], idx['頭']
    zs_spine, zs_neck = [], []
    for vi, wl in enumerate(m['weights']):
        d = {bi: wv for bi, wv in wl}
        if d.get(i_spine, 0) > 0.05:
            zs_spine.append((float(G[vi][2]), float(d[i_spine])))
        if d.get(i_neck, 0) > 0.05 or d.get(i_head, 0) > 0.05:
            zs_neck.append((float(G[vi][2]), float(d.get(i_neck, 0))))
    if zs_spine:
        z = np.array([a for a, _ in zs_spine])
        print('  vertices with 上半身2 weight: z %.1f .. %.1f cm (n=%d)'
              % (z.min(), z.max(), len(z)))
    print('  首   bone retargeted z: %.1f cm' % tr.src['首'][2])
    print('  頭   bone retargeted z: %.1f cm' % tr.src['頭'][2])
    print('  上半身2 retargeted z: %.1f cm' % tr.src['上半身2'][2])
    near = [vi for vi, wl in enumerate(m['weights'])
            if abs(float(G[vi][2]) - tr.src['首'][2]) < 12
            and abs(float(G[vi][0])) < 7]
    if near:
        print('  within +-12 cm of the neck bone and |x|<7: %d verts, '
              'z %.1f .. %.1f' % (len(near), min(float(G[i][2]) for i in near),
                                  max(float(G[i][2]) for i in near)))

    # -------------------------------------------------------------- the feet
    print('\n== feet ==')
    foot = [i for i, w in enumerate(W)
            if sum(v for k, v in w.items()
                   if k.endswith(' Foot') or 'Toe' in k) > 0.5]
    if foot:
        lo = float(Vx[foot][:, 2].min())
        print('  sole after retarget: z = %+.2f cm  (GROUND_Z lifts it to the '
              'vanilla sneaker sole at -0.32)' % lo)
        print('  lift that would be applied: %+.2f cm'
              % (-0.5 - lo) if abs(-0.5 - lo) > 0.05 else '  already on the floor')

    # ------------------------------------------------------------ dimensions
    print('\n== dimensions ==')
    print('  source bbox (units): %s' % np.round(
        np.array([V.min(0), V.max(0)]), 2).tolist())
    print('  height after retarget: %.1f cm (crown %.1f, sole %.1f)'
          % (Vx[:, 2].max() - Vx[:, 2].min(), Vx[:, 2].max(), Vx[:, 2].min()))


if __name__ == '__main__':
    main()
