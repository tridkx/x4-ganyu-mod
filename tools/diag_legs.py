# -*- coding: utf-8 -*-
"""Are the legs' local frames mirrored left/right?  (symptom: catwalk)

X4's 1100+ shared animations are authored against the *vanilla* Biped local
frames, and a Biped rig mirrors its left and right limbs: `Bip01 L Thigh` and
`Bip01 R Thigh` carry the same bone, reflected across the YZ plane, and the
animation therefore rotates them with opposite-signed values to swing the legs
apart.

A retarget that rebuilds each bone's local rotation from its own source bone
can break that reflection without moving a single vertex in the bind pose: the
model still stands correctly, but the moment an animation plays, both legs
rotate the same way -- one swings out and the other crosses the midline, which
is exactly the "catwalk" report (left foot landing where the right should).

So: load every .xac we have (vanilla, ours, the other projects'), take the
leg chain's **local** frames, and compare L against R after reflecting.

For a mirrored pair about the YZ plane the local frames satisfy

    W_R = M @ W_L @ M          M = diag(-1, 1, 1)

which, for the Y axis (the bone's own direction), means

    y_R = (-y_L.x, y_L.y, y_L.z)

and for the X and Z axes the same with a sign flip as well.  What matters in
practice is the *bone direction*: if both thighs point the same way in X, the
legs swing together.

    python tools/diag_legs.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from xac_skeleton import Skeleton                         # noqa: E402

WS = os.path.dirname(os.path.dirname(HERE))               # workspace root

#: (label, path) -- vanilla first: it defines what "mirrored" means here
CANDIDATES = [
    ('vanilla head', os.path.join(
        WS, 'shared', 'x4root', 'assets', 'characters', 'argon', 'heads',
        'char_arg_f_dyn_blend_head.xac')),
    ('vanilla body', os.path.join(
        WS, 'shared', 'x4root', 'assets', 'characters', 'argon', 'bodies',
        'char_arg_f_sweater_leggings_civ_01.xac')),
    ('ganyu body', os.path.join(
        HERE, '..', 'work', 'x4cc_pkg', 'ganyu_body', 'assets',
        'characters', 'mycharacters', 'bodies', 'ganyu_body.xac')),
    ('emilie body', os.path.join(
        WS, 'emilie', 'work', 'x4_emilie_mod', 'assets', 'characters',
        'argon', 'emilie', 'bodies', 'emilie_body.xac')),
    ('boru body', os.path.join(
        WS, 'boru', 'work', 'x4_boru_argon_add', 'assets', 'characters',
        'argon', 'boru', 'bodies', 'boru_body.xac')),
]

#: the chains to inspect: (label, left bone, right bone)
LEG_CHAIN = [
    ('Thigh', 'Bip01 L Thigh', 'Bip01 R Thigh'),
    ('Calf', 'Bip01 L Calf', 'Bip01 R Calf'),
    ('Foot', 'Bip01 L Foot', 'Bip01 R Foot'),
    ('Toe0', 'Bip01 L Toe0', 'Bip01 R Toe0'),
]

MIRROR = np.diag([-1.0, 1.0, 1.0])


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def check(path, label):
    if not os.path.exists(path):
        print('-- %s: MISSING (%s)' % (label, path))
        return None
    sk = Skeleton(path)
    print('-- %s' % label)
    rows = []
    for name, lb, rb in LEG_CHAIN:
        if lb not in sk.by_name or rb not in sk.by_name:
            print('   %-6s missing bone' % name)
            continue
        hL, hR = sk.head(lb), sk.head(rb)
        # local frame axes (columns of the local rotation), X4 world axes
        wL = sk.world[sk.by_name[lb][0]][:3, :3]
        wR = sk.world[sk.by_name[rb][0]][:3, :3]
        # bone direction = +Y of the local frame
        # A mirrored pair satisfies W_R = M @ W_L @ M with M = diag(-1,1,1),
        # i.e. every basis vector is reflected across the YZ plane: the
        # component of each column along X flips, Y and Z stay.  (Flipping Z
        # as well would turn the reflection into a rotation -- the first
        # version of this check did that and reported a 171 degree error for
        # every asset, vanilla included.)
        wantR = wL * np.array([[-1.0], [1.0], [1.0]])
        # per-axis error: angle between the actual right frame and the mirror
        axis_err = []
        for k in range(3):
            a = unit(wR[:, k])
            b = unit(wantR[:, k])
            axis_err.append(float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1)))))
        yL, yR = unit(wL[:, 1]), unit(wR[:, 1])
        same_x = np.sign(yL[0]) == np.sign(yR[0])
        rows.append((name, hL, hR, yL, yR, max(axis_err), same_x))
        print('   %-6s head L(%7.2f,%7.2f,%7.2f) R(%7.2f,%7.2f,%7.2f)'
              % (name, hL[0], hL[1], hL[2], hR[0], hR[1], hR[2]))
        print('          dir L(%+.2f,%+.2f,%+.2f) R(%+.2f,%+.2f,%+.2f)'
              '   mirror err %5.1f deg%s'
              % (yL[0], yL[1], yL[2], yR[0], yR[1], yR[2], max(axis_err),
                 '   <-- SAME SIDE' if same_x else ''))
        print('          axis err  X %5.1f  Y %5.1f  Z %5.1f deg'
              % (axis_err[0], axis_err[1], axis_err[2]))
    return sk, rows


def main():
    results = {}
    for label, path in CANDIDATES:
        r = check(os.path.normpath(path), label)
        if r:
            results[label] = r
    print()
    print('== verdict (worst per-axis mirror error; >5 deg = frame is not'
          ' the mirror of its twin) ==')
    for label, (_sk, rows) in results.items():
        worst = max(r[5] for r in rows) if rows else 0.0
        sames = [r[0] for r in rows if r[6]]
        print('   %-14s worst mirror error %5.1f deg%s'
              % (label, worst,
                 '   same-side bones: %s' % ','.join(sames) if sames else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
