# -*- coding: utf-8 -*-
"""Which side is each foot's geometry actually bound to?  (catwalk check)

The bind pose can look perfect while the *weights* put the left shoe on the
right leg: the geometry would then be swept across the midline the moment an
animation plays, and the idle stance is the one frame where it is invisible.

This reads the dumped mesh (`work/b_<asset>.npz`: verts, weights, part tags)
and, for every leg bone, reports the mean X of the vertices that bone drives.
X4 axes: **+X is the character's right**, so a bone named `L` must own
geometry at negative X (and vice versa).  A sign flip here is the catwalk.

    python tools/diag_weights.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(os.path.dirname(HERE), 'work')

PAIRS = [
    ('ganyu body', os.path.join(WORK, 'b_ganyu_body.npz')),
    ('vanilla body', os.path.join(WORK, 'b_vanilla_body.npz')),
    ('ganyu head', os.path.join(WORK, 'b_ganyu_head.npz')),
    ('vanilla head', os.path.join(WORK, 'b_vanilla_head.npz')),
]

#: bones worth checking: (bone, expected side, human label)
LEG_BONES = [
    ('Bip01 L Thigh', -1, 'left thigh'),
    ('Bip01 R Thigh', +1, 'right thigh'),
    ('Bip01 L Calf', -1, 'left calf'),
    ('Bip01 R Calf', +1, 'right calf'),
    ('Bip01 L Foot', -1, 'left foot'),
    ('Bip01 R Foot', +1, 'right foot'),
    ('Bip01 L Toe0', -1, 'left toe'),
    ('Bip01 R Toe0', +1, 'right toe'),
]


def load(path):
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    return {
        'names': [str(x) for x in d['bone_names']],
        'verts': d['verts'],
        'w_idx': d['w_idx'],
        'w_val': d['w_val'],
        'parts': [str(x) for x in d['part_names']],
        'v_part': d['v_part'],
    }


def main():
    for label, path in PAIRS:
        a = load(path)
        if a is None:
            print('-- %s: missing (%s)' % (label, path))
            continue
        V, WI, WV = a['verts'], a['w_idx'], a['w_val']
        idx = {n: i for i, n in enumerate(a['names'])}
        print('-- %s: %d verts, %d bones' % (label, len(V), len(a['names'])))
        # dominant bone per vertex (w_idx/w_val are per-vertex lists)
        dom = np.array([ix[int(np.argmax(v))]
                        for ix, v in zip(WI, WV)], dtype=object)
        for bone, want, human in LEG_BONES:
            if bone not in idx:
                print('   %-18s missing' % bone)
                continue
            bi = str(bone)
            m = np.array([d == bi for d in dom])
            n = int(m.sum())
            if not n:
                print('   %-18s drives 0 vertices' % bone)
                continue
            x = V[m][:, 0]
            mean = float(x.mean())
            lo, hi = float(x.min()), float(x.max())
            side = np.sign(mean)
            flag = ''
            if side != 0 and side != want:
                flag = '   <-- WRONG SIDE (bone is %s)' % human
            print('   %-18s %5d verts   x mean %+7.2f  range [%+7.2f, %+7.2f]%s'
                  % (bone, n, mean, lo, hi, flag))
        # and the raw geometry: where is the left-most and right-most geometry
        print('   geometry: x range [%+.2f, %+.2f]' % (V[:, 0].min(), V[:, 0].max()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
