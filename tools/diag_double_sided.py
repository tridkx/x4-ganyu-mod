# -*- coding: utf-8 -*-
"""Is the geometry already double-sided, or does the material flag have to be?

Every material in this PMX sets flag bit 0 ("draw both sides"), which MMD
honours at render time.  X4 does not read PMX flags, so a single-sided mesh
would lose half its faces the moment the camera crosses it -- the classic
"the hair disappears when I look from the other side".

There are two ways the source can be safe:

  * the author mirrored the geometry, so a coincident, oppositely-wound copy
    of every thin surface already exists in the mesh (what the previous
    project's model did), or
  * they did not, and the replacement has to duplicate the faces itself.

Measured by hashing each face's sorted vertex positions and counting the pairs
that share a position set but have opposite winding.
"""

import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                       # noqa: E402
from ganyu_src import SRC_PMX, stem_of        # noqa: E402


def main():
    m = read_pmx(SRC_PMX)
    P = np.array(m['positions'], float)
    faces = [f for f in m['faces'] if len(set(f)) == 3]

    # key: frozenset of *quantised* positions, value: list of face indices
    Q = np.round(P * 1000.0).astype(np.int64)
    key_of = {}
    buckets = defaultdict(list)
    for fi, f in enumerate(faces):
        k = tuple(sorted(tuple(Q[v]) for v in f))
        buckets[k].append(fi)

    doubled = 0
    for k, fl in buckets.items():
        if len(fl) < 2:
            continue
        # do any two of them wind oppositely?
        signs = []
        for fi in fl:
            a, b, c = (Q[v] for v in faces[fi])
            # orientation sign of the triangle in its own index order
            signs.append(tuple(faces[fi]))
        for i in range(len(signs)):
            for j in range(i + 1, len(signs)):
                if set(signs[i]) == set(signs[j]) and signs[i] != signs[j]:
                    doubled += 1
                    break
            else:
                continue
            break

    n = len(faces)
    print('faces %d' % n)
    print('faces with a coincident, oppositely-wound twin: %d (%.1f%%)'
          % (doubled, doubled * 100.0 / n))

    # per material
    print('\n%-10s %7s %9s  %s' % ('stem', 'faces', 'doubled', 'verdict'))
    fi = 0
    for mt in m['materials']:
        cnt = mt['face_count'] // 3
        stemsel = list(range(fi, fi + cnt))
        fi += cnt
        stem = stem_of(mt['name'])
        if not stem:
            continue
        good = 0
        tot = 0
        for k, fl in buckets.items():
            sel = [x for x in fl if x in set(stemsel)]
            if len(sel) < 2:
                continue
            tot += len(sel)
            for i in range(len(sel)):
                for j in range(i + 1, len(sel)):
                    if set(faces[sel[i]]) == set(faces[sel[j]]):
                        good += 1
                        break
        frac = good / max(1, tot) * 100 if tot else 0.0
        verdict = ('already double-sided' if frac > 60 else
                   'SINGLE sided -- needs duplication' if frac < 10 else
                   'partly doubled (%.0f%%)' % frac)
        print('%-10s %7d %9d  %s' % (stem, cnt, good, verdict))

    # material flag bit 0
    print('\n-- PMX material flag bit 0 (draw both sides) --')
    for mt in m['materials']:
        stem = stem_of(mt['name'])
        if stem:
            print('  %-10s flag=0x%02x  double_sided=%s'
                  % (stem, mt['flag'], bool(mt['flag'] & 0x01)))


if __name__ == '__main__':
    main()
