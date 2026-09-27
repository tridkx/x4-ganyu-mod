# -*- coding: utf-8 -*-
"""Decide the frame mapping and the triangle winding from the data.

Same two questions as the previous project, re-measured because the answers are
properties of the *source file*, not of "MMD" in general:

1. Which axis is "front"?  Answered by anatomy (eye geometry, teeth, toes).
2. Does the winding have to be reversed?  Answered by comparing PMX's own
   per-vertex normals against the geometric face normal:
       dot > 0  -> source winding agrees with the right-hand rule
   A mapping with determinant +1 preserves that; -1 mirrors it and forces a
   triangle re-index.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                       # noqa: E402
from ganyu_src import SRC_PMX                 # noqa: E402

CANDIDATES = {
    '(x, -z, y)  [-z back -> +y front]': lambda p: np.stack(
        [p[:, 0], -p[:, 2], p[:, 1]], 1),
    '(x, +z, y)': lambda p: np.stack(
        [p[:, 0], p[:, 2], p[:, 1]], 1),
    '(x, y, z)  identity': lambda p: p,
}


def winding_stats(P, N, F):
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    fn = np.cross(b - a, c - a)
    ln = np.linalg.norm(fn, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    fn = fn / ln
    vn = (N[F[:, 0]] + N[F[:, 1]] + N[F[:, 2]]) / 3.0
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    d = np.einsum('ij,ij->i', fn, vn)
    return float(d.mean()), float((d > 0).mean())


def outward_ratio(P, F):
    cen = P.mean(0)
    a, b, c = P[F[:, 0]], P[F[:, 1]], P[F[:, 2]]
    fn = np.cross(b - a, c - a)
    fc = (a + b + c) / 3.0 - cen
    d = np.einsum('ij,ij->i', fn, fc)
    return float((d > 0).mean())


def main():
    m = read_pmx(SRC_PMX)
    P = np.array(m['positions'], np.float64)
    N = np.array(m['normals'], np.float64)
    F = np.array([f for f in m['faces'] if len(set(f)) == 3], np.int64)
    print('mesh: %d verts, %d triangles' % (len(P), len(F)))

    byname = {b['name']: np.array(b['position']) for b in m['bones']}
    print('\n-- which way is "front"? (bone positions, MMD units) --')
    for what, bone, ref in (('left eye', '左目', '頭'),
                            ('toe', '左足先EX', '左足首D'),
                            ('teeth', '齿下', '頭'),
                            ('chest', '左胸上2', '上半身2'),
                            ('ribbon', '左带_0_1', '上半身2')):
        if bone in byname and ref in byname:
            d = byname[bone] - byname[ref]
            print('  %-9s %-10s minus %-10s = (%+6.2f, %+6.2f, %+6.2f)'
                  % (what, bone, ref, *d))

    for bone, r in (('左目', 0.35), ('右目', 0.35)):
        if bone in byname:
            bp = byname[bone]
            near = P[np.linalg.norm(P - bp, axis=1) < r]
            if len(near):
                print('  %-28s %d verts, mean offset (%+.2f, %+.2f, %+.2f)'
                      % ('eye geometry vs ' + bone, len(near), *near.mean(0)))

    print('\n-- winding: mean dot(face normal, PMX vertex normal) --')
    base_mean, base_pos = winding_stats(P, N, F)
    print('  source (as authored)        mean=%+.3f  positive=%.3f  '
          'outward=%.3f' % (base_mean, base_pos, outward_ratio(P, F)))

    print('\n-- after each candidate mapping --')
    for label, fn in CANDIDATES.items():
        P2, N2 = fn(P), fn(N)
        mean, pos = winding_stats(P2, N2, F)
        out = outward_ratio(P2, F)
        det = np.linalg.det(fn(np.eye(3)))
        print('  %-34s det=%+.0f  mean=%+.3f  positive=%.3f  outward=%.3f  -> %s'
              % (label, det, mean, pos, out,
                 'KEEP winding' if pos > 0.5 else 'REVERSE winding'))

    print('\n-- after reversing the triangles (sanity) --')
    for label, fn in CANDIDATES.items():
        P2, N2 = fn(P), fn(N)
        Fr = F[:, ::-1]
        mean, pos = winding_stats(P2, N2, Fr)
        print('  %-34s mean=%+.3f  positive=%.3f  outward=%.3f'
              % (label, mean, pos, outward_ratio(P2, Fr)))


if __name__ == '__main__':
    main()
