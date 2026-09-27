# -*- coding: utf-8 -*-
"""Dump a PMX model's structure: bones, materials, textures, geometry stats."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import read_pmx, bone_children, summary     # noqa: E402

SRC = (r"C:\Users\DKX\Desktop\新建文件夹 (3)"
       r"\艾梅莉埃_by_原神_6d6213999a4e1701a2b4a71bade7e6fd\艾梅莉埃.pmx")


def main(path=SRC):
    m = read_pmx(path)
    print(summary(m))

    pos = m['positions']
    xs = [p[0] for p in pos]
    ys = [p[1] for p in pos]
    zs = [p[2] for p in pos]
    print('  bbox  X[%.2f, %.2f]  Y[%.2f, %.2f]  Z[%.2f, %.2f]'
          % (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)))

    print('\n== textures (%d) ==' % len(m['textures']))
    for i, t in enumerate(m['textures']):
        print('  [%2d] %s' % (i, t))

    print('\n== materials (%d) ==' % len(m['materials']))
    for i, mt in enumerate(m['materials']):
        tex = m['textures'][mt['texture']] if 0 <= mt['texture'] < len(m['textures']) else '-'
        sph = m['textures'][mt['sphere']] if 0 <= mt['sphere'] < len(m['textures']) else '-'
        print('  [%2d] %-24s faces=%-6d tex=%-28s sph=%-22s mode=%d flag=%d '
              'diff=(%.2f,%.2f,%.2f,%.2f) spec=%.2f'
              % (i, mt['name'], mt['face_count'] // 3, tex, sph,
                 mt['sphere_mode'], mt['flag'], *mt['diffuse'], mt['specularity']))

    print('\n== bones (%d) ==' % len(m['bones']))
    kids = bone_children(m['bones'])
    for i, b in enumerate(m['bones']):
        p = m['bones'][b['parent']]['name'] if b['parent'] >= 0 else '<root>'
        p3 = b['position']
        print('  [%3d] %-22s parent=%-22s pos=(%7.2f,%7.2f,%7.2f) kids=%s%s'
              % (i, b['name'], p, p3[0], p3[1], p3[2],
                 ','.join(kids.get(b['name'], [])[:4]),
                 ' IK' if b['flags'] & 0x20 else ''))

    print('\n== weight coverage ==')
    n_no = sum(1 for w in m['weights'] if not w)
    n_bdef1 = sum(1 for w in m['weights'] if len(w) == 1 and w[0][1] > 0.999)
    print('  vertices %d   unweighted %d   single-bone %d   multi-bone %d'
          % (len(m['weights']), n_no, n_bdef1, len(m['weights']) - n_no - n_bdef1))

    from collections import Counter
    c = Counter()
    for w in m['weights']:
        for bi, wv in w:
            c[m['bones'][bi]['name']] += 1
    print('  bones actually referenced: %d' % len(c))
    for name, n in c.most_common(200):
        print('    %-24s %d' % (name, n))


if __name__ == '__main__':
    main(*sys.argv[1:])
