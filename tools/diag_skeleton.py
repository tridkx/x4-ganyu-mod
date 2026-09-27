# -*- coding: utf-8 -*-
"""Print the spine / leg / arm chain hierarchy of a PMX rig.

The bone *declaration order* in a PMX file says nothing about the tree, and for
the spine it is outright misleading (miHoYo rigs declare 上半身2 before
上半身3), so the tree has to be walked, not read off the list.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pmx import read_pmx, bone_children                  # noqa: E402

SRC = (r"C:\Users\DKX\Desktop\新建文件夹 (3)"
       r"\【女主角_荧】_by_原神_39fd8673fd145a6c65858227788cb173\荧.pmx")


def tree(model, roots, max_depth=6):
    kids = bone_children(model['bones'])
    pos = {b['name']: b['position'] for b in model['bones']}

    def walk(name, depth, prefix=''):
        if depth > max_depth:
            return
        p = pos.get(name, (0, 0, 0))
        print('%s%-18s (%6.2f, %6.2f, %6.2f)' % (prefix, name, *p))
        for k in kids.get(name, []):
            walk(k, depth + 1, prefix + '  ')

    for r in roots:
        walk(r, 0)
        print()


def main(path=SRC):
    m = read_pmx(path)
    print('== spine / neck / head ==')
    tree(m, ['センター'], max_depth=4)
    print('== left arm ==')
    tree(m, ['左肩'], max_depth=4)
    print('== left leg (plain + D) ==')
    tree(m, ['左足'], max_depth=3)
    tree(m, ['左足D'], max_depth=3)
    print('== chain roots (skirt / ribbon / hair / hat) ==')
    kids = bone_children(m['bones'])
    for b in m['bones']:
        n = b['name']
        if n.startswith(('Q_', '左带', '右带', '头饰', '+Hair')):
            parent = (m['bones'][b['parent']]['name'] if b['parent'] >= 0
                      else None)
            if not (parent or '').startswith(('Q_', '左带', '右带', '头饰',
                                              '+Hair')):
                print('  %-14s parent=%-14s kids=%s'
                      % (n, parent, ','.join(kids.get(n, [])[:3])))


if __name__ == '__main__':
    main(*sys.argv[1:])
