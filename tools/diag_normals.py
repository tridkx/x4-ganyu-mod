# -*- coding: utf-8 -*-
"""Are the exported normals sane?  (Deterministic, no renderer involved.)

The exporter writes `loop.normal`, so a mesh can look right in a flat-shaded
preview and still ship broken normals.  Two ways they go wrong here:

* **Cancellation.**  Sharing vertices between a face and its reversed twin
  makes Blender average N with -N, so the vertex normal collapses and each
  corner points somewhere arbitrary.  Fixed by giving the back layer its own
  vertices -- this check exists to prove it, since the symptom (white slivers)
  is identical to several other faults.
* **Inside-out smoothing.**  A surface whose winding disagrees with its
  neighbours gets a smoothed normal pointing the other way.

The second half of the report (corner-normal spread) reads **the opposite way
round from how it was first written**: `mesh.loops[i].normal` is the *vertex*
normal under smooth shading and the *face* normal under flat shading, so

    spread 0.000  ->  every corner of a vertex agrees  ->  SMOOTH
    spread > 0    ->  corners differ per face          ->  FLAT

The Lumine project's copy of this file labelled 0 as "flat" and sent one
debugging round chasing a shading bug that did not exist; both projects'
builds measure exactly 0.0000 here, which is correct.

Reported per mesh: the fraction of corners whose normal agrees with their own
polygon's geometric normal, and the same after grouping by material.

    blender -b --factory-startup --python tools/diag_normals.py
"""

import os
import sys
from collections import defaultdict

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(os.path.dirname(HERE), 'work')
STAGE1 = os.path.join(WORK, 'ganyu_x4_stage1.blend')


def main():
    bpy.ops.wm.open_mainfile(filepath=STAGE1)
    print('%-22s %7s %8s | %-30s %s'
          % ('object', 'corners', 'dot<0.5', 'per material (bad / total)',
             'verdict'))
    bad_total = 0
    tot_total = 0
    for ob in sorted((o for o in bpy.data.objects if o.type == 'MESH'),
                     key=lambda o: o.name):
        me = ob.data
        me.calc_loop_triangles()
        ln = np.array([tuple(l.normal) for l in me.loops], float)
        per_mat = defaultdict(lambda: [0, 0])
        bad = 0
        for poly in me.polygons:
            pn = np.array(poly.normal, float)
            n = np.linalg.norm(pn)
            if n < 1e-9:
                continue
            pn = pn / n
            for li in poly.loop_indices:
                v = ln[li]
                nv = np.linalg.norm(v)
                d = float(np.dot(v / nv, pn)) if nv > 1e-9 else 0.0
                mname = (me.materials[poly.material_index].name
                         if me.materials else '?')
                per_mat[mname][1] += 1
                if d < 0.5:
                    bad += 1
                    per_mat[mname][0] += 1
        tot = len(me.loops)
        det = ', '.join('%s %d/%d' % (k.split('.')[-1], v[0], v[1])
                        for k, v in sorted(per_mat.items()) if v[0])
        verdict = ('OK' if bad == 0 else
                   '** %d corners disagree' % bad)
        print('%-22s %7d %8d | %-30s %s'
              % (ob.name, tot, bad, det[:30] if det else '-', verdict))
        bad_total += bad
        tot_total += tot

    print('\ntotal: %d of %d corners have a normal disagreeing with their own '
          'face (%.3f%%)' % (bad_total, tot_total,
                             bad_total * 100.0 / max(1, tot_total)))

    # ------------------------------------------------- smooth vs flat check
    # flat shading would also show as every corner matching its own face, so
    # measure the opposite: how much the corners *within one vertex* differ.
    print('\n-- shading smoothness (corners sharing a vertex) --')
    for ob in sorted((o for o in bpy.data.objects if o.type == 'MESH'),
                     key=lambda o: o.name):
        me = ob.data
        by_vert = defaultdict(list)
        for poly in me.polygons:
            for li in poly.loop_indices:
                by_vert[me.loops[li].vertex_index].append(
                    np.array(me.loops[li].normal, float))
        spread = []
        for v, ns in by_vert.items():
            if len(ns) < 2:
                continue
            A = np.array(ns)
            spread.append(float(np.linalg.norm(A - A.mean(0), axis=1).mean()))
        if spread:
            m = float(np.mean(spread))
            print('  %-22s mean corner-normal spread %.4f  (%s)'
                  % (ob.name, m,
                     'SMOOTH - every corner at a vertex agrees'
                     if m < 1e-6 else
                     'FLAT - corners disagree, one normal per face'))


if __name__ == '__main__':
    main()
