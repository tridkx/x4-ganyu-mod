# -*- coding: utf-8 -*-
"""Is the shipped .xac smooth-shaded or flat-shaded?

The skill's criterion, run against the *exported* asset rather than the
Blender scene:

    |corner normal - face normal| ~ 0.000  -> flat  (every triangle carries
                                              its own face normal; in game
                                              the model looks faceted)
    |corner normal - face normal| ~ 0.4    -> smooth

The exporter writes `loop.normal`, and both pipeline stages rebuild meshes with
`from_pydata`, which defaults to flat -- so a missing `poly.use_smooth = True`
in *either* stage silently produces a faceted model.

    blender -b --factory-startup --python tools/diag_shading.py
"""

import importlib
import os
import pathlib
import sys

import addon_utils
import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
WORK = os.path.join(PROJ, 'work')
SHARED = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "shared")
ADDON_DIR = os.path.join(
    SHARED, "X4CharacterConverter 2152 v0.8.7 2026-06-13T03-09Z QePzPJC03")
X4_ROOT = os.path.join(SHARED, "x4root")
PKG = os.path.join(WORK, 'x4cc_pkg')

ASSETS = [
    ('ganyu_head', os.path.join(PKG, 'ganyu_head', 'assets', 'characters',
                                 'mycharacters', 'bodies', 'ganyu_head.xac')),
    ('ganyu_body', os.path.join(PKG, 'ganyu_body', 'assets', 'characters',
                                 'mycharacters', 'bodies', 'ganyu_body.xac')),
    ('vanilla_head', os.path.join(
        X4_ROOT, r"assets\characters\argon\heads\char_arg_f_dyn_blend_head.xac")),
]

sys.path.insert(0, ADDON_DIR)


def main():
    print('%-16s %8s %10s %10s  %s'
          % ('asset', 'corners', 'mean|vn-fn|', 'median', 'verdict'))
    for tag, path in ASSETS:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        addon_utils.enable("X4CharacterConverter", default_set=True)
        bpy.context.preferences.addons[
            "X4CharacterConverter"].preferences.data_root = X4_ROOT + os.sep
        from X4CharacterConverter import addon as A
        A.import_actor(bpy.context, pathlib.Path(path))

        diffs = []
        smooth_flags = []
        for ob in bpy.data.objects:
            if ob.type != 'MESH':
                continue
            me = ob.data
            me.calc_loop_triangles()
            # Blender 4.1+ moved this; `loop.normal` still resolves but is
            # derived, so prefer corner_normals when present
            try:
                cn = np.array([tuple(v.vector) for v in me.corner_normals],
                              float)
            except AttributeError:
                cn = np.array([tuple(l.normal) for l in me.loops], float)
            for poly in me.polygons:
                pn = np.array(poly.normal, float)
                n = np.linalg.norm(pn)
                if n < 1e-9:
                    continue
                pn /= n
                idx = list(poly.loop_indices)
                d = np.linalg.norm(cn[idx] - pn, axis=1)
                diffs.extend(d.tolist())
                smooth_flags.append(bool(poly.use_smooth))
        if not diffs:
            print('%-16s (no mesh)' % tag)
            continue
        d = np.array(diffs)
        n_smooth = sum(smooth_flags)
        verdict = ('FLAT  (faceted in game)' if d.mean() < 0.05
                   else 'smooth' if d.mean() > 0.2 else 'mixed?')
        print('%-16s %8d %10.4f %10.4f  %s   [use_smooth on %d/%d polys]'
              % (tag, len(d), d.mean(), float(np.median(d)), verdict,
                 n_smooth, len(smooth_flags)))


if __name__ == '__main__':
    main()
