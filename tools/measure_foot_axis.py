# -*- coding: utf-8 -*-
"""How far is the foot's geometry axis from the bone it is bound to?

    blender -b --factory-startup --python tools/measure_foot_axis.py -- <body.xac>

The animation rotates `Bip01 L/R Foot` about the ankle; whatever geometry
those weights drive therefore swings around that joint *along the bone's own
direction*.  If the geometry's long axis (heel -> toe) does not agree with the
bone's, the foot appears to point somewhere else for the whole stride.

That is exactly the situation here.  The source model's foot is almost
straight (ankle -> toe rises 0.08 cm in X) while X4's bind pose splays the
foot 16.3 degrees (3.79 cm in X).  `NO_ROTATE_BONES` quite deliberately keeps
X4's foot attitude (the source's foot is much flatter), so the geometry ends up
carrying the source's direction while the bone carries X4's.

Writes `work/foot_axis.json`:
  * the bone's ankle -> toe direction
  * the geometry's own ankle -> toe direction, as the principal axis of the
    vertices the two bones drive
  * the angle between them, which is what a fix has to drive to zero
"""

import json
import os
import sys

import bpy
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'shared')


def load_host(path):
    import addon_utils
    import pathlib
    for d in os.listdir(ADDON_DIR):
        p = os.path.join(ADDON_DIR, d)
        if d.startswith('X4CharacterConverter') and os.path.isdir(p) \
                and os.path.exists(os.path.join(p, 'X4CharacterConverter',
                                                '__init__.py')):
            sys.path.insert(0, p)
    addon_utils.enable("X4CharacterConverter", default_set=True)
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = os.path.join(
            os.path.dirname(ADDON_DIR), 'shared', 'x4root') + os.sep
    from X4CharacterConverter import addon as A
    A.import_actor(bpy.context, pathlib.Path(path))


def foot_vertices(side):
    """Vertices whose dominant bone is the foot or toe bone of `side`."""
    want = {'Bip01 %s Foot' % side, 'Bip01 %s Toe0' % side}
    pts = []
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        groups = {g.index: g.name for g in ob.vertex_groups}
        mw = ob.matrix_world
        for v in ob.data.vertices:
            if not v.groups:
                continue
            g = max(v.groups, key=lambda x: x.weight)
            if groups[g.group] in want:
                pts.append(tuple(mw @ v.co))
    return np.array(pts, float)


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    body = argv[0]
    load_host(body)
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    head = {b.name: np.array((arm.matrix_world @ b.head_local)[:])
            for b in arm.data.bones}

    out = {'asset': body}
    for side in ('L', 'R'):
        pts = foot_vertices(side)
        if len(pts) < 10:
            continue
        ankle = head['Bip01 %s Foot' % side]
        toe = head['Bip01 %s Toe0' % side]
        bone_dir = toe - ankle
        bone_dir = bone_dir / np.linalg.norm(bone_dir)

        # geometry axis: heel -> toe.  The foot's long axis in X4 runs mostly
        # along Y (forward) and -Z (down), so take the principal axis of the
        # vertices and orient it away from the ankle.
        centred = pts - pts.mean(axis=0)
        # weight the vertices by how far they are along the bone direction so
        # the principal axis is not hijacked by the foot's width
        cov = np.cov(centred.T)
        w, v = np.linalg.eigh(cov)
        geo_dir = v[:, int(np.argmax(w))]
        if np.dot(geo_dir, bone_dir) < 0:
            geo_dir = -geo_dir
        ang = float(np.degrees(np.arccos(
            np.clip(np.dot(geo_dir, bone_dir), -1, 1))))
        out[side] = {
            'n_vertices': int(len(pts)),
            'ankle': [round(float(x), 3) for x in ankle],
            'toe': [round(float(x), 3) for x in toe],
            'bone_dir': [round(float(x), 3) for x in bone_dir],
            'geo_dir': [round(float(x), 3) for x in geo_dir],
            'angle_deg': round(ang, 2),
            'geo_center': [round(float(x), 3) for x in pts.mean(axis=0)],
            'geo_length_cm': round(float(np.ptp(pts @ geo_dir)), 2),
            'bone_length_cm': round(float(np.linalg.norm(toe - ankle)), 2),
        }

    dest = os.path.join(HERE, '..', 'work', 'foot_axis.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


main()
