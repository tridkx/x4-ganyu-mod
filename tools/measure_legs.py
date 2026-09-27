# -*- coding: utf-8 -*-
"""Measure a built body asset's leg geometry, straight from the .xac.

    blender -b --factory-startup --python tools/measure_legs.py -- <body.xac>

Writes `work/leg_measure.json`, so the numbers can be diffed between builds:

  * the mean X of the vertices each leg bone dominates -- `+X` is the
    character's left (X4's own `left_eye_dummy` sits at +X), so a bone named
    `L` must own geometry at positive X
  * the resulting foot spacing, which is the number the catwalk report is
    about: it was 18.0 cm before the damping fix against vanilla's 39.8
  * the bone positions themselves, for the joint-vs-geometry comparison

Doing this in Blender rather than by parsing the .xac is deliberate: the
converter addon already knows how to read the file, and a hand-rolled parser
got the chunk layout wrong twice in a row.
"""

import json
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ADDON_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), 'shared')
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools"))

LEG = ['Bip01 L Thigh', 'Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L Toe0',
       'Bip01 R Thigh', 'Bip01 R Calf', 'Bip01 R Foot', 'Bip01 R Toe0']


def load_host(path):
    """Import the .xac the same way stage 2 does.

    The addon has to be *enabled* (`addon_utils.enable`), not merely imported:
    its Scene properties (`x4cc_actor_id`) only exist after registration, and
    `import_actor` writes them.
    """
    import addon_utils
    import pathlib
    for d in os.listdir(ADDON_DIR):
        p = os.path.join(ADDON_DIR, d)
        if d.startswith('X4CharacterConverter') and os.path.isdir(p) \
                and os.path.exists(os.path.join(p, 'X4CharacterConverter',
                                                '__init__.py')):
            sys.path.insert(0, p)
    addon_utils.enable("X4CharacterConverter", default_set=True)
    x4_root = os.path.join(os.path.dirname(ADDON_DIR), 'shared', 'x4root')
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = x4_root + os.sep
    from X4CharacterConverter import addon as A
    A.import_actor(bpy.context, pathlib.Path(path))
    return A


def main():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    body = argv[0]
    load_host(body)

    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    bones = {}
    for b in arm.data.bones:
        bones[b.name] = (arm.matrix_world @ b.head_local)[:]

    # per-vertex dominant bone across every imported mesh
    owner = {}
    verts = {}
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        groups = {g.index: g.name for g in ob.vertex_groups}
        mw = ob.matrix_world
        for v in ob.data.vertices:
            if not v.groups:
                continue
            g = max(v.groups, key=lambda x: x.weight)
            owner.setdefault(groups[g.group], []).append(
                tuple(mw @ v.co))

    out = {'asset': body, 'bones': {}, 'geometry': {}, 'counts': {}}
    for name in LEG:
        if name in bones:
            out['bones'][name] = [round(float(x), 3) for x in bones[name]]
        pts = owner.get(name)
        if not pts:
            continue
        import statistics
        xs = [p[0] for p in pts]
        out['geometry'][name] = {
            'n': len(pts),
            'x_mean': round(statistics.fmean(xs), 3),
            'x_min': round(min(xs), 3),
            'x_max': round(max(xs), 3),
        }
    for name in ('Bip01 L Foot', 'Bip01 R Foot'):
        if name in out['geometry']:
            out.setdefault('x_mean', {})[name] = out['geometry'][name]['x_mean']
    if 'Bip01 L Foot' in out['geometry'] and 'Bip01 R Foot' in out['geometry']:
        lf = out['geometry']['Bip01 L Foot']['x_mean']
        rf = out['geometry']['Bip01 R Foot']['x_mean']
        out['foot_spacing_cm'] = round(lf - rf, 3)
    if 'Bip01 L Toe0' in out['geometry'] and 'Bip01 R Toe0' in out['geometry']:
        out['toe_spacing_cm'] = round(
            out['geometry']['Bip01 L Toe0']['x_mean']
            - out['geometry']['Bip01 R Toe0']['x_mean'], 3)

    dest = os.path.join(HERE, '..', 'work', 'leg_measure.json')
    with open(dest, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    print('written: %s' % os.path.abspath(dest))


main()
