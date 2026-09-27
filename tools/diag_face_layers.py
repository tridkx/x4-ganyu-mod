# -*- coding: utf-8 -*-
"""Which material is that white patch?

    blender -b --factory-startup --python tools/diag_face_layers.py

Renders the exported head with every material replaced by a flat, emissive
colour, so shading cannot be mistaken for geometry.  Layered facial parts
(eye white, iris, highlight, lid line, lash, brow, face shading) all sit within
a couple of millimetres of each other, and a stray layer is far easier to name
by colour than to infer from a lit render.
"""

import importlib
import math
import re
import os
import pathlib
import sys

import addon_utils
import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
WORK = os.path.join(PROJ, 'work')
SHARED = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "shared")
ADDON_DIR = os.path.join(
    SHARED, "X4CharacterConverter 2152 v0.8.7 2026-06-13T03-09Z QePzPJC03")
X4_ROOT = os.path.join(SHARED, "x4root")
PKG = os.path.join(WORK, 'x4cc_pkg')
OUT = os.path.join(WORK, 'preview')

ASSETS = [
    os.path.join(PKG, 'ganyu_head', 'assets', 'characters', 'mycharacters',
                 'bodies', 'ganyu_head.xac'),
    os.path.join(PKG, 'ganyu_body', 'assets', 'characters', 'mycharacters',
                 'bodies', 'ganyu_body.xac'),
]

#: material local name -> flat colour.  Chosen to be unmistakable and to make
#: the pairs that can be confused (eyewhite/iris/eyespec) far apart.
COLORS = {
    'face': (0.95, 0.80, 0.72), 'face2': (1.0, 0.0, 1.0),
    'eyewhite': (1.0, 1.0, 1.0), 'iris': (0.1, 0.1, 0.9),
    'eyespec': (0.0, 1.0, 1.0), 'eyelid': (1.0, 0.0, 0.0),
    'eyelash': (0.0, 0.0, 0.0), 'brow': (0.5, 0.0, 0.5),
    'mouth': (0.9, 0.2, 0.2), 'teeth': (1.0, 1.0, 0.4),
    'hair': (0.9, 0.7, 0.1), 'acc1': (0.0, 0.8, 0.0),
    'acc2': (1.0, 0.5, 0.0),
    'skin': (0.9, 0.75, 0.65), 'cloth': (0.3, 0.4, 0.9),
    'ribbon': (0.6, 0.3, 0.9), 'skirt': (0.2, 0.6, 0.6),
    'petti': (0.8, 0.8, 0.8), 'elem': (0.0, 1.0, 0.3),
}
DEFAULT = (0.5, 0.5, 0.5)

sys.path.insert(0, ADDON_DIR)


def flat_materials():
    n = 0
    for mat in bpy.data.materials:
        # strip Blender's duplicate suffix only when it really is one:
        # `ganyu.face` must NOT become `ganyu`, or every material collapses
        # onto the same colour and the whole render becomes meaningless.
        m = re.match(r'^(.*)\.\d{3}$', mat.name)
        base = m.group(1) if m else mat.name
        local = base.split('.', 1)[1] if '.' in base else base
        col = COLORS.get(local, DEFAULT)
        mat.use_nodes = True
        nt = mat.node_tree
        nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        emit = nt.nodes.new('ShaderNodeEmission')
        emit.inputs[0].default_value = (col[0], col[1], col[2], 1.0)
        emit.inputs[1].default_value = 1.0
        nt.links.new(emit.outputs[0], out.inputs['Surface'])
        mat.use_backface_culling = True
        n += 1
        print('   %-22s -> %s' % (base, col))
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    importlib.import_module("X4CharacterConverter")
    addon_utils.enable("X4CharacterConverter", default_set=True)
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = X4_ROOT + os.sep
    from X4CharacterConverter import addon as A

    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("X4CharacterConverter", default_set=True)
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = X4_ROOT + os.sep
    for path in ASSETS:
        A.import_actor(bpy.context, pathlib.Path(path))
    n = flat_materials()
    print('flat materials: %d' % n)

    scn = bpy.context.scene
    try:
        scn.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scn.render.engine = 'BLENDER_EEVEE'
    scn.render.resolution_x = 700
    scn.render.resolution_y = 700
    scn.view_settings.view_transform = 'Standard'
    world = bpy.data.worlds.new('W')
    scn.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (0.02, 0.02, 0.02, 1)
    # no lights: emission only, so every pixel is exactly its material colour

    cam_data = bpy.data.cameras.new('cam')
    cam_data.type = 'ORTHO'
    cam = bpy.data.objects.new('cam', cam_data)
    scn.collection.objects.link(cam)
    scn.camera = cam

    views = [('layers_face', 0.0, 46, 168), ('layers_face34', 40.0, 46, 168),
             ('layers_head', 0.0, 70, 165), ('layers_body', 0.0, 230, 100)]
    for name, ang, ortho, tz in views:
        a = math.radians(ang)
        cam.data.ortho_scale = ortho
        cam.location = (math.sin(a) * 600, math.cos(a) * 600, tz)
        d = Vector((0, 0, tz)) - Vector(cam.location)
        cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        scn.render.filepath = os.path.join(OUT, '%s.png' % name)
        bpy.ops.render.render(write_still=True)
        print('   wrote %s.png' % name)


if __name__ == '__main__':
    main()
