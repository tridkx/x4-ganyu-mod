# -*- coding: utf-8 -*-
"""Render the assembled .xac assets through Blender for inspection.

    blender -b --factory-startup --python tools/render_preview.py

**This render is not trustworthy for appearance.  Use
`render_textured.py --x4` instead.**  It imports the exported .xac (good: it
proves the material names resolve and shows the real welded vertex count), but
the material it ends up drawing is assembled by the importer and then patched
by `wire_textures`, and the result does not match what the game shows: the
chest renders with dark wedges, the whole model reads grey, and the collar
looks like a stiff flap -- none of which appear in the software render of the
same geometry with the same source textures.  The cause is somewhere in that
material patch-up; it has not been chased down because the software renderer is
the better oracle (it shares `DROP_STEMS`, alpha tests what the engine alpha
tests, and samples the source atlas directly).

What it *is* good for: the vertex/triangle counts it prints after import, and a
rough sanity check that nothing is grossly misplaced.
"""

import importlib
import json
import math
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
    ('head', os.path.join(PKG, 'ganyu_head', 'assets', 'characters',
                          'mycharacters', 'bodies', 'ganyu_head.xac')),
    ('body', os.path.join(PKG, 'ganyu_body', 'assets', 'characters',
                          'mycharacters', 'bodies', 'ganyu_body.xac')),
]

sys.path.insert(0, ADDON_DIR)


def wire_textures(manifest):
    """Attach the prepared DDS maps to the imported materials by name.

    Both assets reference the same `ganyu` collection, so materials such as
    `ganyu.cloth` exist twice and the second import becomes `ganyu.cloth.001`
    -- Blender cannot hold two datablocks with one name.  Matching on the exact
    name silently skips those and they then render in the Principled BSDF's
    default grey, which reads as "the texture is wrong".  Strip the duplicate
    suffix before looking the material up.
    """
    n = 0
    for mat in bpy.data.materials:
        entry = manifest.get(mat.name)
        if entry is None and '.' in mat.name:
            entry = manifest.get(mat.name.rsplit('.', 1)[0])
        if not entry or not mat.use_nodes:
            continue
        nt = mat.node_tree
        # Rebuild rather than patch: the importer wires its own texture nodes
        # (pointing at vanilla maps that the unpacked root does not have), and
        # adding a second link to Base Color leaves whatever else it set --
        # metallic, normal, emission -- in charge.  A clean tree is the only
        # way the render shows the mod's own material.
        nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        out.location = (420, 0)
        bsdf = nt.nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (120, 0)
        nt.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        for role, path in entry['textures'].items():
            if not os.path.exists(path):
                continue
            img = bpy.data.images.load(path, check_existing=True)
            img.colorspace_settings.name = (
                'Non-Color' if role in ('Normal', 'Smoothness') else 'sRGB')
            node = nt.nodes.new('ShaderNodeTexImage')
            node.image = img
            node.name = 'pv_' + role
            node.location = (-500, {'Diffuse': 300, 'Normal': 0,
                                    'Smoothness': -300}.get(role, -600))
            if role == 'Diffuse':
                nt.links.new(node.outputs['Color'], bsdf.inputs['Base Color'])
            elif role == 'Smoothness':
                # X4's `Smoothness` map is glossiness; Blender wants the
                # inverse as roughness.  Without this the importer's default
                # roughness leaves every surface mirror-like and the render
                # reads as polished metal rather than cloth.
                inv = nt.nodes.new('ShaderNodeInvert')
                inv.location = (-200, -300)
                nt.links.new(node.outputs['Color'], inv.inputs['Color'])
                nt.links.new(inv.outputs['Color'], bsdf.inputs['Roughness'])
        # MMD sources are dielectric; the importer's default metallic makes the
        # dress look chromed under the three suns.
        if 'Metallic' in bsdf.inputs:
            bsdf.inputs['Metallic'].default_value = 0.0
        elif 'Metallic IOR' in bsdf.inputs:
            bsdf.inputs['Metallic IOR'].default_value = 0.0
        if os.environ.get('LUMINE_NO_SMOOTH_LINK'):
            pass          # diagnostic: leave roughness at the importer default
        elif os.environ.get('LUMINE_FLAT_LIGHT'):
            # diagnostic: kill every specular response, so anything still
            # white is geometry or texture rather than a highlight
            for link in list(nt.links):
                if link.to_socket.name in ('Roughness', 'Specular IOR Level',
                                           'Specular Tint'):
                    nt.links.remove(link)
            bsdf.inputs['Roughness'].default_value = 1.0
            if 'Specular IOR Level' in bsdf.inputs:
                bsdf.inputs['Specular IOR Level'].default_value = 0.0
        n += 1
    return n


def hide_placeholders():
    """Hide the collapsed slots.

    Stage 2 cannot delete a host mesh slot -- the exporter rebuilds a deleted
    slot from the template with the *vanilla* geometry -- so unused slots are
    filled with a 1 cm triangle at (0, 0, 120), inside the torso, where the
    body hides it.  In game it is genuinely invisible.  Here it is not: the
    unpacked game root has none of the vanilla textures, so those slots fall
    back to an untextured black material and the triangle renders as a black
    wedge on the chest -- which is exactly what it looked like, and it is a
    preview artefact rather than a mod defect.
    """
    n = 0
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.name.startswith('ganyu_unused'):
            ob.hide_render = True
            n += 1
    return n


def enable_backface_culling():
    """Match the engine's render state before believing anything on screen.

    EEVEE draws both sides by default, so the back layer stage 1 adds -- the
    same vertices with reversed winding, at exactly the same depth -- competes
    for the depth buffer and shows up as a spray of white slivers.  X4 culls
    back faces (`p1_character` needs an explicit `blendmode="TWOSIDED"` on the
    one vanilla cloak that wants both sides, which is the proof), so the game
    sees one layer and no conflict.  Rendering with culling on is therefore the
    honest preview; leaving it off manufactures a bug that does not ship.
    """
    n = 0
    for mat in bpy.data.materials:
        if mat.use_nodes:
            mat.use_backface_culling = True
            n += 1
    return n


def setup_scene():
    scn = bpy.context.scene
    try:
        scn.render.engine = 'BLENDER_EEVEE_NEXT'
    except TypeError:
        scn.render.engine = 'BLENDER_EEVEE'
    scn.render.resolution_x = 620
    scn.render.resolution_y = 900
    scn.render.film_transparent = False
    scn.view_settings.view_transform = 'Standard'
    world = bpy.data.worlds.new('W')
    scn.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (0.05, 0.055, 0.07, 1.0)
    bg.inputs[1].default_value = 1.0

    # Lights are rebuilt per view -- see `light_for()`.  Fixed world-space suns
    # light the front of the model and leave the back in shadow, which made
    # every back view look muddy and sent me chasing a contrast problem that
    # was only ever the key light sitting on the wrong side.

    cam_data = bpy.data.cameras.new('cam')
    cam_data.type = 'ORTHO'
    cam_data.ortho_scale = 230
    cam = bpy.data.objects.new('cam', cam_data)
    bpy.context.scene.collection.objects.link(cam)
    scn.camera = cam
    return cam


def light_for(angle_deg):
    """Three suns placed relative to the *camera*, not to the world."""
    for ob in [o for o in bpy.data.objects if o.type == 'LIGHT']:
        bpy.data.objects.remove(ob, do_unlink=True)
    a = math.radians(angle_deg)
    fwd = (math.sin(a), math.cos(a), 0.0)          # camera -> subject
    right = (math.cos(a), -math.sin(a), 0.0)
    for name, (along, side, up), energy in (
            ('key', (0.75, 0.55, 0.75), 3.2),
            ('fill', (0.35, -0.90, 0.35), 1.1),
            ('rim', (-0.90, 0.10, 0.55), 1.6)):
        direction = tuple(fwd[i] * along + right[i] * side for i in range(2)) \
            + (up,)
        light = bpy.data.lights.new(name, 'SUN')
        light.energy = energy
        ob = bpy.data.objects.new(name, light)
        ob.location = tuple(c * 100 for c in direction)
        bpy.context.scene.collection.objects.link(ob)
        d = Vector((0, 0, 100)) - Vector(ob.location)
        ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def aim(cam, angle_deg, target=(0, 0, 100), dist=600, ortho=230, height=0.0):
    """angle 0 looks at the character's FRONT.

    In X4/Blender the character faces +Y, so the camera has to sit on the +Y
    side; putting it at -Y renders the back and makes every check ambiguous.
    """
    a = math.radians(angle_deg)
    cam.data.ortho_scale = ortho
    cam.location = (math.sin(a) * dist, math.cos(a) * dist, target[2] + height)
    d = Vector(target) - Vector(cam.location)
    cam.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(WORK, 'tex_out', 'mats',
                                           'manifest.json'), encoding='utf-8'))
    importlib.import_module("X4CharacterConverter")
    addon_utils.enable("X4CharacterConverter", default_set=True)
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = X4_ROOT + os.sep
    from X4CharacterConverter import addon as A

    # one scene with both assets, which is how the game composes an NPC
    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("X4CharacterConverter", default_set=True)
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = X4_ROOT + os.sep
    for tag, path in ASSETS:
        A.import_actor(bpy.context, pathlib.Path(path))
    n = wire_textures(manifest)
    culled = enable_backface_culling()
    hidden = hide_placeholders()
    print('hidden %d collapsed placeholder slots' % hidden)
    print('backface culling enabled on %d materials' % culled)
    meshes = [o for o in bpy.data.objects if o.type == 'MESH']
    tl = sum(len(o.data.loops) for o in meshes)
    verts = sum(len(o.data.vertices) for o in meshes)
    print('[combined] %d meshes, %d blender verts, %d triangles, '
          '%d materials textured'
          % (len(meshes), verts, tl // 3, n))

    cam = setup_scene()
    views = [('front', 0.0, 230, 100, 0.0), ('side', 90.0, 230, 100, 0.0),
             ('back', 180.0, 230, 100, 0.0), ('q34', 35.0, 230, 100, 0.0),
             ('face', 0.0, 46, 168, 0.0), ('face34', 40.0, 46, 168, 0.0),
             ('lower', 0.0, 130, 45, 0.0),
             # the three regions reported from the software preview: arm /
             # chest, the collar, and the back of the neck
             ('chest', 0.0, 70, 130, 0.0), ('arm', 0.0, 90, 120, 0.0),
             ('backneck', 180.0, 60, 158, 0.0)]
    for name, ang, ortho, tz, h in views:
        aim(cam, ang, target=(0, 0, tz), ortho=ortho, height=h)
        light_for(ang)
        bpy.context.scene.render.filepath = os.path.join(OUT, 'lu_%s.png' % name)
        bpy.ops.render.render(write_still=True)
        print('   wrote lu_%s.png' % name)


if __name__ == '__main__':
    main()
