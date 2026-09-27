# -*- coding: utf-8 -*-
"""甘雨 -> X4 Argon female mod builder (stage 2): fill host slots, export XACs.

    blender -b --factory-startup --python tools/build_ganyu_mod.py

Two assets, because an X4 NPC macro has a separate `head` and `torso` mesh
slot:

    head : host char_arg_f_dyn_blend_head          (91-bone reference rig)
           slot 0 face+eyes, slot 1 hair, slot 2 horns
    body : host char_arg_f_sweater_leggings_civ_01
           slot 0 skin, slot 1 dress + skirt + sash + Vision

Both are *vanilla* hosts, because the converter can only rebuild meshes the
template already declares -- a slot that is deleted comes back carrying the
original geometry, so unused slots are collapsed to a 1 cm triangle buried
inside the body instead.

The hosts are the Argon-female ones, which is what this mod targets:
`character_argon_female_01` is the shared component every Argon woman uses, so
the skeleton, the 1100+ animations and the eye look-at all come from it.
"""

import importlib
import os
import shutil
import sys

import addon_utils
import bpy
import pathlib

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
WORK = os.path.join(PROJ, 'work')
SHARED = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "shared")
ADDON_DIR = os.path.join(
    SHARED, "X4CharacterConverter 2152 v0.8.7 2026-06-13T03-09Z QePzPJC03")
X4_ROOT = os.path.join(SHARED, "x4root")
RETARGET_TOOLS = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools")

STAGE1 = os.path.join(WORK, "ganyu_x4_stage1.blend")
MOD_ROOT = os.path.join(WORK, "x4_ganyu_mod")
MOD_ID = "x4_ganyu_mod"
DDS_DIR = os.path.join(WORK, "tex_out", "mats")
PKG_DIR = os.path.join(WORK, "x4cc_pkg")

for p in (HERE, RETARGET_TOOLS, ADDON_DIR):
    sys.path.insert(0, p)

import x4_materials                                    # noqa: E402

HOSTS = {
    'head': r"assets\characters\argon\heads\char_arg_f_dyn_blend_head.xac",
    'body': r"assets\characters\argon\bodies\char_arg_f_sweater_leggings_civ_01.xac",
}

#: target -> [(object name, host mesh_id, [(stem, region), ...])]
SLOT_PLAN = {
    'head': [
        ("ganyu_head_face", 0, [
            ('face', 'head'), ('face2', 'head'), ('eyewhite', 'head'),
            ('brow', 'head'), ('eyelid', 'head'), ('eyelash', 'head'),
            ('mouth', 'head'), ('teeth', 'head'), ('eyespec', 'head'),
            ('iris', 'head')]),
        ("ganyu_head_hair", 1, [('hair', 'head')]),
        ("ganyu_head_acc", 2, [('horn', 'head')]),
    ],
    'body': [
        ("ganyu_body_skin", 0, [('skin', 'body')]),
        ("ganyu_body_cloth", 1, [
            ('cloth', 'body'), ('skirt', 'body'), ('extra', 'body'),
            ('waist', 'body'), ('vision', 'body')]),
    ],
}

#: where each .xac lands under the extension root
ASSET_PATH = {
    'head': r"assets\characters\argon\ganyu\heads\ganyu_head.xac",
    'body': r"assets\characters\argon\ganyu\bodies\ganyu_body.xac",
}


def read_part_geometry():
    """{(stem, region): [ {material_name, verts, weights, uvs, faces} ]}."""
    bpy.ops.wm.open_mainfile(filepath=STAGE1)
    parts = {}
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        stem = ob.get('x4cc_part')
        region = ob.get('x4cc_region')
        if not stem:
            continue
        me = ob.data
        gname = {g.index: g.name for g in ob.vertex_groups}
        verts = [tuple(v.co) for v in me.vertices]
        weights = []
        for v in me.vertices:
            wd = {}
            for ge in v.groups:
                if ge.weight > 1e-6:
                    nm = gname.get(ge.group)
                    if nm:
                        wd[nm] = wd.get(nm, 0.0) + ge.weight
            weights.append(wd)
        uv_layer = me.uv_layers[0].data if len(me.uv_layers) else None
        uvs = []
        if uv_layer is not None:
            for loop in me.loops:
                uvs.append(tuple(uv_layer[loop.index].uv))
        mat = me.materials[0] if me.materials else None
        parts.setdefault((stem, region), []).append({
            'material_name': mat.name if mat else None,
            'verts': verts, 'weights': weights, 'uvs': uvs,
            'faces': [tuple(p.vertices) for p in me.polygons],
        })
    return parts


def fill_slot(host_ob, obj_name, sources, materials):
    """Replace a host mesh with our merged geometry."""
    verts, weights, faces, uvs, mat_order, mat_index = [], [], [], [], [], {}

    for src in sources:
        mn = src['material_name']
        mat = materials.get(mn) or materials.get(None)
        if mat is None:
            continue
        if mat.name not in mat_index:
            mat_index[mat.name] = len(mat_order)
            mat_order.append(mat)
        slot = mat_index[mat.name]

        base = len(verts)
        verts.extend(tuple(p) for p in src['verts'])
        weights.extend(src['weights'])
        for f in src['faces']:
            faces.append((f[0] + base, f[1] + base, f[2] + base, slot))
        uvs.extend(src.get('uvs') or [])

    if not faces:
        # A .xac cannot express an empty slot: deleting the object makes the
        # exporter rebuild it from the template (with the *vanilla* geometry),
        # and a 3-vertex stub is rejected for having no bone weights.  A 1 cm
        # triangle inside the chest is invisible and valid both ways.
        verts = [(0.0, 0.0, 120.0), (1.0, 0.0, 120.0), (0.0, 1.0, 120.0)]
        weights = [{'Bip01 Spine1': 1.0} for _ in range(3)]
        faces = [(0, 1, 2, 0)]
        uvs = [(0.0, 0.0)] * 3
        mat_order = [next(iter(materials.values()))]

    me = bpy.data.meshes.new(obj_name)
    me.from_pydata(verts, [], [(f[0], f[1], f[2]) for f in faces])
    # NOT mesh.validate(): stage 1 deliberately adds a back layer that re-uses
    # the same vertices with reversed winding, and validate() treats a
    # coincident reversed polygon as a "duplicate" and deletes it -- which
    # would remove exactly the double-siding and make the hair vanish from
    # behind.  (In the previous project the same call silently halved every
    # small facial part instead.)
    me.update()
    # smooth shading again: this mesh is rebuilt from scratch and from_pydata
    # defaults to flat, and the exporter writes loop.normal
    for poly in me.polygons:
        poly.use_smooth = True
    for m in mat_order:
        me.materials.append(m)
    # `validate()` can drop degenerate or duplicate polygons, and the material
    # index is applied by zipping against the *original* face list -- if the
    # counts differ, every face after the dropped one silently gets the wrong
    # material (one symptom is an eye layer painted with the lashes' map).
    if len(me.polygons) != len(faces):
        print("   !! %s: from_pydata produced %d of %d faces -- material "
              "assignment would misalign"
              % (obj_name, len(me.polygons), len(faces)))
    else:
        for poly, f in zip(me.polygons, faces):
            poly.material_index = f[3]

    uv = me.uv_layers.new(name="UVMap")
    if len(uvs) == len(me.loops):
        # pass through unchanged: the exporter applies `1 - uv.y` itself
        for loop, v in zip(me.loops, uvs):
            uv.data[loop.index].uv = (v[0], v[1])
    else:
        print("   !! %s: uv count %d != loops %d, using (0,0)"
              % (obj_name, len(uvs), len(me.loops)))
        for loop in me.loops:
            uv.data[loop.index].uv = (0.0, 0.0)

    host_ob.data = me
    host_ob.name = obj_name
    for g in list(host_ob.vertex_groups):
        host_ob.vertex_groups.remove(g)
    groups = {}
    for vi, wd in enumerate(weights):
        for gname, gval in wd.items():
            g = groups.get(gname)
            if g is None:
                g = host_ob.vertex_groups.new(name=gname)
                groups[gname] = g
            g.add([vi], gval, 'REPLACE')
    return len(verts), len(faces), len(mat_order)


def build_asset(target):
    parts = read_part_geometry()
    used = {s['material_name'] for e in parts.values() for s in e}
    print("[%s] parts=%d materials=%d" % (target, len(parts), len(used)))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    addon_utils.enable("X4CharacterConverter", default_set=True)
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = X4_ROOT + os.sep
    from X4CharacterConverter import addon as A2

    mats = x4_materials.create_materials(DDS_DIR, only=used,
                                         collection='ganyu')
    print("[%s] materials built: %d" % (target, len(mats) - 1))

    A2.import_actor(bpy.context, pathlib.Path(
        os.path.join(X4_ROOT, HOSTS[target])))
    slots = {}
    for ob in bpy.data.objects:
        if ob.type == 'MESH' and ob.get("x4cc_actor_id"):
            mid = ob.get("x4cc_mesh_id")
            if mid is not None:
                slots[int(mid)] = ob
    print("[%s] host slots: %s" % (target, sorted(slots)))

    kept = []
    for obj_name, mesh_id, wanted in SLOT_PLAN[target]:
        host_ob = slots.get(mesh_id)
        if host_ob is None:
            print("   !! slot %d missing" % mesh_id)
            continue
        sources = [s for key in wanted for s in parts.get(key, [])]
        if not sources:
            fill_slot(host_ob, obj_name, [], mats)
            print("   -- slot %d (%s) emptied" % (mesh_id, obj_name))
            kept.append(host_ob)
            continue
        nv, nf, nm = fill_slot(host_ob, obj_name, sources, mats)
        print("   %-20s slot=%d verts=%-6d faces=%-6d mats=%d"
              % (obj_name, mesh_id, nv, nf, nm))
        kept.append(host_ob)

    for mesh_id, ob in slots.items():
        if ob not in kept:
            fill_slot(ob, "ganyu_unused_%d" % mesh_id, [], mats)
            print("   -- slot %d collapsed (empty)" % mesh_id)

    export_name = os.path.splitext(os.path.basename(ASSET_PATH[target]))[0]
    pkg_target = os.path.join(PKG_DIR, export_name)
    if os.path.isdir(pkg_target):
        shutil.rmtree(pkg_target)
    pkg = A2.export_package(bpy.context, pathlib.Path(pkg_target))
    print("[%s] exported package -> %s" % (target, pkg))
    return pkg


def main():
    os.makedirs(PKG_DIR, exist_ok=True)
    for target in ('head', 'body'):
        build_asset(target)


if __name__ == '__main__':
    main()
