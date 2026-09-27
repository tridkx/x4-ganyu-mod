# -*- coding: utf-8 -*-
"""甘雨 (Ganyu, Genshin Impact) -> X4 Argon female, stage 1: retarget + mesh.

    blender -b --factory-startup --python tools/build_ganyu_x4.py

Steps
-----
1. Import a vanilla X4 Argon-female XAC through X4CharacterConverter to get the
   authoritative Biped bind pose (Z-up, cm).  Argon women own the shared
   component (`character_argon_female_01`), so this rig *is* the target.
2. Read the PMX (geometry + skin weights + materials + bone tree).
3. Retarget bone by bone (see retarget_core / ganyu_mmd_to_x4).
4. Split by material into the two assets X4 needs -- a macro has a `head` slot
   and a `torso` slot.

Splitting by material rather than by height: this model's materials are already
cleanly separated -- every face/hair/head part is its own material -- and a
height split would cut `发` (which hangs from 251 cm... i.e. from the crown down
to z = 99, well below the shoulders) in half and put the lower strands in the
torso asset.  The strands are weighted to `Bip01 Head` and to `发_*_*` chains
that fold onto it, so they must travel with the head.

Materials whose geometry sits *exactly* on another material's are pushed apart
along their own normals (`CLOTH_OFFSET`, `FACE2_OFFSET`, `VISION_OFFSET`) --
two shells at the same depth z-fight in the engine's depth buffer, which in
game reads as flickering patches.  Measured pairs are in
`tools/diag_ganyu_overlays.py`.

Stage 2 wires the X4 materials and calls the addon's export_package.
"""

import importlib
import os
import sys

import addon_utils
import bpy
import numpy as np
import pathlib

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
WORK = os.path.join(PROJ, 'work')
SHARED = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "shared")
ADDON_DIR = os.path.join(
    SHARED, "X4CharacterConverter 2152 v0.8.7 2026-06-13T03-09Z QePzPJC03")
X4_ROOT = os.path.join(SHARED, "x4root")
RETARGET_TOOLS = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools")

for p in (HERE, RETARGET_TOOLS, ADDON_DIR):
    sys.path.insert(0, p)

from pmx import read_pmx                                   # noqa: E402
from ganyu_src import (SRC_DIR, SRC_PMX, MAT_ASCII,        # noqa: E402
                       DROP_STEMS, ALPHA_KEEP, sample_face_alpha)
from ganyu_mmd_to_x4 import (MmdAdapter, build_bone_map, report_map,  # noqa: E402
                       check_core, fix_neck_source_weights,
                       blend_chain_offsets, LIMB_CHAINS)  # noqa: F401
from retarget_core import BindPoseRetarget                 # noqa: E402

HOST_XAC = os.path.join(
    X4_ROOT, r"assets\characters\argon\heads\char_arg_f_dyn_blend_head.xac")

#: which asset each material belongs to, and which host mesh slot it fills.
#: Slots the host declares but we do not use are collapsed by stage 2.
#:
#: 13 of the 19 source materials are head parts (`面1` face skin, the eye and
#: mouth layers, `发` hair, `角` horns); 6 are body parts (`肌` skin, `服`
#: dress, `裙摆` skirt hem, `新規` waist wrap, `腰饰` sash, `神之眼` Vision).
#:
#: `脸红` (blush) is **deliberately dropped**: `表情.png` is alpha <= 66 over
#: its whole surface (median 0 through its own UVs), so it is a faint decal
#: that an NPC never shows; keeping it would mean shipping an ALPHA8 layer for
#: nothing.  See `DROP_STEMS` in ganyu_src.py.
#:
#: `角` takes the head's third slot (ornaments) rather than the hair slot: it
#: uses the hair atlas but is a separate part, and the host declares three head
#: mesh ids anyway.
SLOTS = {
    'head': [
        ('face', ['face', 'face2', 'eyewhite', 'brow', 'eyelid', 'eyelash',
                  'mouth', 'teeth', 'eyespec', 'iris'], 0),
        ('hair', ['hair'], 1),
        ('acc', ['horn'], 2),
    ],
    'body': [
        ('skin', ['skin'], 0),
        ('cloth', ['cloth', 'skirt', 'extra', 'waist', 'vision'], 1),
    ],
}

#: which materials live in which asset -- derived from SLOTS so the two can
#: never drift apart.  `发` is head *and* body in the sense that its lower
#: strands hang past the shoulders, but only the head asset is right: the
#: strands are weighted to `Bip01 Head` and would tear if split by height.
REGION_OF = {}
for _asset, _slots in SLOTS.items():
    for _name, _stems, _slot in _slots:
        for _s in _stems:
            REGION_OF[_s] = _asset

#: `cloth` sits exactly on `skin`: measured in `diag_ganyu_overlays.py`, **100%
#: of `skin`'s 766 vertices** (and 91% of `cloth`'s 7440) have a neighbour
#: within 1 mm, minimum 0.00 -- the dress is modelled on the body.  Push the
#: garment out along its own normals so the two stop z-fighting; 2 mm is far
#: below anything visible at NPC range and far above the depth buffer's
#: resolution there.
CLOTH_OFFSET = 0.2

#: `face2` (the face shading layer) floats a median of 0.4 mm above `face`,
#: with 68% of it inside 1 mm -- close enough that the depth buffer can tie.
FACE2_OFFSET = 0.15

#: `vision` (神之眼, the Vision on her hip) sits a median of 1.5 mm off `waist`
#: with half of its vertices inside 1 mm.
VISION_OFFSET = 0.15

#: Materials that are closed solids -- their outside is all the camera ever
#: sees.  No geometry is duplicated for anything (see the note in the mesh loop
#: below); this set only decides the blend mode in stage 2.
#:
#: Everything else is a **thin sheet** (hair cards, sleeves, skirt, lashes,
#: decals).  The source marks every material "draw both sides" (flag bit 0 is
#: set on all 19) and relies on MMD's renderer to honour it; X4 culls back
#: faces, so a single-layer sheet simply vanishes when seen from behind -- the
#: "hair disappears from the other side" symptom.  `blendmode="TWOSIDED"` is
#: the fix; duplicating the geometry was tried in an earlier project and had to
#: be removed (two shells at identical depth flicker).
SOLID_STEMS = {'face', 'eyewhite', 'iris', 'teeth', 'skin'}

#: No decimation anywhere: the model is 16.6k vertices, which lands the two
#: assets around 5-11k -- inside the 6x budget without touching geometry.
#: Decimation rewrites UVs on collapse, and this model samples a shared 2048
#: atlas per material, so a shifted UV reads a neighbouring part of the sheet.
DECIMATE_RATIO = {}
DECIMATE_FLOOR = 150

#: vanilla's own sneaker mesh bottoms out at -0.32 cm
GROUND_Z = -0.5


def load_x4_armature():
    importlib.import_module("X4CharacterConverter")
    addon_utils.enable("X4CharacterConverter", default_set=True)
    bpy.context.preferences.addons[
        "X4CharacterConverter"].preferences.data_root = X4_ROOT + os.sep
    from X4CharacterConverter import addon as A

    A.import_actor(bpy.context, pathlib.Path(HOST_XAC))
    arm = next(o for o in bpy.data.objects if o.type == 'ARMATURE')
    for ob in list(bpy.data.objects):
        if ob.type == 'MESH':
            bpy.data.objects.remove(ob, do_unlink=True)

    mw = arm.matrix_world
    bones = {}
    for b in arm.data.bones:
        bones[b.name] = {
            'head': tuple(mw @ b.head_local),
            'tail': tuple(mw @ b.tail_local),
            'parent': b.parent.name if b.parent else None,
        }
    return arm, bones


def decimate(ob, ratio, floor=DECIMATE_FLOOR):
    """Collapse-decimate in place; returns (verts, unweighted, failed)."""
    if ratio >= 0.999 or len(ob.data.polygons) < 4:
        return len(ob.data.vertices), 0, False
    before = len(ob.data.vertices)
    mod = ob.modifiers.new(name="Decimate", type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    mod.use_collapse_triangulate = True
    dg = bpy.context.evaluated_depsgraph_get()
    new_me = bpy.data.meshes.new_from_object(ob.evaluated_get(dg))
    old = ob.data
    ob.modifiers.clear()
    ob.data = new_me
    bpy.data.meshes.remove(old)
    if len(ob.data.vertices) < floor:
        return before, 0, True
    unweighted = sum(1 for v in ob.data.vertices
                     if not any(ge.weight > 1e-6 for ge in v.groups))
    return len(ob.data.vertices), unweighted, False


#: Laplacian smoothing passes over the skin weights.
#:
#: This is what keeps a joint from creasing when its vertices are split across
#: bones whose fitted rotations differ.  The neck is the visible case: its
#: lower third is ~60% `上半身2` (-> Spine2, which carries a 9.4 degree fit
#: rotation) and its upper third is 100% `頭` (-> Head, 0.3 degrees), so the
#: blend sweeps the lower neck 1.65 cm sideways relative to the upper neck and
#: the profile shows a visible kink mid-neck.  More passes widen each vertex's
#: neighbourhood, so the transition happens over more of the neck instead of
#: across one band of triangles.
WEIGHT_SMOOTH_ROUNDS = 6


def smooth_vertex_weights(ob, rounds=WEIGHT_SMOOTH_ROUNDS, alpha=0.5):
    """Laplacian-smooth skin weights so a sharp transition cannot pinch.

    The cloth chains are bound to a single bone on purpose (see ganyu_mmd_to_x4), so
    the places this matters are the shoulder/hip/neck transitions where the MMD
    rig and the X4 biped disagree about where a joint is.
    """
    me = ob.data
    n = len(me.vertices)
    adj = [[] for _ in range(n)]
    for e in me.edges:
        a, b = e.vertices
        adj[a].append(b)
        adj[b].append(a)
    gname = {g.index: g.name for g in ob.vertex_groups}
    W = [{gname[ge.group]: ge.weight for ge in v.groups
          if ge.weight > 1e-6 and ge.group in gname} for v in me.vertices]
    for _ in range(rounds):
        NW = []
        for i, w in enumerate(W):
            acc = {k: v * (1.0 - alpha) for k, v in w.items()}
            nb = adj[i]
            if nb:
                share = alpha / len(nb)
                for j in nb:
                    for k, val in W[j].items():
                        acc[k] = acc.get(k, 0.0) + val * share
            tot = sum(acc.values())
            NW.append({k: v / tot for k, v in acc.items() if v > 1e-6}
                      if tot > 1e-9 else w)
        W = NW
    for g in list(ob.vertex_groups):
        ob.vertex_groups.remove(g)
    groups = {}
    for i, w in enumerate(W):
        tot = sum(w.values())
        if tot > 1e-9:
            w = {k: v / tot for k, v in w.items()}
            top = max(w, key=w.get)
            w[top] += 1.0 - sum(w.values())
        for name, val in w.items():
            g = groups.get(name)
            if g is None:
                g = ob.vertex_groups.new(name=name)
                groups[name] = g
            g.add([i], val, 'REPLACE')


def foot_vertex_mask(weights):
    out = []
    for wd in weights:
        s = sum(wv for bn, wv in wd.items()
                if bn.endswith(' Foot') or 'Toe' in bn)
        out.append(s > 0.5)
    return np.array(out, bool)


def lift_feet(verts, weights, ground=GROUND_Z):
    """Align the soles to the floor, either direction."""
    mask = foot_vertex_mask(weights)
    if not mask.any():
        return verts, 0.0
    dz = ground - float(verts[mask][:, 2].min())
    if abs(dz) < 0.05:
        return verts, 0.0
    out = verts.copy()
    out[mask, 2] += dz
    return out, dz


#: Blender-side cache for the source atlases' alpha channels.
_ALPHA_CACHE = {}


def atlas_alpha(path):
    """(H, W) alpha of a source atlas, row 0 = the top of the image.

    Blender's image rows start at the *bottom* and PMX's v = 0 is at the top,
    so the array is flipped once here; downstream everything can then treat it
    like the PIL array `ganyu_src.sample_face_alpha` is written for, and the
    builder and `diag_ganyu_alpha_faces.py` share one definition of "invisible
    face" instead of each inventing its own.
    """
    key = os.path.normpath(path).lower()
    if key not in _ALPHA_CACHE:
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = 'Non-Color'
        w, h = img.size
        px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
        _ALPHA_CACHE[key] = px[::-1, :, 3] * 255.0
    return _ALPHA_CACHE[key]


def make_material(name, texture):
    """Placeholder material: stage 2 replaces these with the X4 ones.

    A submesh with no material slot at all is silently dropped, so every
    material referenced by the geometry has to exist here even though the real
    DDS binding happens later.
    """
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (100, 0)
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if texture and os.path.exists(texture):
        img = bpy.data.images.load(texture, check_existing=True)
        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.label = 'Diffuse'
        tex.location = (-350, 200)
        nt.links.new(tex.outputs["Color"], bsdf.inputs['Base Color'])
    else:
        bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.55
    return mat


def main():
    arm, x4_bones = load_x4_armature()
    print("X4 armature: %d bones" % len(x4_bones))

    m = read_pmx(SRC_PMX)
    bones = m['bones']
    check_core(bones)
    print("PMX: %d verts, %d faces, %d materials, %d bones"
          % (len(m['positions']), len(m['faces']), len(m['materials']),
             len(bones)))

    src_pos = {b['name']: b['position'] for b in bones}
    src_par = {b['name']: (bones[b['parent']]['name'] if b['parent'] >= 0
                           else None) for b in bones}
    bmap = build_bone_map(bones)
    adapter = MmdAdapter(bmap)
    weighted = sorted({bones[bi]['name'] for w in m['weights'] for bi, _ in w})
    report_map(bones, bmap, set(x4_bones), weighted)

    transfer = BindPoseRetarget(src_pos, src_par, x4_bones, adapter=adapter)

    V = np.array(m['positions'], float)
    fix_neck_source_weights(m, transfer, V)
    # DISABLED (see docs §3.12): blending the spine/leg offsets removed the
    # tearing numerically but introduced visible bending at the arms and
    # ankles, which is worse than the seam it fixed.  Left here so the
    # experiment is reproducible.
    N = np.array(m['normals'], float)
    idx_bone = [b['name'] for b in bones]
    Vx, n_unw, n_targets = transfer.transform(V, m['weights'], idx_bone)
    print("  retargeted: %d verts, %d unweighted, %d target bones"
          % (len(Vx), n_unw, n_targets))
    Wx = transfer.merge_weights(m['weights'], idx_bone)
    Vx, lifted = lift_feet(Vx, Wx)
    print("  feet: %s" % ("raised %.2f cm" % lifted if lifted
                          else "already on the floor"))

    # ------------------------------------- lift coincident layers apart
    # The source normal goes through the rigid frame's *rotation only* -- a
    # normal must not be scaled or translated.
    Nx = (transfer.R @ N.T).T

    def push(verts, idx, distance):
        idx = np.array(sorted(idx))
        if not len(idx):
            return 0
        nn = Nx[idx].copy()
        ln = np.linalg.norm(nn, axis=1, keepdims=True)
        ln[ln < 1e-9] = 1.0
        verts[idx] += (nn / ln) * distance
        return len(idx)

    # `diag_ganyu_overlays` measures that **every one of `skin`'s 766 vertices**
    # has a `cloth` vertex within 1 mm, minimum 0.00 -- the dress is modelled on
    # the body.  Two surfaces at identical depth z-fight, and in game that is a
    # flickering patch wherever the garment meets the skin.  Lifting the garment
    # along its own normals separates them; 2 mm is far below anything visible
    # at NPC range and far above the depth buffer's resolution there.
    #
    # The whole garment moves together -- `skirt`, `extra` and `waist` are
    # separate materials but one continuous surface with `cloth`, so pushing
    # only `cloth` would open a 2 mm step at every seam.
    GARMENT = ('cloth', 'skirt', 'extra', 'waist')
    mat_idx = {}
    fi = 0
    for mt in m['materials']:
        n = mt['face_count'] // 3
        sel = m['faces'][fi:fi + n]
        fi += n
        stem = MAT_ASCII.get(mt['name'])
        if stem in GARMENT or stem in ('vision', 'face2'):
            mat_idx.setdefault(stem, set()).update(
                i for f in sel if len(set(f)) == 3 for i in f)

    garment_v = set()
    for stem in GARMENT:
        garment_v |= mat_idx.get(stem, set())
    n_cloth = push(Vx, garment_v, CLOTH_OFFSET)
    print('  garment: %d verts pushed %.1f mm off the skin (kills the '
          'z-fighting where the dress meets the body)'
          % (n_cloth, CLOTH_OFFSET * 10))
    # `vision` rides on `waist` (half its vertices within 1 mm), so it keeps a
    # little extra distance of its own on top of the garment push.
    n_vis = push(Vx, mat_idx.get('vision', ()), VISION_OFFSET)
    if n_vis:
        print('  vision: %d verts pushed an extra %.1f mm off the sash'
              % (n_vis, VISION_OFFSET * 10))
    # `face2` is the face shading layer: a median 0.4 mm above `face`, 68%
    # inside 1 mm.  It has to stay on the face -- 1.5 mm is invisible there.
    n_f2 = push(Vx, mat_idx.get('face2', ()), FACE2_OFFSET)
    if n_f2:
        print('  face2: %d verts pushed %.1f mm off the face skin'
              % (n_f2, FACE2_OFFSET * 10))

    # ------------------------------------------------------------------ mesh
    tex_root = SRC_DIR
    mats = m['materials']
    faces = m['faces']
    fi = 0
    groups = {}
    invisible = 0
    for mt in mats:
        n = mt['face_count'] // 3
        sel = [f for f in faces[fi:fi + n] if len(set(f)) == 3]
        fi += n
        stem = MAT_ASCII.get(mt['name'])
        if stem is None or not sel:
            continue
        if stem in DROP_STEMS:
            continue
        region = REGION_OF.get(stem)
        if region is None:
            print('   !! %s has no asset assignment, skipping' % stem)
            continue
        tex = m['textures'][mt['texture']] if 0 <= mt['texture'] < len(
            m['textures']) else None
        # Faces the author painted over: their whole surface samples alpha 0
        # through the material's own atlas, so MMD never drew them (rendered by
        # `diag_ganyu_alpha_faces.py`: 906 of `cloth`'s 9944 -- the lining
        # under the bodice -- and 86 of `skin`'s 1086, the patches the dress
        # covers).  Dropping them is exactly equivalent, and it is what lets
        # `cloth` ship as an opaque TWOSIDED material instead of needing an
        # alpha mode it cannot have (X4's blendmode is a single value).
        if tex:
            texp = os.path.normpath(os.path.join(SRC_DIR, tex))
            if os.path.exists(texp):
                a = atlas_alpha(texp)
                if float(a.min()) < 254.0:
                    uvf = np.array([[m['uvs'][i] for i in f] for f in sel],
                                   float)
                    keep = sample_face_alpha(a, uvf) >= ALPHA_KEEP
                    invisible += int((~keep).sum())
                    sel = [f for f, k in zip(sel, keep) if k]
                    if not sel:
                        continue
        g = groups.setdefault((stem, region),
                              {'faces': [], 'tex': tex, 'mat': mt['name']})
        g['faces'].extend(sel)
    print('  invisible faces dropped: %d (alpha < %.2f through their own atlas)'
          % (invisible, ALPHA_KEEP))

    blender_mats = {}
    built = []
    for (stem, region), g in sorted(groups.items()):
        used = sorted({i for f in g['faces'] for i in f})
        remap = {o: n for n, o in enumerate(used)}
        sverts = [tuple(float(x) for x in Vx[i]) for i in used]
        # NO winding flip on the source triangles: the mapping
        # (x, y, z) -> (x, -z, y) has determinant +1, so it preserves
        # orientation.  PMX's own vertex normals confirm the source winding
        # agrees with the right-hand rule (mean dot(face normal, vertex
        # normal) = +0.985, positive = 1.000).
        sfaces = [tuple(remap[i] for i in f) for f in g['faces']]
        # NO duplicated back shell.  It was added so that thin sheets stay
        # visible from behind under back-face culling, but two coincident
        # surfaces at *identical* depth z-fight, and in game that reads as
        # flickering patches -- the same failure mode the previous projects hit
        # with coincident cloth.  The game already models "draw both sides"
        # explicitly: `blendmode="TWOSIDED"` (a vanilla character cloak uses
        # it), applied in prepare_textures_ganyu.SHADER.  That costs nothing
        # in geometry and cannot fight itself.

        mat_name = 'ganyu.%s' % stem
        if mat_name not in blender_mats:
            texp = (os.path.normpath(os.path.join(tex_root, g['tex']))
                    if g['tex'] else None)
            blender_mats[mat_name] = make_material(mat_name, texp)

        obj_name = 'ganyu_%s_%s' % (stem, region[0])
        me = bpy.data.meshes.new(obj_name)
        me.from_pydata(sverts, [], sfaces)
        # NOT mesh.validate(): MMD models build double-sided surfaces by
        # mirroring geometry, and validate() treats the mirrored copy as a
        # "duplicate polygon" and deletes it -- which removed exactly half of
        # every small facial part in the previous project and left the eyes
        # rendering solid black.
        me.update()
        # smooth shading: from_pydata defaults to flat and the exporter reads
        # loop.normal, so flat here means every triangle exports its own face
        # normal (the "faceted face" symptom) and the vertex count goes up.
        for poly in me.polygons:
            poly.use_smooth = True
        # PMX stores v with 0 at the TOP of the image; Blender's UV space has
        # 0 at the BOTTOM.  Copying v through unchanged therefore flips every
        # texture vertically inside Blender -- and because the exporter applies
        # its own `1 - uv.y` (`addon.py:855`, mirrored by `1 - uv.y` on import
        # at :531), the flip survives into the .xac as `1 - v_pmx`, i.e. the
        # game samples the atlas upside down.  Flipping here makes the Blender
        # value semantically correct, and the exporter's flip then cancels it.
        uvl = [(m['uvs'][i][0], 1.0 - m['uvs'][i][1]) for i in used]
        uv_layer = me.uv_layers.new(name="UVMap")
        for loop in me.loops:
            uv_layer.data[loop.index].uv = uvl[loop.vertex_index]

        ob = bpy.data.objects.new(obj_name, me)
        bpy.context.scene.collection.objects.link(ob)
        ob.data.materials.append(blender_mats[mat_name])

        groups_vg = {}
        for newi, oldi in enumerate(used):
            for gname, gval in Wx[oldi].items():
                vg = groups_vg.get(gname)
                if vg is None:
                    vg = ob.vertex_groups.new(name=gname)
                    groups_vg[gname] = vg
                vg.add([newi], gval, 'REPLACE')

        ratio = DECIMATE_RATIO.get(stem, 1.0)
        nv_before = len(me.vertices)
        nv_after, unw, failed = decimate(ob, ratio)
        if failed:
            print("      %s: decimate would leave %d verts, keeping %d"
                  % (obj_name, nv_after, nv_before))
        smooth_vertex_weights(ob)

        ob['x4cc_part'] = stem
        ob['x4cc_region'] = region
        ob['x4cc_material'] = mat_name
        ob.parent = arm
        mod = ob.modifiers.new(name="Armature", type='ARMATURE')
        mod.object = arm
        built.append(ob)
        print("  %-22s %-5s faces %-6d verts %d -> %d  mats=%d groups=%d%s"
              % (obj_name, region, len(sfaces), nv_before, nv_after,
                 len(ob.data.materials), len(ob.vertex_groups),
                 '' if stem in SOLID_STEMS else '  (thin sheet -> TWOSIDED)'))

    out = os.path.join(WORK, "ganyu_x4_stage1.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("SAVED", out)
    print("objects: %d" % len(built))


main()
