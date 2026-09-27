# -*- coding: utf-8 -*-
"""Textured software render of the *source* PMX, for ground truth.

The untextured previews use each material's mean colour, which is fine for
silhouettes but cannot answer "is this white band on her face part of the
model, or did the conversion put it there?" -- a mean colour erases exactly the
detail in question.  This samples the actual atlas, so the source can be
compared against the converted render pixel for pixel.

    python tools/render_textured.py [--x4] [--view face|full] [--angle 0]

`--x4` renders `work/retarget_check.npz` (the retargeted geometry) instead of
the raw PMX, which is the direct A/B.
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx                                   # noqa: E402
from ganyu_src import SRC_DIR, SRC_PMX, DROP_STEMS, stem_of   # noqa: E402

WORK = os.path.join(os.path.dirname(HERE), 'work')
OUT = os.path.join(WORK, 'preview')


def build_atlas(model):
    """material index -> mip pyramid for that material's texture.

    A mip chain is not a nicety here.  The hair atlas is a dense strand pattern
    a few pixels wide per strand; point-sampling it through a UV that sweeps
    many texels per screen pixel aliases into a moire that reads as **black
    stripes across golden hair** -- which looks exactly like a broken UV or a
    wrong texture.  The source model renders with the same stripes, so the
    stripes are the renderer's, not the conversion's; sampling the right mip
    level is what makes that visible instead of merely arguable.
    """
    cache = {}

    def load(rel):
        if rel not in cache:
            rel_use = rel
            if os.environ.get('LUMINE_CLEAN_ATLAS'):
                alt = os.path.join(
                    SRC_DIR, os.path.dirname(rel),
                    '_clean_' + os.path.basename(rel))
                if os.path.exists(alt):
                    rel_use = alt
            p = os.path.join(SRC_DIR, rel_use.replace('\\', os.sep)
                             .replace('/', os.sep))
            im = Image.open(p).convert('RGBA')
            # Same colour bleed the texture step applies before encoding, so
            # this preview shows what the shipped DDS mips actually contain.
            # Without it the transparent-black region of 肌.png bleeds into
            # neighbouring texels at every mip level and the masked edges read
            # as dark speckling -- which the engine's alpha test cannot remove
            # either, because the averaged alpha lands above the cut-off.
            try:
                sys.path.insert(0, HERE)
                from prepare_textures_ganyu import bleed_colors
                if (np.asarray(im)[..., 3] < 128).any():
                    im = bleed_colors(im)
            except Exception as exc:
                print('   (colour bleed unavailable: %s)' % exc)
            rgb, alpha = [], []
            cur = im
            while True:
                a = np.asarray(cur, np.float32) / 255.0
                rgb.append(a[..., :3])
                alpha.append(a[..., 3])
                if min(cur.size) <= 4:
                    break
                cur = cur.resize((max(1, cur.size[0] // 2),
                                  max(1, cur.size[1] // 2)), Image.BOX)
            cache[rel] = (rgb, alpha)
        return cache[rel]
    return load


def render(verts, faces, face_mat, model, path, size=(760, 760),
           eye=None, target=None, fov=30.0, bg=(0.10, 0.11, 0.13),
           light=(0.35, 0.85, 0.45), flip_y=True, cull=True,
           force_lod=None, id_out=None, vnormals=None, shade=0.55,
           alpha_test=True):
    """Z-buffered raster with per-pixel UV interpolation and texture lookup."""
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)
    W, H = size
    UV = np.array(model['uvs'], float)
    load = build_atlas(model)

    # resolve each material's atlas once
    mat_tex = {}
    for mi, mt in enumerate(model['materials']):
        ti = mt['texture']
        if 0 <= ti < len(model['textures']):
            mat_tex[mi] = load(model['textures'][ti])

    if eye is None or target is None:
        c = V.mean(0)
        r = float(np.abs(V - c).max()) * 2.1
        target = c
        eye = c + np.array([0.0, r, r * 0.25])
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    f = target - eye
    f = f / np.linalg.norm(f)
    up = np.array([0.0, 0.0, 1.0])
    s = np.cross(f, up)
    s = s / max(np.linalg.norm(s), 1e-9)
    u = np.cross(s, f)
    basis = np.stack([s, u, f])
    cam = (V - eye) @ basis.T
    z = np.maximum(cam[:, 2], 1e-3)
    scale = (H * 0.5) / np.tan(np.radians(fov) * 0.5)
    px = cam[:, 0] / z * scale
    if flip_y:
        px = -px
    px = px + W * 0.5
    py = H * 0.5 - cam[:, 1] / z * scale

    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.array(bg, np.float32)
    zbuf = np.full((H, W), np.inf, np.float32)
    # which material actually won each pixel -- without this, "the hair is
    # striped" cannot be told apart from "something is drawn over the hair"
    idbuf = np.full((H, W), -1, np.int32)
    uvbuf = np.full((H, W, 2), -1.0, np.float32)
    fbuf = np.full((H, W), -1, np.int32)

    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    fn = np.cross(V[b] - V[a], V[c] - V[a])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
    ld = np.array(light, float)
    ld /= np.linalg.norm(ld)

    # Back-face culling, because the engine does it.
    #
    # Thin-sheet materials carry a back layer at *exactly* the same depth with
    # reversed winding (that is the whole point -- the game culls one of them).
    # A rasteriser that draws both puts two coincident surfaces in the depth
    # buffer, and which one wins flips per triangle: that is the striped
    # hair.  `p1_character` needs an explicit `blendmode="TWOSIDED"` on the one
    # vanilla cloak that wants both sides, which is the proof the engine culls.
    to_eye = eye[None, :] - (V[a] + V[b] + V[c]) / 3.0
    front = np.einsum('ij,ij->i', fn, to_eye) > 0.0
    if cull:
        F = F[front]
        face_mat = face_mat[front]
        fn = fn[front]
        a, b, c = F[:, 0], F[:, 1], F[:, 2]

    depth = np.minimum(np.minimum(z[a], z[b]), z[c])

    for fi in np.argsort(-depth):
        i0, i1, i2 = F[fi]
        mi = face_mat[fi]
        mips = mat_tex.get(mi)
        if mips is None:
            continue
        tex = mips[0]
        x0, y0 = px[i0], py[i0]
        x1, y1 = px[i1], py[i1]
        x2, y2 = px[i2], py[i2]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))) + 1, W)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))) + 1, H)
        if minx >= maxx or miny >= maxy:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx) + 0.5,
                             np.arange(miny, maxy) + 0.5)
        w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
        w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
        w2 = 1.0 - w0 - w1
        mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not mask.any():
            continue
        zz = w0 * z[i0] + w1 * z[i1] + w2 * z[i2]
        sub = zbuf[miny:maxy, minx:maxx]
        hit = mask & (zz < sub)
        if not hit.any():
            continue
        # NOTE: the depth write happens *after* the alpha test, further down.
        # Writing it here stored depth for texels the alpha test then discarded,
        # so a masked-out surface still occluded whatever sat behind it and the
        # hole showed the background instead of the geometry behind -- which is
        # what made the masked edges of this model's leg trim read as black
        # stripes.  A GPU tests alpha before it writes depth; so does this now.
        # pick a mip level from this triangle's texel-per-pixel ratio
        h0, w0p = mips[0][0].shape[:2]
        # Area ratio, not edge ratio: the bangs are long thin slivers, and an
        # edge-length estimate calls them "small on screen" and keeps the
        # sharpest mip, which left moire stripes exactly there while the rest
        # of the head came out clean.
        uvd = np.array([UV[i1] - UV[i0], UV[i2] - UV[i0]]) * [w0p, h0]
        uv_area = abs(uvd[0, 0] * uvd[1, 1] - uvd[0, 1] * uvd[1, 0])
        pxd = np.array([[x1 - x0, y1 - y0], [x2 - x0, y2 - y0]])
        px_area = abs(pxd[0, 0] * pxd[1, 1] - pxd[0, 1] * pxd[1, 0])
        if force_lod is None:
            ratio = max(uv_area / max(px_area, 1e-6), 1.0)
            lod = int(np.clip(0.5 * np.log2(ratio), 0, len(mips) - 1))
        else:
            lod = min(force_lod, len(mips) - 1)
        rgb_mips, a_mips = mips
        tex = rgb_mips[lod]
        amap = a_mips[lod]
        h, w = tex.shape[:2]
        uu = (w0 * UV[i0][0] + w1 * UV[i1][0] + w2 * UV[i2][0]) % 1.0
        vv = (w0 * UV[i0][1] + w1 * UV[i1][1] + w2 * UV[i2][1]) % 1.0
        # Bilinear, not nearest.  The hair sheet is a dense strand pattern:
        # neighbouring triangles land on texels that alternate light and dark,
        # so nearest-neighbour paints each triangle one flat texel and the
        # whole fringe reads as black-and-gold stripes -- at *any* mip level,
        # because the alternation is between faces, not within them.  A real
        # GPU filters per pixel, which is why the same texture looks like plain
        # golden hair everywhere else.
        fx = uu * (w - 1)
        fy = vv * (h - 1)
        x0i = np.clip(fx.astype(int), 0, w - 1)
        y0i = np.clip(fy.astype(int), 0, h - 1)
        x1i = np.clip(x0i + 1, 0, w - 1)
        y1i = np.clip(y0i + 1, 0, h - 1)
        ax = np.clip(fx - x0i, 0.0, 1.0)[..., None]
        ay = np.clip(fy - y0i, 0.0, 1.0)[..., None]
        c00 = tex[y0i, x0i]
        c10 = tex[y0i, x1i]
        c01 = tex[y1i, x0i]
        c11 = tex[y1i, x1i]
        col = (c00 * (1 - ax) * (1 - ay) + c10 * ax * (1 - ay)
               + c01 * (1 - ax) * ay + c11 * ax * ay)

        # Shading: interpolate the source's own vertex normals when we have
        # them (Gouraud), otherwise fall back to one normal per face.
        #
        # Per-face normals make every triangle a flat facet, so a smooth
        # surface reads as creased and "crumpled" -- which is exactly the
        # symptom a missing `use_smooth` produces in a real export.  Chasing
        # that here would have been a wild goose chase: `diag_shading.py`
        # measures the *shipped* .xac at mean |vn - fn| = 0.186 (head) and
        # 0.242 (body) against vanilla's 0.205, i.e. the export is smooth.
        # The facets were this rasteriser's.
        if vnormals is not None:
            nrm = (w0[..., None] * vnormals[i0] + w1[..., None] * vnormals[i1]
                   + w2[..., None] * vnormals[i2])
            nn = np.linalg.norm(nrm, axis=-1, keepdims=True)
            nrm = nrm / np.maximum(nn, 1e-9)
            nd_map = np.abs(np.einsum('...k,k->...', nrm, ld))
            lit = shade + (1.0 - shade) * nd_map
        else:
            lit = shade + (1.0 - shade) * abs(float(np.dot(fn[fi], ld)))

        # Alpha test.  This model paints its cut-outs as *transparent black*
        # in the atlas -- 肌.png is 88% alpha 0 -- so a rasteriser that ignores
        # alpha paints solid black bars across the thighs and the back of the
        # neck, and they read as corruption.  In game those materials ship as
        # ALPHA1 (see prepare_textures_ganyu.ALPHA_STEMS), so the engine drops
        # exactly these pixels; ignoring alpha here manufactures a defect that
        # does not exist.
        if alpha_test and amap is not None:
            a00 = amap[y0i, x0i]
            a10 = amap[y0i, x1i]
            a01 = amap[y1i, x0i]
            a11 = amap[y1i, x1i]
            # alpha is a single channel, so the mix factors must be 2-D --
            # (h, w) with (h, w, 1) broadcasts into garbage instead of erroring
            ax2, ay2 = ax[..., 0], ay[..., 0]
            av = (a00 * (1 - ax2) * (1 - ay2) + a10 * ax2 * (1 - ay2)
                  + a01 * (1 - ax2) * ay2 + a11 * ax2 * ay2)
            # `av` is already (h, w): indexing `av[..., 0]` on a 2-D array
            # takes a column, not a channel
            hit = hit & (av > 0.5)
            if not hit.any():
                continue
        if vnormals is not None:
            img[miny:maxy, minx:maxx][hit] = np.clip(
                col[hit] * lit[hit][..., None], 0, 1)
        else:
            img[miny:maxy, minx:maxx][hit] = np.clip(col[hit] * lit, 0, 1)
        if id_out is not None:
            uvbuf[miny:maxy, minx:maxx][hit] = np.stack(
                [uu[hit], vv[hit]], -1)
            fbuf[miny:maxy, minx:maxx][hit] = fi
        sub[hit] = zz[hit]
        if id_out is not None:
            idbuf[miny:maxy, minx:maxx][hit] = mi

    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    if id_out is not None:
        np.save(id_out, idbuf)
        np.save(id_out.replace('.npy', '_uv.npy'), uvbuf)
        np.save(id_out.replace('.npy', '_face.npy'), fbuf)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x4', action='store_true')
    ap.add_argument('--glob', action='store_true',
                    help='source geometry through the global fit only -- '
                         'directly comparable with --x4')
    ap.add_argument('--view', default='face',
                    choices=('face', 'full', 'eyes', 'upper', 'neck'))
    ap.add_argument('--angle', type=float, default=0.0)
    ap.add_argument('--scale', type=float, default=1.0,
                    help='render resolution multiplier')
    ap.add_argument('--id-out', default=None,
                    help='also write a per-pixel material-index buffer')
    ap.add_argument('--no-alpha', action='store_true',
                    help='skip the alpha test (diagnostic: what alpha removes)')
    ap.add_argument('--no-shade', action='store_true',
                    help='write the sampled texture colour with no lighting '
                         '(diagnostic: what is dark is dark *in the atlas*)')
    ap.add_argument('--only', default=None,
                    help='render a single material stem (diagnostic)')
    ap.add_argument('--lod', type=int, default=None,
                    help='force one mip level (diagnostic)')
    ap.add_argument('--no-cull', action='store_true',
                    help='draw back faces too (diagnostic: coincident shells '
                         'then fight in the depth buffer)')
    a = ap.parse_args()

    m = read_pmx(SRC_PMX)
    faces_all = m['faces']
    face_mat = []
    fi = 0
    for mi, mt in enumerate(m['materials']):
        n = mt['face_count'] // 3
        face_mat.extend([mi] * n)
        fi += n
    keep = [i for i, f in enumerate(faces_all) if len(set(f)) == 3]
    # Honour the same DROP_STEMS the builder uses.  Reading the raw PMX and
    # rendering *every* material made the preview lie: `hairspec` (髪+) is a
    # 98%-transparent black specular overlay sitting exactly on top of `hair`,
    # and a rasteriser with no alpha blending paints it solid -- so the hair
    # came out black-striped in the picture while being perfectly golden in
    # the mod, which drops that material.  A diagnostic that does not share
    # the pipeline's configuration is worse than no diagnostic.
    keep = [i for i in keep
            if stem_of(m['materials'][face_mat[i]]['name']) not in DROP_STEMS]
    F = np.array([faces_all[i] for i in keep], np.int64)
    FM = np.array([face_mat[i] for i in keep], np.int64)
    dropped = {stem_of(mt['name']) for mt in m['materials']} & DROP_STEMS
    if dropped:
        print('dropped (same as the builder): %s' % ', '.join(sorted(dropped)))

    if a.only:
        keepm = np.array([stem_of(m['materials'][mi]['name']) == a.only
                          for mi in FM])
        F, FM = F[keepm], FM[keepm]
        print('only %s: %d faces' % (a.only, len(F)))

    def build_retarget():
        import json
        sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools"))
        from ganyu_mmd_to_x4 import (MmdAdapter, build_bone_map, LIMB_CHAINS,
                               blend_chain_offsets)
        from retarget_core import BindPoseRetarget
        x4 = {n: {'head': tuple(b['head']), 'tail': tuple(b['tail']),
                  'parent': b['parent']}
              for n, b in json.load(open(
                  os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "work", "vanilla", "x4_skeleton_dump.json"),
                  encoding='utf-8'))['head']['bones'].items()}
        bones = m['bones']
        sp = {b['name']: b['position'] for b in bones}
        par = {b['name']: (bones[b['parent']]['name'] if b['parent'] >= 0
                           else None) for b in bones}
        tr = BindPoseRetarget(sp, par, x4,
                              adapter=MmdAdapter(build_bone_map(bones)),
                              verbose=False)
        return tr

    vnorm = None
    if a.x4 or a.glob:
        _tr = build_retarget()
        # normals take the rotation only -- no scale, no translation
        vnorm = (_tr.R @ np.array(m['normals'], float).T).T
        vnorm /= np.maximum(np.linalg.norm(vnorm, axis=1, keepdims=True), 1e-9)

    if a.x4:
        V = np.load(os.path.join(WORK, 'retarget_check.npz'))['Vx']
        prefix = 'tex_x4'
    elif a.glob:
        # The source geometry carried through the GLOBAL fit only -- same
        # scale, same rotation, same translation as the retarget, but with no
        # per-bone correction.  Rendering this in the retarget's own frame
        # makes the two images directly comparable pixel for pixel, so
        # "the conversion did this" and "the model always looked like that"
        # stop being a matter of opinion.
        V = _tr.global_only(np.array(m['positions'], float))
        prefix = 'tex_glob'
    else:
        P = np.array(m['positions'], float)
        V = np.stack([P[:, 0], -P[:, 2], P[:, 1]], 1)
        prefix = 'tex_src'

    # Frame from the model's own bounding box rather than from hard-coded
    # centimetres: the source PMX is in MMD units (height ~19) and the
    # retargeted result is in centimetres (height ~190), a factor of ten apart.
    # Hard-coded targets framed the X4 result and left the source render empty,
    # which silently removed the A/B this script exists to provide.
    if a.x4 or a.glob:
        # Fixed camera for the retarget modes.  Deriving it from the mesh
        # bounding box re-centres each render on its own subject, so a genuine
        # 2 cm shift of the head becomes invisible -- both images just look
        # centred.  These numbers come from the X4 skeleton (Spine z=106,
        # Head z=161) so `--glob` and `--x4` frame the identical volume and
        # can be compared pixel for pixel.
        c = np.array([0.0, 0.0, 100.0])
        hgt = 190.0
    else:
        lo, hi = V.min(0), V.max(0)
        c = (lo + hi) / 2.0
        hgt = float(hi[2] - lo[2])
    lo = c - hgt * 0.5
    hi = c + hgt * 0.5
    head_z = lo[2] + hgt * 0.88
    if a.view == 'face':
        target = np.array([c[0], c[1], head_z])
        dist = hgt * 0.30
        eye = target + np.array([np.sin(np.radians(a.angle)) * dist,
                                 np.cos(np.radians(a.angle)) * dist, hgt * 0.04])
        fov, size = 22.0, (760, 760)
    elif a.view == 'eyes':
        target = np.array([c[0], c[1], head_z])
        dist = hgt * 0.13
        eye = target + np.array([np.sin(np.radians(a.angle)) * dist,
                                 np.cos(np.radians(a.angle)) * dist, hgt * 0.012])
        fov, size = 9.0, (760, 760)
    elif a.view == 'upper':
        target = np.array([c[0], c[1], lo[2] + hgt * 0.78])
        dist = hgt * 0.62
        eye = target + np.array([np.sin(np.radians(a.angle)) * dist,
                                 np.cos(np.radians(a.angle)) * dist, hgt * 0.10])
        fov, size = 26.0, (760, 760)
    elif a.view == 'neck':
        target = np.array([c[0], c[1], lo[2] + hgt * 0.86])
        dist = hgt * 0.34
        eye = target + np.array([np.sin(np.radians(a.angle)) * dist,
                                 np.cos(np.radians(a.angle)) * dist, hgt * 0.06])
        fov, size = 20.0, (760, 760)
    else:
        # Frame the WHOLE figure.  A 30-degree vertical field needs
        # dist >= height / (2*tan(15)) to fit a 190 cm model, and the old
        # 1.35*height cropped the head off -- which is a poor way to present a
        # character whose face is the point.
        target = np.array([c[0], c[1], lo[2] + hgt * 0.50])
        dist = hgt * 2.15
        eye = target + np.array([np.sin(np.radians(a.angle)) * dist,
                                 np.cos(np.radians(a.angle)) * dist, hgt * 0.06])
        fov, size = 30.0, (620, 980)

    size = (int(size[0] * a.scale), int(size[1] * a.scale))
    tag = ('_only_' + a.only) if a.only else ''
    p = os.path.join(OUT, '%s%s_%s_%03d.png'
                     % (prefix, tag, a.view, int(a.angle)))
    render(V, F, FM, m, p, size=size, eye=eye, target=target, fov=fov,
           cull=not a.no_cull, force_lod=a.lod, id_out=a.id_out,
           vnormals=None if a.no_shade else vnorm,
           shade=1.0 if a.no_shade else 0.55,
           alpha_test=not a.no_alpha)
    print('wrote %s' % p)


if __name__ == '__main__':
    main()
