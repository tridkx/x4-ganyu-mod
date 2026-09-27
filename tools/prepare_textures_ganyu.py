# -*- coding: utf-8 -*-
"""Convert the PMX textures to X4 DDS and write the material manifest.

Runs under the *system* Python (Blender's bundled one has no PIL).

    python tools/prepare_textures_ganyu.py

This model is a typical MMD one: **diffuse maps only** -- no normal map, no
roughness map, no metallic map.  The other two roles X4 wants are generated:

* `Normal`     -- a flat (128, 128, 255) BC5 map, same size as the diffuse; a
                  16x16 placeholder is rejected by the game and shows as a
                  missing-texture magenta.
* `Smoothness` -- a constant BC4 map per material, standing in for the MMD
                  specular exponent the source does not carry.

Two things differ from the previous project's model, and both are measured, not
assumed (`diag_alpha_faces.py` bins every triangle by the mean alpha its own
UVs sample):

* **This model's face/hair/cloth atlas carries no alpha at all** -- 颜.png,
  髪.png and 衣.png are 100% alpha 255.  So the lashes, brows and hair are
  *geometry*-shaped, not texture-masked, and shipping them BC1 is safe.  (The
  previous project's model had its masks in the texture, and dropping the alpha
  turned black stockings into solid black shells.)
* Only `skin` (肌.png: 12.7% of its surface is soft or clear) and `elem`
  (元素1.png: 88% clear -- the elemental crest) need an alpha channel.

Textures are shared: 颜.png alone feeds nine materials, so each source file is
encoded **once** and the materials point at the same DDS.  That is ~19 MB saved
against encoding per material, and it also guarantees the nine agree.
"""

import json
import os
import sys

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "x4-character-retarget", "tools"))

import bc_encode                                  # noqa: E402
from pmx import read_pmx                          # noqa: E402
from ganyu_src import (SRC_DIR, SRC_PMX, MAT_ASCII,   # noqa: E402
                       DROP_STEMS)

PROJ = os.path.dirname(HERE)
WORK = os.path.join(PROJ, 'work')
DDS_DIR = os.path.join(WORK, 'tex_out', 'mats')

COLLECTION = 'ganyu'

#: X4 wants max 1024 for a body texture; these sources are 2048^2 atlases.
MAX_SIZE = 1024

#: material stem -> smoothness (0 = matte, 1 = mirror).  MMD has no roughness
#: channel; these are read off how each part should look in a PBR renderer.
SMOOTHNESS = {
    'face': 0.22, 'face2': 0.22, 'eyewhite': 0.55, 'eyelid': 0.25,
    'eyelash': 0.35, 'mouth': 0.30, 'teeth': 0.42, 'brow': 0.25,
    'eyespec': 0.75, 'iris': 0.70,
    'hair': 0.45, 'horn': 0.35,
    'cloth': 0.18, 'skirt': 0.20, 'extra': 0.18, 'waist': 0.30,
    'vision': 0.75,
    'skin': 0.28,
}
DEFAULT_SMOOTHNESS = 0.25

#: Materials whose diffuse needs an alpha channel (BC3 instead of BC1).
#: Measured per triangle in `diag_alpha_faces.py`; every other sheet samples
#: alpha 255 across its whole surface.
#:
#: Only `skin` needs one.  `cloth` measures 9% clear too, but those faces are
#: *deleted* before the mesh is built (`build_ganyu_x4.atlas_alpha`), which
#: leaves it fully opaque -- and that matters, because X4's `blendmode` is a
#: single value: a material can be TWOSIDED or alpha-tested, never both.  The
#: dress has to be TWOSIDED (it is a one-layer sheet), so it cannot also carry
#: an alpha test.
#:
#: `blush` measures as 93% cut-out too, but it is dropped outright (see
#: `DROP_STEMS` in ganyu_src.py): a decal whose whole 4096^2 atlas is alpha
#: <= 66 is not worth an ALPHA8 blend layer on an NPC.
ALPHA_STEMS = {'skin'}

#: X4 shader + blend mode per material stem.
#:
#: The values follow the vanilla conventions read out of
#: `libraries/material_library.xml`:
#:
#:     faces / clothes / props   p1_character   NONE
#:     hair                      p1_hair        ALPHA1
#:     eye balls                 p1_eye_ball    NONE
#:     eyelashes                 xu_hair        ALPHA8
#:
#: Two deliberate departures, both inherited from the two earlier projects
#: because the reasoning is about the *engine*, not about the model:
#:
#: * the eyes keep `p1_character` rather than `p1_eye_ball`.  `p1_eye_ball`
#:   expects a specific map set (diffuse/normal/smooth plus anisotropy and
#:   detail maps) that a generated flat normal cannot fill convincingly.
#: * the eyelashes keep `p1_character` + TWOSIDED.  `xu_hair` + ALPHA8 means an
#:   8-bit alpha blend, and this model's lash atlas is 100% opaque -- the mask
#:   is in the geometry, so an alpha blend is pure risk for no gain.
#: closed solids -- one surface is all the camera can ever see
_SOLID = {'face', 'eyewhite', 'iris', 'teeth'}
#: everything else is a thin sheet and has to be drawn from both sides.
#: `horn` is a closed solid in principle, but it is 400 faces and sits on top of
#: the hair; TWOSIDED costs nothing measurable and cannot be wrong.
_THIN = ('face2', 'brow', 'eyelid', 'eyelash', 'mouth', 'eyespec', 'hair',
         'horn', 'cloth', 'skirt', 'extra', 'waist', 'vision')

SHADER = {}
for _s in _SOLID:
    SHADER[_s] = ('p1_character', 'NONE')
for _s in _THIN:
    # `TWOSIDED` is the game's own "draw both sides" switch; a vanilla
    # character cloak (`p1_char_spl_f_cloak_gen_01`) uses exactly this pair.
    # Geometry duplication was tried instead and had to be removed: the two
    # shells sit at identical depth and flicker.
    SHADER[_s] = ('p1_hair' if _s == 'hair' else 'p1_character', 'TWOSIDED')
# measured alpha surfaces take the alpha-test blend mode: they sit on a base
# layer and are never seen edge-on, so they do not need TWOSIDED
SHADER['skin'] = ('p1_character', 'ALPHA1')


def load_tex(rel, size=MAX_SIZE):
    path = os.path.join(SRC_DIR, rel.replace('\\', os.sep).replace('/', os.sep))
    if not os.path.exists(path):
        return None
    img = Image.open(path)
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    if max(img.size) > size:
        w, h = img.size
        s = size / float(max(w, h))
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))),
                         Image.LANCZOS)
    return img


def bleed_colors(img, rounds=12):
    """Dilate opaque colour into the transparent region.

    `肌.png` is 88% alpha 0, and its transparent pixels are **pure black**.
    Mip generation then averages colour blindly: a level-2 texel sitting on the
    mask boundary is a mix of skin and black, so a UV that samples anywhere
    near the edge reads as a dark grey, and the alpha it averages with it lands
    *above* the 0.5 cut-off -- so the engine's alpha test does not remove it
    either.  In game that is dark speckling along every masked edge.

    Filling the transparent texels with the nearest opaque colour makes the
    average meaningful: the colour stays skin-toned at every mip level while
    alpha keeps carrying the mask.  This is the standard fix and it has to
    happen *before* the mip chain is built.
    """
    a = np.asarray(img.convert('RGBA')).astype(np.float32)
    rgb, alpha = a[..., :3].copy(), a[..., 3]
    filled = alpha > 127
    if filled.all() or not filled.any():
        return img
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]
    for _ in range(rounds):
        acc = np.zeros_like(rgb)
        cnt = np.zeros(rgb.shape[:2], np.float32)
        for dy, dx in shifts:
            acc += np.roll(np.roll(rgb, dy, 0), dx, 1) * \
                np.roll(np.roll(filled, dy, 0), dx, 1)[..., None]
            cnt += np.roll(np.roll(filled, dy, 0), dx, 1)
        todo = (~filled) & (cnt > 0)
        if not todo.any():
            break
        rgb[todo] = acc[todo] / cnt[todo][..., None]
        filled |= todo
    out = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGBA')


def flat_normal(size):
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[..., 0] = 128
    a[..., 1] = 128
    a[..., 2] = 255
    return Image.fromarray(a, 'RGB')


#: Source atlas -> the ascii stem its DDS files are named after.
#:
#: Spelled out rather than transliterated: X4 resolves asset paths as plain
#: byte strings, and `ganyu_衣_diff.dds` is a path the game may simply fail to
#: open -- with a missing-texture magenta as the only diagnostic.  A generated
#: `isalnum()` filter is not enough, because CJK characters *are* alphanumeric.
ATLAS_NAME = {
    'tex/面.png': 'face',
    'tex/发.png': 'hair',
    'tex/服.png': 'cloth',
    'tex/肌.png': 'skin',
    'tex/表情.png': 'expr',
}


def atlas_stem(rel):
    """Source texture path -> ascii file-name stem (never non-ascii)."""
    key = rel.replace('\\', '/')
    if key in ATLAS_NAME:
        return ATLAS_NAME[key]
    base = os.path.splitext(os.path.basename(key))[0]
    out = ''.join(c if (c.isascii() and (c.isalnum() or c == '_')) else '_'
                  for c in base).strip('_')
    if not out:
        raise SystemExit('cannot derive an ascii name for texture %r' % rel)
    return out


def main():
    os.makedirs(DDS_DIR, exist_ok=True)
    m = read_pmx(SRC_PMX)

    # stem -> source texture, and the reverse grouping so shared sheets are
    # encoded once
    stem_tex = {}
    for mt in m['materials']:
        stem = MAT_ASCII.get(mt['name'])
        if stem is None:
            print('  !! material with no ascii name: %s' % mt['name'])
            continue
        if stem in DROP_STEMS:
            continue
        ti = mt['texture']
        if 0 <= ti < len(m['textures']):
            stem_tex.setdefault(stem, m['textures'][ti])

    by_tex = {}
    for stem, rel in sorted(stem_tex.items()):
        by_tex.setdefault(rel, []).append(stem)

    print('-- shared atlases --')
    for rel, stems in sorted(by_tex.items(), key=lambda z: -len(z[1])):
        print('   %-16s -> %s' % (rel, ', '.join(stems)))

    # ----------------------------------------------------------- diffuse
    tex_files = {}
    for rel, stems in sorted(by_tex.items()):
        img = load_tex(rel)
        if img is None:
            print('  !! texture missing: %s' % rel)
            continue
        # a sheet needs an alpha channel if *any* material using it does
        alpha = any(s in ALPHA_STEMS for s in stems)
        out = os.path.join(DDS_DIR, '%s_%s_diff.dds' % (COLLECTION, atlas_stem(rel)))
        # Colour-bleed *before* encoding, in both branches.  `服.png` keeps its
        # transparent texels even though the faces that sampled them are gone:
        # the mip chain still averages them into the island edges, and without
        # an alpha channel to hide the darkening that reads as dark fringing
        # along every edge of the dress.
        rgba = img.convert('RGBA')
        if rgba.getchannel('A').getextrema()[0] < 128:
            rgba = bleed_colors(rgba)
        if alpha:
            # every mip level of the BC3 chain then carries skin colour where
            # the mask is transparent instead of black
            bc_encode.encode_bc3(rgba, out)
        else:
            bc_encode.encode_bc1(rgba.convert('RGB'), out)

        nrm = os.path.join(DDS_DIR, '%s_%s_nrm.dds' % (COLLECTION, atlas_stem(rel)))
        bc_encode.encode_bc5(flat_normal(img.size), nrm)
        tex_files[rel] = {'Diffuse': out, 'Normal': nrm, 'alpha': alpha,
                          'size': img.size}
        print('  %-16s %sx%s -> BC%d%s'
              % (rel, img.size[0], img.size[1], 3 if alpha else 1,
                 '  (shared by %d)' % len(stems) if len(stems) > 1 else ''))

    # -------------------------------------------------------- per material
    manifest = {}
    for stem, rel in sorted(stem_tex.items()):
        if rel not in tex_files:
            continue
        tf = tex_files[rel]
        shader, blend = SHADER.get(stem, ('p1_character', 'NONE'))
        smooth = SMOOTHNESS.get(stem, DEFAULT_SMOOTHNESS)
        size = tf['size']
        sm = os.path.join(DDS_DIR, '%s_%s_smooth.dds' % (COLLECTION, stem))
        grey = Image.fromarray(
            np.full((size[1], size[0]), int(round(smooth * 255)), np.uint8), 'L')
        bc_encode.encode_bc4(grey, sm)

        name = '%s.%s' % (COLLECTION, stem)
        manifest[name] = {
            'x4_name': name,
            'source_texture': rel,
            'stem': stem,
            'alpha': stem in ALPHA_STEMS,
            'shader': shader,
            'blendmode': blend,
            'smoothness': smooth,
            'depth': 0.5,          # thickness for the exporter's collision box
            'textures': {'Diffuse': tf['Diffuse'], 'Normal': tf['Normal'],
                         'Smoothness': sm},
        }
        print('  %-9s %-11s %-14s %-8s smooth=%.2f'
              % (stem, os.path.basename(rel), shader, blend, smooth))

    out = os.path.join(DDS_DIR, 'manifest.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print('manifest: %s (%d materials)' % (out, len(manifest)))
    total = sum(os.path.getsize(p) for e in manifest.values()
                for p in e['textures'].values())
    print('DDS payload: %.1f MB' % (total / 1e6))


if __name__ == '__main__':
    main()
