# -*- coding: utf-8 -*-
"""Lighten thin dark lines in the cloth atlas (experiment).

Measured mechanism: decoration edges in this model are *geometry* mapped to a
**narrow strip of the atlas** (a face spanning 56 x 2 texels).  The strip holds
a dark line, so the renderer magnifies two texels into a band you read as a
black stripe on white cloth.  The line is in the source art; the magnification
is what makes it shout.

This experiment lightens atlas pixels that are (a) dark and (b) reached only by
*thin* UV strips, leaving broad dark areas -- the corset, the blue trim, the
gold motifs -- alone.  It is a look change, not a bug fix.
"""
import os
import sys
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pmx import read_pmx
from ganyu_src import SRC_DIR, SRC_PMX, stem_of

THIN_TEXELS = 8.0     # a face whose UV is thinner than this in one axis
DARK = 150.0          # what counts as a "line" pixel
LIFT = 0.55           # how far toward the local bright level


def build_thin_mask(model, size):
    """Boolean atlas mask of pixels reached by thin UV strips."""
    W, H = size
    mask = np.zeros((H, W), bool)
    fi = 0
    for mt in model['materials']:
        n = mt['face_count'] // 3
        sel = model['faces'][fi:fi + n]; fi += n
        if stem_of(mt['name']) not in ('cloth', 'skirt', 'ribbon', 'petti'):
            continue
        for f in sel:
            if len(set(f)) != 3:
                continue
            u = np.array([model['uvs'][i] for i in f])
            span = (u.max(0) - u.min(0)) * [W, H]
            if min(span) > THIN_TEXELS:
                continue
            x0 = int(np.clip(u[:, 0].min() * W, 0, W - 1))
            x1 = int(np.clip(u[:, 0].max() * W, 0, W - 1))
            y0 = int(np.clip(u[:, 1].min() * H, 0, H - 1))
            y1 = int(np.clip(u[:, 1].max() * H, 0, H - 1))
            mask[y0:y1 + 1, x0:x1 + 1] = True
    return mask


def main():
    m = read_pmx(SRC_PMX)
    p = os.path.join(SRC_DIR, 'Texture', '衣.png')
    im = Image.open(p).convert('RGBA')
    a = np.asarray(im).astype(np.float32)
    W, H = im.size
    thin = build_thin_mask(m, (W, H))
    print('thin-UV pixels: %d of %d (%.1f%%)'
          % (thin.sum(), W * H, thin.sum() * 100.0 / (W * H)))

    lum = a[..., :3].mean(2)
    target = thin & (lum < DARK)
    print('dark pixels inside those strips: %d' % target.sum())

    # local bright reference: box-blur of the not-dark pixels
    bright = lum >= DARK
    ref = np.where(bright, lum, np.nan)
    k = 9
    pad = np.pad(ref, k // 2, mode='edge')
    stack = np.stack([pad[i:i + H, j:j + W] for i in range(k)
                      for j in range(k)])
    with np.errstate(invalid='ignore'):
        local = np.nanmean(stack, axis=0)
    local = np.where(np.isnan(local), lum, local)

    out = a.copy()
    for c in range(3):
        ch = out[..., c]
        ch[target] = ch[target] + (local[target] - ch[target]) * LIFT
    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), 'RGBA').save(
        os.path.join(SRC_DIR, 'Texture', '_clean_衣.png'))
    print('wrote _clean_衣.png')


if __name__ == '__main__':
    main()
