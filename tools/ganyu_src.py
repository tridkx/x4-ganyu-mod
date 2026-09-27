# -*- coding: utf-8 -*-
"""Where the source model lives and what its Japanese material names mean.

Kept in one place so every diagnostic and both build stages agree; the Lumine
project lost a round to a diagnostic that used its own copy of this table and
silently disagreed with the builder.

This model is a *different author's* Ganyu rig (模型改造：观海 / BiliBili
观海子) than the Lumine and Emilie projects used, so the table is regenerated
from the actual material list rather than adapted -- the same discipline the
Lumine project applied to `髪` (U+9AEA) vs the previous project's `髮`
(U+9AEE).  Measured layout of the 19 source materials is in
`docs/甘雨移植进展.md` §1.
"""

import os

#: The source lives outside the workspace.  Resolved from home so the project
#: keeps working after being moved; `GANYU_SRC` overrides it when the model
#: sits somewhere else.
_DEFAULT_SRC = os.path.join(
    os.path.expanduser('~'), 'Desktop', '新建文件夹 (3)',
    'ganyu_by_原神_7381ccd84ee8763ce63b3ad638e1c49b')

SRC_DIR = os.environ.get('GANYU_SRC', _DEFAULT_SRC)
SRC_PMX = os.path.join(SRC_DIR, '甘雨.pmx')

#: The converter only accepts `[a-z0-9_]` for object names and
#: `[a-z0-9_]+\.[a-z0-9_]+` for materials, so every Chinese material name has
#: to become an ascii stem before it reaches the addon.
MAT_ASCII = {
    '面1': 'face',         # face skin
    '面2': 'face2',        # face shading overlay (coincident with 面1)
    '睫': 'eyelash',       # lashes
    '二重': 'eyelid',      # double-eyelid line
    '眉': 'brow',
    '目': 'iris',          # iris + pupil (author put them in 发.png)
    '目星': 'eyespec',     # eye highlight
    '口舌': 'mouth',       # mouth + tongue
    '齿': 'teeth',
    '白目': 'eyewhite',    # sclera
    '发': 'hair',
    '角': 'horn',          # qilin horns
    '服': 'cloth',         # dress, sleeves, bodice
    '新規': 'extra',       # waist wrap / back drape
    '腰饰': 'waist',       # waist ornaments (asymmetric, +X only)
    '神之眼': 'vision',    # Vision (elemental sigil), +X hip
    '肌': 'skin',
    '裙摆': 'skirt',       # skirt hem
    '脸红': 'blush',       # blush decal
}


def stem_of(name):
    return MAT_ASCII.get(name)


#: Materials dropped before anything is built because they are coincident
#: overlays on a base layer that would z-fight in the engine's depth buffer.
#:
#: Measured with `diag_ganyu_overlays.py`; the numbers that justify each entry
#: are recorded in `docs/甘雨移植进展.md` §2.  An unused drop is a silently
#: missing body part, so nothing goes in here on suspicion alone.
#:
#: `blush` (脸红, 60 faces on 表情.png): the atlas is alpha <= 66 over its
#: entire 4096x4096 surface and its own UVs sample a **median alpha of 0**, so
#: this is a faint blush decal that only an expression morph would ever raise.
#: An NPC never plays one; keeping it would mean shipping an ALPHA8 blend layer
#: (the only mode that can show alpha 0.26) for something invisible at NPC
#: range.
DROP_STEMS = {'blush'}

#: A triangle is thrown away when the mean alpha its three corners sample is
#: below this.  Only `cloth` and `skin` have such triangles (the author painted
#: over faces they did not want); everything else is 100% opaque.
#:
#: The threshold has to be low: MMD renders these with an alpha blend, so a face
#: at alpha 0.2 is *faintly visible*, and eating it would leave a hole in a
#: silhouette.  At 0.03 only the genuinely invisible panels go, and the
#: surviving feather faces are drawn by the material's own alpha mode.
ALPHA_KEEP = 0.03


def sample_face_alpha(alpha_img, uv_corners):
    """Mean alpha (0..1) sampled by each triangle's three UV corners.

    `alpha_img` is an (H, W) uint8 array; `uv_corners` is (F, 3, 2).  Defined
    here, next to `MAT_ASCII`, so the builder and every diagnostic measure the
    same thing -- a diagnostic that invents its own sampling has already cost
    one project a debugging round.
    """
    import numpy as np
    h, w = alpha_img.shape[:2]
    u = np.clip(uv_corners[..., 0] % 1.0, 0, 1)
    v = np.clip(uv_corners[..., 1] % 1.0, 0, 1)
    x = np.clip((u * (w - 1)).astype(int), 0, w - 1)
    y = np.clip((v * (h - 1)).astype(int), 0, h - 1)
    return alpha_img[y, x].astype(float).mean(axis=1) / 255.0
