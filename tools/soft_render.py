# -*- coding: utf-8 -*-
"""A small z-buffered software renderer.

Why not Blender: verifying a coordinate fix means looking at a picture, and a
Blender round trip is ~40 s of process startup plus a scene to maintain.  This
rasterises a few thousand triangles in well under a second, so a mapping change
can be *looked at* immediately.

It is deliberately plain -- flat shading, one directional light, optional
texture sampling -- because it only has to answer "is the model the right way
round, the right way up, and are the parts where they should be", not to
predict the game's shading.
"""

import numpy as np
from PIL import Image


def _normalise(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    f = _normalise(target - eye)
    up = np.asarray(up, float)
    s = np.cross(f, up)
    if np.linalg.norm(s) < 1e-9:
        s = np.cross(f, (0.0, 1.0, 0.0))
    s = _normalise(s)
    u = np.cross(s, f)
    return eye, np.stack([s, u, f])


def render(verts, faces, face_colors, path, size=(600, 900), eye=None,
           target=None, fov=32.0, bg=(30, 32, 38), light=(0.35, -0.75, 0.55),
           ortho=False, shade=0.55, flip_y=True):
    """Rasterise `faces` of `verts` (Z-up, cm) to `path`.

    face_colors : (F, 3) float 0..1, or (F, 4) with alpha to skip faces
    flip_y      : Blender/X4 `+Y` is *forward*, so the camera sits on +Y and
                  the horizontal screen axis has to be mirrored to avoid a
                  left/right flip in the output
    """
    V = np.asarray(verts, float)
    F = np.asarray(faces, np.int64)
    C = np.asarray(face_colors, float)
    W, H = size

    if eye is None or target is None:
        c = V.mean(0)
        r = float(np.abs(V - c).max()) * 1.6
        target = c
        eye = c + np.array([0.0, r * 2.2, r * 0.35])

    eye, basis = look_at(eye, target)
    rel = V - eye
    cam = rel @ basis.T                      # x right, y up, z forward
    z = cam[:, 2].copy()
    z[z < 1e-3] = 1e-3

    scale = (H * 0.5) / np.tan(np.radians(fov) * 0.5)
    sx = cam[:, 0] / z * scale
    sy = cam[:, 1] / z * scale
    if flip_y:
        sx = -sx
    px = sx + W * 0.5
    py = H * 0.5 - sy

    a, b, c = F[:, 0], F[:, 1], F[:, 2]
    depth = np.minimum(np.minimum(z[a], z[b]), z[c])
    order = np.argsort(-depth)               # far to near

    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.array(bg, np.float32) / 255.0
    zbuf = np.full((H, W), np.inf, np.float32)

    # per-face normal for shading, in world space
    fn = _normalise(np.cross(V[b] - V[a], V[c] - V[a]))
    light_dir = _normalise(np.array(light, float))

    for fi in order:
        i0, i1, i2 = F[fi]
        col = C[fi]
        if col.shape[0] == 4 and col[3] <= 0.01:
            continue
        x0, y0, z0 = px[i0], py[i0], z[i0]
        x1, y1, z1 = px[i1], py[i1], z[i1]
        x2, y2, z2 = px[i2], py[i2], z[i2]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))) + 1, W)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))) + 1, H)
        if minx >= maxx or miny >= maxy:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        xs = np.arange(minx, maxx) + 0.5
        ys = np.arange(miny, maxy) + 0.5
        gx, gy = np.meshgrid(xs, ys)
        w0 = ((x1 - gx) * (y2 - gy) - (x2 - gx) * (y1 - gy)) / area
        w1 = ((x2 - gx) * (y0 - gy) - (x0 - gx) * (y2 - gy)) / area
        w2 = 1.0 - w0 - w1
        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue
        zz = w0 * z0 + w1 * z1 + w2 * z2
        sub = zbuf[miny:maxy, minx:maxx]
        hit = mask & (zz < sub)
        if not hit.any():
            continue
        sub[hit] = zz[hit]
        # two-sided lighting: a mirrored shell must not read as black
        nd = abs(float(np.dot(fn[fi], light_dir)))
        lit = shade + (1.0 - shade) * nd
        img[miny:maxy, minx:maxx][hit] = np.clip(col[:3] * lit, 0, 1)

    Image.fromarray((img * 255).astype(np.uint8)).save(path)
    return path


def orbit_views(verts, faces, face_colors, out_dir, prefix, angles=(0, 90, 180, 270),
                size=(600, 900), **kw):
    """Render around the model.  Angle 0 = looking at the model's front."""
    V = np.asarray(verts, float)
    c = V.mean(0)
    r = float(np.abs(V - c).max()) * 2.1
    made = []
    for ang in angles:
        t = np.radians(ang)
        # +Y is forward, so the camera starts on +Y and orbits around Z
        eye = c + np.array([r * np.sin(t), r * np.cos(t), r * 0.30])
        p = '%s/%s_%03d.png' % (out_dir, prefix, ang)
        render(V, faces, face_colors, p, size=size, eye=eye, target=c, **kw)
        made.append(p)
    return made
