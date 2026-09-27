# -*- coding: utf-8 -*-
"""PMX (MMD) rig -> X4 Argon/Terran female Biped mapping for 荧 (Ganyu).

Target
------
The X4 Terran female NPC macros reference `<component ref="character_argon_female_01" />`
-- the *same* shared component as the Argon women, so the skeleton, the 1100+
animations and the eye look-at all come from the identical 91-bone Biped the
previous project targeted.  Nothing about the rig changes; only which
appearance pools point at the result does.

Frame
-----
Measured, not assumed (`check_winding.py`):

    左目   - 頭       = (0, +0.68, -0.72)   eyes in front
    齿下   - 頭       = (0,  0.00, -1.08)   teeth in front
    左足先EX - 左足首D = (0, -0.43, -0.83)   toes in front
    -> -Z is "front"

    X4 `left_eye_dummy` is at +y from `Bip01 Head`, `Bip01 L Toe0` at +y from
    `Bip01 L Foot`  -> +Y is "front"

so the mapping is `(x, y, z) -> (x, -z, y)`.  Its determinant is **+1**, so the
triangle winding is kept as authored (confirmed: PMX's own vertex normals give
mean dot(face normal, vertex normal) = +0.985, positive = 1.000).

Units
-----
MMD units are arbitrary.  The global Kabsch fit solves the scale; it is *not*
inherited from the previous project (that model was 9.625 cm/unit and this one
is a different author's rig with different proportions).
"""

import numpy as np

#: source bone -> X4 bone, one to one and anatomically unambiguous.
#:
#: Spine: this rig has FOUR segments between the waist and the neck
#: (`下半身` / `上半身` / `上半身3` / `上半身2`) where X4 has three (`Spine` /
#: `Spine1` / `Spine2`), so the mapping has to stay monotonic in height or the
#: torso shears.  Measured heights above the hip joint (`左足D`, y = 10.96):
#:
#:     下半身  y = 12.10   +1.14        X4 Bip01 Spine   +12.38 cm above hip
#:     上半身  y = 12.17   +1.21        X4 Bip01 Spine1  +25.12
#:     上半身3 y = 12.70   +1.74        X4 Bip01 Spine2  +38.84
#:     上半身2 y = 13.51   +2.55        X4 Bip01 Spine2  (shares)
#:
#: MMD deliberately puts `下半身` and `上半身` at almost the same height (they
#: are two rotation centres for the same waist, one driving each half), so the
#: fit cannot match them both; what matters is that the sequence increases.
#: `上半身2` carries the most weight (1705 verts vs 847 for `上半身3`) and is
#: the chest, so it takes the top segment and `上半身3` shares it -- folding a
#: bone onto the *next* one up is exactly what the cloth chains do.
#:
#: `腰` (waist, y = 11.41) is deliberately absent: it is a pass-through
#: semi-standard bone with no skin weights of its own, and mapping it would
#: put a fourth driver inside the same 12 cm of spine.
#:
#: **Exactly one source bone may map to a given X4 bone.**  `上半身3` therefore
#: has no entry of its own and folds onto `上半身` (its parent, and therefore
#: `Bip01 Spine1`) through the normal chain-root rule.
#:
#: This was a real bug.  The first version mapped both `上半身3` and `上半身2`
#: to `Bip01 Spine2` -- "sharing", the way cloth chains fold.  But a *direct*
#: target keeps only one `(src, dst, R)` record, and which source wins depends
#: on dict iteration order; `上半身3` won, so `Bip01 Spine2`'s source position
#: became 119.0 cm instead of `上半身2`'s ~132.  Everything bound to Spine2 was
#: then translated **+13.9 cm upward** -- which is the two back ribbons, whose
#: whole chain folds onto Spine2, being hoisted from the shoulders to above the
#: head.  `_assert_unique_targets()` now makes the mistake impossible.
#:
#: Legs use the **D bones** (`左足D`/`左ひざD`/`左足首D`/`左足先EX`): the plain
#: `左足` chain is the IK chain and carries no weights, the D copies sit at
#: identical positions and carry all of them.
#:
#: `左肩C` and `左肩P` are absent on purpose: `左肩C` is coincident with `左腕`
#: and `左肩P` with `左肩`, so either would fight its twin for the same target.
CORE = {
    '下半身': 'Bip01 Spine',
    '上半身': 'Bip01 Spine1',
    '上半身2': 'Bip01 Spine2',
    '首': 'Bip01 Neck',
    '頭': 'Bip01 Head',
    '左肩': 'Bip01 L Clavicle',
    '左腕': 'Bip01 L UpperArm',
    '左ひじ': 'Bip01 L Forearm',
    '左手首': 'Bip01 L Hand',
    '右肩': 'Bip01 R Clavicle',
    '右腕': 'Bip01 R UpperArm',
    '右ひじ': 'Bip01 R Forearm',
    '右手首': 'Bip01 R Hand',
    '左足D': 'Bip01 L Thigh',
    '左ひざD': 'Bip01 L Calf',
    '左足首D': 'Bip01 L Foot',
    '左足先EX': 'Bip01 L Toe0',
    '右足D': 'Bip01 R Thigh',
    '右ひざD': 'Bip01 R Calf',
    '右足首D': 'Bip01 R Foot',
    '右足先EX': 'Bip01 R Toe0',
    '左目': 'left_eye_dummy',
    '右目': 'right_eye_dummy',
}

#: fingers: MMD `親指０/１/２` (note the FULL-WIDTH digit U+FF10 -- the rig uses
#: full-width numerals throughout), `人指/中指/薬指/小指 １/２/３`.  X4 names the
#: thumb `Finger0` + `Finger01/02` and the rest `FingerN` + `FingerN1/N2`, with
#: the middle phalanx doubling the digit number.
_FINGERS = [
    ('親指０', 'Finger0'), ('親指１', 'Finger01'), ('親指２', 'Finger02'),
    ('人指１', 'Finger1'), ('人指２', 'Finger11'), ('人指３', 'Finger12'),
    ('中指１', 'Finger2'), ('中指２', 'Finger21'), ('中指３', 'Finger22'),
    ('薬指１', 'Finger3'), ('薬指２', 'Finger31'), ('薬指３', 'Finger32'),
    ('小指１', 'Finger4'), ('小指２', 'Finger41'), ('小指３', 'Finger42'),
]
for _side, _x4s in (('左', 'L'), ('右', 'R')):
    for _mmd, _x4 in _FINGERS:
        CORE[_side + _mmd] = 'Bip01 %s %s' % (_x4s, _x4)

#: pairs the global frame fit uses: unambiguous, symmetric, spread over the
#: whole body so the fit is well conditioned in every axis.
#:
#: `上半身2` and `上半身3` both target `Bip01 Spine2`; only one of them may
#: appear here or the pair list would contain the same target twice and bias
#: the fit.
ALIGN_PAIRS = [
    ('下半身', 'Bip01 Spine'), ('上半身', 'Bip01 Spine1'),
    ('上半身2', 'Bip01 Spine2'),
    ('首', 'Bip01 Neck'), ('頭', 'Bip01 Head'),
    ('左肩', 'Bip01 L Clavicle'), ('右肩', 'Bip01 R Clavicle'),
    ('左腕', 'Bip01 L UpperArm'), ('右腕', 'Bip01 R UpperArm'),
    ('左ひじ', 'Bip01 L Forearm'), ('右ひじ', 'Bip01 R Forearm'),
    ('左足D', 'Bip01 L Thigh'), ('右足D', 'Bip01 R Thigh'),
    ('左ひざD', 'Bip01 L Calf'), ('右ひざD', 'Bip01 R Calf'),
    ('左足首D', 'Bip01 L Foot'), ('右足首D', 'Bip01 R Foot'),
]

#: X4 drives these with a look-at controller; source geometry near them is
#: re-bound to the head so a gaze change cannot swing the eyeballs out of the
#: sockets.
EYE_CONTROLLERS = {'left_eye_dummy', 'right_eye_dummy'}
HEAD_BONE = 'Bip01 Head'

#: translated but never rotated (see retarget_core.NO_ROTATE_BONES): the X4
#: foot chain is steeper than the source's, so matching axis directions would
#: tip the foot and leave the heel hanging.
NO_ROTATE_BONES = {'Bip01 L Foot', 'Bip01 R Foot', 'Bip01 L Toe0', 'Bip01 R Toe0'}

#: chain roots whose whole chain must fold onto one X4 bone rather than each
#: link finding its own nearest target.  Handled generically by chain-root
#: folding in `build_bone_map`; listed here only for the report.
CHAIN_PREFIXES = ('Q_', '左带_', '右带_', '头饰', '+Hair')


#: Neck region, in retargeted centimetres.
#:
#: The neck's lower third is authored ~58% `上半身2` and ~42% `首`.  That is
#: fine in MMD, where both are driven by the same animation, but after a
#: retarget `上半身2` -> `Bip01 Spine2` carries a **9.9 degree** fit rotation
#: while `首` -> `Bip01 Neck` carries 0.3.  The neck sits ~26 cm from the
#: Spine2 pivot, so blending the two frames sweeps the lower neck **1.6 cm
#: sideways** while the upper neck (100% Head) does not move at all -- a
#: visible kink halfway up the neck.
#:
#: Reassigning the weights *after* the fact does nothing: the vertex positions
#: are computed from the weights inside `transform()`, so by then the geometry
#: has already been swept.  The source weights have to be changed first.  That
#: was a real dead end here -- a rebind was written, ran, moved 674 vertices,
#: and changed the profile by 0.00 cm.
NECK_FIX_Z = (145.0, 157.0)
NECK_FIX_X = 7.0


def fix_neck_source_weights(m, transfer, V, verbose=True):
    """Reassign 上半身2 -> 首 for vertices in the neck, before retargeting.

    The neck loses its (small) follow-the-chest component, which does not read
    on an NPC, and gains a straight profile, which does.
    """
    G = transfer.global_only(V)
    idx = {b['name']: i for i, b in enumerate(m['bones'])}
    i_spine, i_neck = idx['上半身2'], idx['首']
    moved = 0
    for vi, wl in enumerate(m['weights']):
        if not (abs(G[vi][0]) < NECK_FIX_X
                and NECK_FIX_Z[0] < G[vi][2] < NECK_FIX_Z[1]):
            continue
        if not any(bi == i_spine for bi, _ in wl):
            continue
        m['weights'][vi] = [(i_neck if bi == i_spine else bi, w)
                            for bi, w in wl]
        moved += 1
    if verbose:
        print('  neck fix: %d vertices rebound from 上半身2 to 首 '
              '(before retargeting)' % moved)
    return moved


#: Spine chain, root first.  These are the bones whose fits disagree most
#: with the *source* rig's proportions, and they drive overlapping vertex sets.
SPINE_CHAIN = ('Bip01 Spine', 'Bip01 Spine1', 'Bip01 Spine2')

#: Chains that also get one shared offset.  **Legs only, deliberately.**
#:
#: The arms were tried and had to be reverted: their offset spread is 18.8 cm
#: (dominated by the hand, which is 15.4 cm from its X4 target), so averaging
#: moves every bone ~10 cm off its own fit and the hands come out splayed like
#: claws.  The legs' spread is 14.4 cm and the result reads fine -- the feet
#: carry no geometry as dense as fingers, and their own offset is already
#: shared with the ankle by `_feet_share_one_offset`.
#:
#: So arms keep a residual tear at the wrist and elbow.  That is the better
#: trade: a hand in the wrong place is far more visible than a seam.
LIMB_CHAINS = (
    ('Bip01 L Thigh', 'Bip01 L Calf', 'Bip01 L Foot', 'Bip01 L Toe0'),
    ('Bip01 R Thigh', 'Bip01 R Calf', 'Bip01 R Foot', 'Bip01 R Toe0'),
)


def blend_chain_offsets(transfer, chain=SPINE_CHAIN, blend=1.0, verbose=True):
    """Give a bone chain one shared translation offset.

    Why: this rig's `下半身` and `上半身` sit **0.7 cm apart** (MMD stacks its
    spine bones at the waist), while X4's `Bip01 Spine` and `Bip01 Spine1` are
    **12.8 cm apart**.  Matching each bone to its own X4 counterpart therefore
    pulls two vertex sets that are almost coincident in the source 12.8 cm
    apart -- and because the skinning blends them, every edge spanning the two
    tears.  Measured as edge stretch (result / global-fit length), 394 edges
    stretch past 2x, and 72 of those have the *same* dominant bone at both
    ends: a rigid transform cannot change a distance, so the stretch is purely
    the blend between differently-offset bones.

    Weight smoothing cannot fix it (4, 10, 20 and 40 passes all give exactly
    313 by the coarser metric), because mixing two frames 12.8 cm apart tears
    no matter how gradually the mix changes.  The offsets themselves have to
    agree.  Same reasoning as `_harmonise_head_neck` and
    `_feet_share_one_offset` in retarget_core -- this just extends it to the
    spine, which the previous projects' models did not need because their
    spine proportions happened to match.

    `blend=1.0` gives the whole chain the mean offset; lower values keep more
    of each bone's own fit.
    """
    present = [B for B in chain if B in transfer.delta and transfer.delta[B]]
    if len(present) < 2:
        return 0
    offs = np.array([transfer.delta[B][1] - transfer.delta[B][0]
                     for B in present])
    before = float(np.ptp(offs, axis=0).max())
    mean = offs.mean(axis=0)
    for B, own in zip(present, offs):
        src, dst, R = transfer.delta[B]
        transfer.delta[B] = (src, src + own * (1.0 - blend) + mean * blend, R)
    if verbose:
        after = float(np.ptp(offs * (1 - blend) + mean * blend, axis=0).max())
        print('  offsets blended on %-22s (blend=%.2f): spread %.1f -> %.1f cm'
              % (chain[0], blend, before, after))
    return len(present)


def mmd_to_blender(p):
    """MMD (x, up, back) units -> Blender/X4 (x, forward, up), same units."""
    return np.array([p[0], -p[2], p[1]], float)


def side_of(name):
    """'L' / 'R' / None, from the MMD 左 / 右 prefix."""
    if name.startswith('左'):
        return 'L'
    if name.startswith('右'):
        return 'R'
    return None


def is_direct_bone(name):
    return name in CORE


def check_core(bones):
    """Every CORE source name must exist in the rig, and no two may share a
    target.

    A silent miss sends that bone's whole vertex group to the global frame, and
    a duplicated target silently picks whichever source the iteration happens
    to reach last -- which is how the ribbons ended up above the head.  Both
    look fine in every count and only show in the geometry.
    """
    have = {b['name'] for b in bones}
    missing = sorted(set(CORE) - have)
    if missing:
        raise SystemExit('!! CORE bones absent from the rig: %s' % missing)
    _assert_unique_targets()
    return True


def _assert_unique_targets():
    seen = {}
    for src, dst in CORE.items():
        if dst in seen:
            raise SystemExit(
                '!! two source bones map to %s: %s and %s.  Only one (src, '
                'dst, R) record survives, and which one is undefined -- let '
                'the lower-priority bone fold onto its parent instead.'
                % (dst, seen[dst], src))
        seen[dst] = src


def build_bone_map(bones):
    """{source bone -> x4 bone} for **every** bone, including cloth chains.

    Direct bones map by name.  Everything else is a *folding* bone: it has no
    counterpart in the X4 rig, so it folds onto the target of the **root of its
    chain** rather than of its own position.

    Why the chain root: this rig's 80 `Q_*` skirt links hang from `下半身`, the
    38 ribbon links (`左带_*`/`右带_*`) from `上半身2`, the hair and ornaments
    (`+Hair*`, `头饰*`) from `頭`.  Folding each link by its own position sends
    the hem of a long chain to whatever core bone happens to be nearest -- the
    ribbon tips at y = 5.5 would land on the thighs and flap with every step.
    The anchor is what the author attached the chain to, so the anchor decides.
    """
    pos = {b['name']: b['position'] for b in bones}
    parent = {b['name']: (bones[b['parent']]['name'] if b['parent'] >= 0
                          else None) for b in bones}

    out = {}
    for b in bones:
        n = b['name']
        if n in CORE:
            out[n] = CORE[n]

    root_cache = {}

    def chain_root(n):
        if n in root_cache:
            return root_cache[n]
        seen = []
        cur = n
        while cur is not None and cur not in CORE:
            seen.append(cur)
            cur = parent.get(cur)
        root = cur if cur is not None else n
        for s in seen:
            root_cache[s] = root
        root_cache.setdefault(n, root)
        return root

    for b in bones:
        n = b['name']
        if n in out:
            continue
        r = chain_root(n)
        out[n] = CORE.get(r)
    return out


def report_map(bones, bone_map, x4_names, weighted):
    direct = [b for b in weighted if b in CORE]
    folded = [b for b in weighted if b not in CORE and bone_map.get(b)]
    lost = [b for b in weighted
            if not bone_map.get(b) or bone_map[b] not in x4_names]
    print('  bone map: %d weighted bones -> %d direct, %d folded, %d unmapped'
          % (len(weighted), len(direct), len(folded), len(lost)))
    if lost:
        print('   !! UNMAPPED: %s' % sorted(lost)[:20])
    return lost


class MmdAdapter:
    """Everything `retarget_core` needs to know about the PMX source."""

    name = 'mmd'
    align_pairs = ALIGN_PAIRS
    eye_controllers = EYE_CONTROLLERS
    head_bone = HEAD_BONE
    no_rotate_bones = NO_ROTATE_BONES
    fingers_bind_to_palm = True
    #: MMD rigs insert two or three twist bones (`腕捩`, `手捩`) between a joint
    #: and the next real joint, and those fold onto the *same* X4 bone as their
    #: parent.  Without walking past them the axis falls back to the parent
    #: direction and the upper arms come out rotated ~90 degrees (this was a
    #: real, measured 91-degree error in the previous project).
    walk_axis_chain = True
    eye_pairs = (('左目', 'left_eye_dummy'), ('右目', 'right_eye_dummy'))

    #: How much of the sideways travel to keep, per X4 leg bone.
    #:
    #: The two rigs disagree about which way a leg points, not just how wide
    #: the stance is.  Measured in cm (source fitted, X4 bind):
    #:
    #:            source            X4 Biped
    #:     hip    +9.15     ->     +11.61   (thigh)
    #:     knee   +7.58     ->     +14.81   (calf)
    #:     ankle  +5.84     ->     +17.70   (foot)
    #:
    #: i.e. the source's legs taper *inwards* (ankles 11.7 cm apart) while the
    #: X4 bind pose splays *outwards* (ankles 35.4 cm apart).  Matching the leg
    #: bones outright would swing the whole leg out -- the model standing in a
    #: wide A with its legs out of the skirt -- so only part of the **sideways**
    #: travel is kept (the vertical is left alone, so leg length and ground
    #: contact are untouched).
    #:
    #: The first version tapered *along* the chain -- 0.60 thigh, 0.30 calf,
    #: 0.20 foot -- and that is what made the character walk as if on a line.
    #: Tapering along the chain does not just shift the leg, it bends the bone
    #: chain inwards, and the joints end up inside the geometry they drive:
    #:
    #:     bone            X4 bind    old targets   geometry it drives
    #:     Bip01 L Thigh    +11.61      +10.62            +10.76
    #:     Bip01 L Calf     +14.81       +9.75             +8.79
    #:     Bip01 L Foot     +17.70       +8.21             +8.99
    #:     Bip01 L Toe0     +21.50       +9.05             +9.84
    #:
    #: The binding pose still looked right (the geometry was built from those
    #: very targets) but the ankle joint sat 7 cm inside the foot it carries
    #: and the whole leg was 16.4 cm wide at the ankles -- **under half** of
    #: vanilla's 39.8 cm -- so a stride put the feet on one line: the "catwalk"
    #: and the "sitting with the feet almost touching" report are the same
    #: number.  (The source is not at fault: its own leg bones sit 0.7-1.0 cm
    #: from the geometry they drive.)
    #:
    #: What the damping is really setting is the *stride width*, and that is
    #: what the "catwalk" report is about: a catwalk is two feet landing on one
    #: line.  Vanilla's women walk with their feet 39.8 cm apart; the strongly
    #: damped build put them 18.0 cm apart, i.e. inside the hips (23 cm), so
    #: every step crossed the midline.  The foot geometry follows its bone, so
    #: the first fix (0.72) only reached 30.5 cm -- still 23% narrower than
    #: vanilla and still reading as a catwalk.  The foot therefore keeps
    #: almost all of the travel (0.95) and the calf most of it (0.85), while
    #: the thigh stays at 0.60: the chain still splays monotonically outwards
    #: like a Biped chain, the ankles land within ~2 cm of vanilla's 17.70,
    #: and the mesh is rebuilt from those targets so geometry and bone agree.
    #:
    #: Still well inside the skirt: its narrowest ring (z 40-50) is 24.3 cm
    #: half-width, the wide part 36-42 cm, and this model's leg geometry is
    #: hidden above z = 42 anyway.
    #:
    #: Both the translation and the axis direction read these targets, so
    #: changing them moves position and orientation together (see
    #: `retarget_core._pair_axis.target_dir`, which is the bug that tore one
    #: earlier project's knees apart).
    LATERAL_DAMP = {
        'Bip01 L Thigh': 0.60, 'Bip01 R Thigh': 0.60,
        'Bip01 L Calf': 0.85, 'Bip01 R Calf': 0.85,
        'Bip01 L Foot': 0.95, 'Bip01 R Foot': 0.95,
        'Bip01 L Toe0': 0.95, 'Bip01 R Toe0': 0.95,
    }

    def adjust_target(self, x4_bone, src_pos, dst_pos):
        k = self.LATERAL_DAMP.get(x4_bone)
        if k is None:
            return dst_pos
        out = np.array(dst_pos, float)
        out[0] = src_pos[0] + (dst_pos[0] - src_pos[0]) * k
        return out

    def __init__(self, bone_map):
        self.bone_map = bone_map

    def to_blender(self, p):
        return mmd_to_blender(p)

    def map_bone(self, name):
        return self.bone_map.get(name)

    def is_direct_bone(self, name):
        return name in CORE

    def side_of(self, name):
        return side_of(name)
