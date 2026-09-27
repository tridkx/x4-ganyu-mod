# -*- coding: utf-8 -*-
"""Assemble the Ganyu X4 mod from the two exported packages.

    python tools/make_mod.py --mode add       # -> work/x4_ganyu_argon_add
    python tools/make_mod.py --mode replace   # -> work/x4_ganyu_argon_replace
    # then pack:  XRCatTool.exe -in work/x4_ganyu_argon_<mode> \
    #                          -out work/x4_ganyu_argon_<mode>/ext_01.cat

`--mode` picks the shape, and both are shipped:

* **add** (default) -- one new macro plus one `<select>` per pool.  Every
  vanilla macro and every vanilla pool entry is left exactly as it is, so
  Ganyu is one candidate among N: she takes 1/(N+1) of the spawns in that job
  and the rest of the women are untouched, names and voices included.
  Story/plot NPCs, which no pool reaches, keep their vanilla appearance.
* **replace** -- every Argon female macro named in
  `work/argon_female_macros.json` gets its `<models>` rewritten, so *every*
  Argon woman is her, story NPCs included.

The two shapes produce different mod trees, hence the two directories: they
are meant to be installed one at a time (both rewrite the same asset paths),
so keeping them apart avoids one silently overwriting the other.

Why the race is fixed to Argon here
-----------------------------------
The Lumine project parameterised `--race` because it shipped a Terran *and* an
Argon build.  Ganyu targets the Argon women only, so the table has one entry
and `--race` is kept purely so the pool/macro machinery below stays identical
to the project it came from (and so a Terran build is a one-entry change, plus
its macro list).

`tools/find_terran_female_macros.py` is what enumerates the macros `--mode
replace` consumes -- run with `--race argon` it produced the 122-entry
`work/argon_female_macros.json`, which is also the evidence that the pool list
of `--mode add` is complete.

The base macro's `facemods`/`eyepositions` ride along untouched, so the
face-shaping system still runs; it simply has no effect on a head mesh that
carries none of the blend bones.
"""

import glob
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
WORK = os.path.join(PROJ, 'work')
PKG = os.path.join(WORK, "x4cc_pkg")
MOD_ID = "x4_ganyu_mod"
DDS_DIR = os.path.join(WORK, 'tex_out', 'mats')

#: The race this mod targets.  Argon women own `character_argon_female_01`
#: (the shared component with the skeleton and the 1100+ animations), so the
#: macro's `ref` is what makes her identifiable as one of them, and
#: `pool_prefixes` is which appearance pools select her.
#:
#: `macro_list` is the output of `find_terran_female_macros.py --race argon`:
#: 122 macros whose *effective* race (resolved along the `ref` chain) is argon
#: and whose identification says female.  Names are not the criterion -- two
#: `character_yaki_female_*` macros are argon, and the argon diplomat macros
#: are what the Argon pools actually select.
RACES = {
    'argon': {
        'race': 'argon',
        'base_macro': 'character_argon_female_cau_base_01_macro',
        'macro': 'character_argon_female_ganyu_macro',
        'asset_dir': 'argon',
        'pool_prefixes': ('argon.',),
        'macro_list': os.path.join(WORK, 'argon_female_macros.json'),
        'label': 'Argon',
        'label_cn': '阿贡（Argon）',
    },
}

#: resolved by main() from --race / --mode; every writer below reads these
RACE = None
MODE = 'add'
MOD = None
ASSET_BASE = None
MACRO_NAME = None
MACRO_HEAD = None
MACRO_BODY = None

#: unpacked game libraries, in load order -- the same roots
#: find_terran_female_macros.py uses, so both tools see one merged tree.
#: Resolved by tools/ws_paths.py rather than spelled out: the workspace's
#: work tree has already moved once.
import ws_paths
GAME = ws_paths.GAMEDATA
MACRO_SOURCES = [
    os.path.join(GAME, 'vanilla', 'libraries', 'character_macros.xml'),
] + sorted(glob.glob(os.path.join(GAME, 'dlc_all', '*', 'libraries',
                                  'character_macros.xml')))
POOL_SOURCES = [
    os.path.join(GAME, 'vanilla', 'libraries', 'charactergroups.xml'),
] + sorted(glob.glob(os.path.join(GAME, 'dlc_all', '*', 'libraries',
                                  'charactergroups.xml')))

COMPONENT = 'character_argon_female_01'

#: every pool ends in this; the race prefix comes from RACES
POOL_SUFFIX = '.female'


def pool_sources():
    """The pool libraries in load order -- `verify_mod.py` re-reads them to
    prove the add-mode patch covers exactly the pools that were discovered."""
    return POOL_SOURCES


def configure(race, mode, tag):
    """Bind the module-level names the writers use.  Called once by main()."""
    global RACE, MODE, MOD, ASSET_BASE, MACRO_NAME, MACRO_HEAD, MACRO_BODY
    RACE = RACES[race]
    MODE = mode
    MOD = os.path.join(WORK, 'x4_ganyu_%s_%s' % (race, tag))
    ASSET_BASE = ('extensions/%s/assets/characters/%s/ganyu'
                  % (MOD_ID, RACE['asset_dir']))
    MACRO_NAME = RACE['macro']
    MACRO_HEAD = ASSET_BASE + '/heads/ganyu_head'
    MACRO_BODY = ASSET_BASE + '/bodies/ganyu_body'
    return MOD


def read(path):
    with open(path, encoding='utf-8') as fh:
        return fh.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(text)
    return path


def merge_assets():
    """Copy both packages' assets into the mod tree under one directory."""
    dst_root = os.path.join(MOD, 'assets', 'characters', RACE['asset_dir'],
                            'ganyu')
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.makedirs(dst_root, exist_ok=True)

    xacs, textures = [], {}
    XAC_DEST = {'ganyu_head.xac': 'heads', 'ganyu_body.xac': 'bodies'}
    for pkg in ('ganyu_head', 'ganyu_body'):
        src = os.path.join(PKG, pkg, 'assets', 'characters', 'mycharacters')
        for root, _dirs, files in os.walk(src):
            for fn in files:
                sp = os.path.join(root, fn)
                if fn.endswith('.xac'):
                    sub = (XAC_DEST.get(fn)
                           or ('heads' if 'head' in fn.lower() else 'bodies'))
                    dp = os.path.join(dst_root, sub, fn)
                else:
                    dp = os.path.join(dst_root, 'textures', fn)
                os.makedirs(os.path.dirname(dp), exist_ok=True)
                shutil.copyfile(sp, dp)
                if fn.endswith('.xac'):
                    xacs.append(dp)
                else:
                    textures[fn] = dp
    return xacs, textures


def merge_material_library(textures):
    """Merge the two generated material libraries, then repair them.

    Two repairs, both because the converter's `build_material_library()` writes
    a fixed template rather than reading the Blender materials:

    * every texture path is the literal `PUT_YOUR_TEXTURE_PATH_HERE`;
    * **every material gets `shader="p1_character" blendmode="NONE"`**, no
      matter what `x4cc_shader` / `x4cc_blendmode` the material carries.  The
      previous project set `ALPHA1` on its stockings and shipped `NONE` -- its
      black stockings were never transparent in game.  So the shader and blend
      mode are taken from the manifest here, which is the only place they can
      come from.
    """
    manifest = json.load(open(os.path.join(DDS_DIR, 'manifest.json'),
                              encoding='utf-8'))
    collections = {}
    for pkg in ('ganyu_head', 'ganyu_body'):
        p = os.path.join(PKG, pkg, 'libraries', 'material_library.xml')
        if not os.path.exists(p):
            continue
        for m in re.finditer(r'<collection name="([^"]+)">(.*?)</collection>',
                             read(p), re.S):
            name, inner = m.group(1), m.group(2)
            block = collections.setdefault(name, {})
            for mm in re.finditer(r'<material name="([^"]+)".*?</material>',
                                  inner, re.S):
                block[mm.group(1)] = mm.group(0)

    if not collections:
        raise RuntimeError('no material collections found in the exported '
                           'packages -- did build_ganyu_mod.py run?')

    def fix_path(match):
        return ('value="%s\\textures\\%s"'
                % (ASSET_BASE.replace('/', '\\'), match.group(1)))

    out = ["<?xml version='1.0' encoding='utf-8'?>", '<diff>',
           '  <add sel="/materiallibrary" pos="prepend">']
    total = 0
    unknown = []
    for coll, mats in sorted(collections.items()):
        out.append('    <collection name="%s">' % coll)
        for mname in sorted(mats):
            block = mats[mname]
            entry = manifest.get('%s.%s' % (coll, mname))
            if entry is None:
                unknown.append('%s.%s' % (coll, mname))
            else:
                block = re.sub(r'shader="[^"]*"',
                               'shader="%s"' % entry['shader'], block)
                block = re.sub(r'blendmode="[^"]*"',
                               'blendmode="%s"' % entry['blendmode'], block)
            block = re.sub(r'value="PUT_YOUR_TEXTURE_PATH_HERE\\([^"]+)"',
                           fix_path, block)
            out.append('      ' + block.strip())
            total += 1
        out.append('    </collection>')
    out += ['  </add>', '</diff>', '']
    if unknown:
        raise RuntimeError('no manifest entry for %s -- the shader and blend '
                           'mode would silently fall back to the converter '
                           'defaults' % unknown)
    write(os.path.join(MOD, 'libraries', 'material_library.xml'),
          '\n'.join(out))
    return total


def merge_library(sources):
    """{name: body} for every `<macro>` / `<character>` in load order."""
    out = {}
    for path in sources:
        if not os.path.exists(path):
            continue
        text = read(path)
        for tag in ('macro', 'character'):
            for m in re.finditer(r'<%s\s+name="([^"]+)"[^>]*>(.*?)</%s>'
                                 % (tag, tag), text, re.S):
                out[m.group(1)] = m.group(2)      # later libraries override
    return out


def female_pools_with_macros():
    """Every appearance pool that may spawn a woman of the target race.

    Two filters, both deliberate:

    * the pool is named `<race>.*` + `.female` -- the faction's own pools.  For
      Terran that also means `pioneers.*`, the second faction that uses Terran
      bodies.  The pool *name* is the criterion, not the race of the macros
      inside it: the two Terran diplomat pools select macros that resolve to
      `race="argon"`;
    * the pool must **list macros directly**.  A pool that only contains
      `<select character="...">` links is a router (`argon.trader.female` ->
      `argon.civilian.female`); adding to it would be dead weight, and the pool
      it routes to is in this list anyway.

    Discovered rather than hard-coded so a DLC that adds a pool is picked up.
    """
    pools = merge_library(POOL_SOURCES)
    out = []
    for name in sorted(pools):
        if not (name.startswith(RACE['pool_prefixes'])
                and name.endswith(POOL_SUFFIX)):
            continue
        if re.search(r'<select\s+macro="', pools[name]):
            out.append(name)
    return out


def write_content():
    race = RACE['label']
    if MODE == 'replace':
        desc = ('Replaces every %s female NPC body with Ganyu from Genshin '
                'Impact.' % race)
    else:
        desc = ('Adds Ganyu from Genshin Impact to the %s female NPC '
                'appearance pools as one possible body among the existing '
                'ones.' % race)
    cdesc = ('把《原神》的甘雨%s%s女性 NPC 的外观%s。'
             % ('替换全部' if MODE == 'replace' else '加入',
                RACE['label_cn'],
                '（所有该种族女性都是甘雨）' if MODE == 'replace'
                else '池：她是随机出现的其中一种，其余女性保持原样'))
    text = '''<?xml version="1.0" encoding="utf-8"?>
<content id="{id}" name="Ganyu (Genshin Impact)" version="100" date="2026-09-27" save="0"
         description="{desc}">
  <text language="7"  name="Ganyu (Genshin Impact)" description="{desc}"/>
  <text language="44" name="Ganyu (Genshin Impact)" description="{desc}"/>
  <text language="86" name="甘雨 (原神)" description="{cdesc}"/>
</content>
'''.format(id=MOD_ID, desc=desc, cdesc=cdesc)
    return write(os.path.join(MOD, 'content.xml'), text)


def write_character_macros():
    """`--mode add`: one new macro and no <replace> at all.

    The macro is generated here rather than inheriting the base macro's
    `<models>` because it *must* differ from the base in exactly that block;
    everything else (`identification`, `eyepositions`, `facemods`, `bonemods`)
    is inherited by the `ref`.

    `props` (the random hairstyle) is set to `none`: Ganyu's hair is baked
    into the head asset, so an inherited vanilla hairdo would be drawn on top
    of it.  `props2` is stated for symmetry with what the exporter writes.
    """
    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<diff>',
        '',
        '  <!-- Ganyu as one more %s-female body.' % RACE['label'],
        '',
        '       This is an <add>, not a <replace>: every vanilla macro is left',
        '       exactly as it is, so she shows up in the pools below only as',
        '       one candidate among the existing ones.  Story and plot NPCs',
        '       (which no pool reaches) keep their vanilla appearance.',
        '',
        "       ref'ing the cau base is what makes her selectable: the",
        '       identification inherited through it says race="%s"'
        % RACE['race'],
        '       female="true", so a pool that asked for one of this race\'s',
        '       women accepts this macro. -->',
        '  <add sel="/macros">',
        '    <macro name="%s" class="npc"' % MACRO_NAME,
        '           ref="%s">' % RACE['base_macro'],
        '      <component ref="%s" />' % COMPONENT,
        '      <properties>',
        '        <models>',
        '          <model type="head"  ref="%s" />' % MACRO_HEAD,
        '          <model type="torso" ref="%s" />' % MACRO_BODY,
        '          <model type="props"  ref="none" />',
        '          <model type="props2" ref="none" />',
        '        </models>',
        '      </properties>',
        '    </macro>',
        '  </add>',
        '',
        '</diff>',
        '',
    ]
    path = write(os.path.join(MOD, 'libraries', 'character_macros.xml'),
                 '\n'.join(lines))
    print('character_macros: +1 macro (%s), 0 vanilla macros touched'
          % MACRO_NAME)
    return path


def load_macro_list(path):
    """The macro names out of either shape `find_terran_female_macros.py` writes.

    It has produced two: a bare list of names (the Terran run) and a dict that
    carries the evidence for every decision -- `{'targets': [{'name': ...,
    'why': ..., 'types': [...]}, ...], 'skipped': [...], 'mismatch': [...]}` --
    which is what the argon run wrote.  Reading only the first shape would have
    silently replaced **zero** macros: a mod that installs, loads, and changes
    nothing, with no error anywhere.  Both are accepted, and an empty result is
    a hard failure rather than an empty `<diff>`.
    """
    data = json.load(open(path, encoding='utf-8'))
    if isinstance(data, dict):
        entries = data.get('targets')
        if entries is None:
            raise RuntimeError('%s is a dict without a "targets" key -- keys '
                               'are %s' % (path, sorted(data)))
        names = [e['name'] if isinstance(e, dict) else e for e in entries]
    elif isinstance(data, list):
        names = [e['name'] if isinstance(e, dict) else e for e in data]
    else:
        raise RuntimeError('%s is neither a list nor a dict' % path)
    if not names:
        raise RuntimeError('%s holds no macros -- refusing to write an empty '
                           'replace pass' % path)
    return names


def write_character_macros_replace():
    """`--mode replace`: rewrite the <models> of every female macro.

    Reads the macro list produced by `find_terran_female_macros.py`, which is
    also the evidence that the list is complete (effective `race` + `female`,
    plus every macro reachable from the race's own female pools).  Only
    `<models>` is touched, so `identification` / `facemods` / `bonemods` keep
    working as before.
    """
    path_in = RACE['macro_list']
    if not os.path.exists(path_in):
        raise RuntimeError('%s missing -- --mode replace needs the macro list; '
                           'run tools/find_terran_female_macros.py --race %s '
                           'first' % (path_in, RACE['race']))
    macros = load_macro_list(path_in)

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<diff>',
        '',
        '  <!-- TOTAL CONVERSION: every %s-female NPC macro has its' % RACE['label'],
        '       <models> swapped for Ganyu, so every woman of this race is',
        '       her, including the story/plot NPCs, which no appearance pool',
        '       reaches.  The vanilla macros are not deleted, only re-pointed,',
        '       so identification/facemods/bonemods still run.',
        '',
        '       Each macro is patched individually because the derived macros',
        '       restate <models>, shadowing the base macro they ref. -->',
        '',
    ]
    for name in macros:
        lines += [
            "  <replace sel=\"/macros/macro[@name='%s']/properties/models\">" % name,
            '    <models>',
            '      <model type="head"  ref="%s" />' % MACRO_HEAD,
            '      <model type="torso" ref="%s" />' % MACRO_BODY,
            '      <model type="props" ref="none" />',
            '      <model type="props2" ref="none" />',
            '    </models>',
            '  </replace>',
            '',
        ]
    # A convenience macro, so the model can also be spawned by name with
    # `add_npc` / the debug character test.  In add mode this macro is the
    # whole mod; here the pools are not touched, so it exists purely as a
    # handle.  It lives in this same file rather than its own: an extra
    # `libraries/*.xml` is only loaded if the game happens to scan the
    # directory, and a missing entry here fails silently.
    lines += [
        '  <!-- No pool selects it: a handle for spawning her directly. -->',
        '  <add sel="/macros">',
        '    <macro name="%s" class="npc"' % MACRO_NAME,
        '           ref="%s">' % RACE['base_macro'],
        '      <component ref="%s" />' % COMPONENT,
        '      <properties>',
        '        <models>',
        '          <model type="head"  ref="%s" />' % MACRO_HEAD,
        '          <model type="torso" ref="%s" />' % MACRO_BODY,
        '          <model type="props"  ref="none" />',
        '          <model type="props2" ref="none" />',
        '        </models>',
        '      </properties>',
        '    </macro>',
        '  </add>',
        '',
        '</diff>',
        '',
    ]
    path = write(os.path.join(MOD, 'libraries', 'character_macros.xml'),
                 '\n'.join(lines))
    print('character_macros: %d macros replaced (+1 spawnable macro)'
          % len(macros))
    return path


def write_character_groups(weight=1):
    """Append Ganyu to every female pool that lists macros.

    `weight` = how many times her `<select>` is emitted per pool.  X4 picks
    uniformly among a pool's entries, so one line in a pool of three vanilla
    candidates gives her 1/4 of that job's spawns; emitting it twice gives
    2/5.  Raise it (or edit the file) to make her more common.

    The alternative shape -- rewriting the macros instead of joining the pools
    -- is `write_character_macros_replace()`, selected by `--mode replace`.
    """
    pools = female_pools_with_macros()
    if not pools:
        raise RuntimeError('no %s female pool found -- are the libraries '
                           'unpacked under %s?' % (RACE['label'], GAME))

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<diff>',
        '',
        '  <!-- Ganyu joins the pool.  Only pools that list macros directly',
        '       are touched: a pool that is pure select-character routing',
        '       reaches one of these anyway.',
        '',
        '       One line = one share.  These pools hold three vanilla',
        '       candidates, so a single line makes her 1 in 4 of that job;',
        '       the vanilla macros stay in place, so the other 3 in 4 are',
        '       unchanged, names and voices included. -->',
        '',
    ]
    for name in pools:
        lines.append("  <add sel=\"/characters/character[@name='%s']\">" % name)
        for _ in range(weight):
            lines.append('    <select macro="%s" />' % MACRO_NAME)
        lines.append('  </add>')
        lines.append('')
    lines += ['</diff>', '']
    path = write(os.path.join(MOD, 'libraries', 'charactergroups.xml'),
                 '\n'.join(lines))
    print('charactergroups: %s added to %d pools (x%d each)'
          % (MACRO_NAME, len(pools), weight))
    for name in pools:
        print('   %s' % name)
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--race', choices=sorted(RACES), default='argon',
                    help='which race\'s women she joins (default: argon; it is '
                         'the only race this project builds for)')
    ap.add_argument('--mode', choices=('add', 'replace'), default='add',
                    help="add = one more pool option (default); "
                         "replace = every woman of that race")
    ap.add_argument('--weight', type=int, default=1, metavar='N',
                    help='add mode only: how many times Ganyu is listed per '
                         'pool; the vanilla pools hold 3 candidates, so '
                         "1 = 25%% of that job's spawns, 2 = 40%% (default: 1)")
    ap.add_argument('--out', default=None,
                    help='override the output directory (default: '
                         'work/x4_ganyu_<race>_<mode>)')
    args = ap.parse_args()

    configure(args.race, args.mode, args.mode)
    if args.out:
        globals()['MOD'] = os.path.abspath(args.out)

    # Check the macro list *before* touching the output directory: a missing
    # list used to fail halfway through, leaving a half-built tree behind that
    # looks like a real artifact (it has content.xml and assets, but no macros
    # and no .cat).
    if MODE == 'replace' and not os.path.exists(RACE['macro_list']):
        print('%s missing: --mode replace rewrites the macro list it names.\n'
              'Generate it with:\n'
              '  python tools/find_terran_female_macros.py --race %s '
              '--out %s'
              % (RACE['macro_list'], RACE['race'], RACE['macro_list']),
              file=sys.stderr)
        return 2

    if os.path.exists(MOD):
        shutil.rmtree(MOD)

    xacs, textures = merge_assets()
    print('assets copied : %d xac, %d textures' % (len(xacs), len(textures)))
    for x in sorted(xacs):
        print('   %s (%d KB)' % (os.path.relpath(x, MOD),
                                 os.path.getsize(x) // 1024))

    n = merge_material_library(textures)
    print('materials     : %d merged into one collection' % n)
    write_content()
    if MODE == 'replace':
        write_character_macros_replace()
    else:
        write_character_macros()
        write_character_groups(args.weight)
    print('xml written   : content.xml + macros%s'
          % ('' if MODE == 'replace' else ' + pools'))

    total = sum(os.path.getsize(os.path.join(r, f))
                for r, _d, fs in os.walk(MOD) for f in fs)
    print('mod size      : %.1f MB -> %s' % (total / 1e6, MOD))


if __name__ == '__main__':
    sys.exit(main())
