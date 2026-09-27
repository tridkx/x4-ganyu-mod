# -*- coding: utf-8 -*-
"""Enumerate every Terran-**race** female NPC macro the mod has to rewrite.

Why not just patch the base macro
---------------------------------
`character_terran_female_cau_base_01_macro` is what the others inherit from,
so patching it looks like the whole job.  It is not: every derived macro carries
its own `<models>` block, which shadows the base's.  (Argon women -- the
previous project -- inherit their models, which is why that one only needed its
appearance pools rewritten.  Nothing about this carries across a race.)

So each macro's `<models>` is replaced individually.  That is strictly better
than rewriting the appearance pools: the pools keep selecting different macros,
so NPC identities (names, backgrounds, job titles) stay varied and only the body
changes -- and story/plot NPCs that no pool reaches are covered too.

How a macro is classified
-------------------------
By the *effective* `race` and `female` flags, resolved along the `ref` chain,
and by faction.  Matching on the name instead gets this wrong in both
directions, which is worth spelling out because the names are so suggestive:

    character_yaki_female_cau_base_01_macro         race="argon"     NOT terran
    character_yaki_female_plot_yaki_civilian_macro  race="terran"    IS terran
    character_player_custom_f_terran_cau_macro      race="terran"
                                                 faction="player"    the player
    character_scenario_combat_ter_gunner_macro   torso char_ter_m_pilot_suit_01
                                                                     male
    character_scenario_combat_ter_marine_macro   no gender in the name, female
                                                 only via its ref     female

The Yaki are an Argon-race gang, but at least one Yaki story NPC is authored as
a Terran; and the player's own custom character is `race="terran"` as well,
which would hand the player Ganyu's body -- and fight the character creator.

One more rule is needed on top of `race`: the two diplomat macros
(`character_ter_f_diplomat_01_macro`, `character_pio_f_diplomat_01_macro`)
`ref` the *Argon* helper macro, so they resolve to `race="argon"` -- yet the
Terran and Pioneer diplomat pools select them and their torsos are
`char_ter_f_diplomat_suit_01` / `char_pio_f_diplomat_suit_01`.  By race they
are Argon; by every appearance pool the player meets them as Terran faction
members.  So a macro also qualifies when it is female and **reachable from a
`terran.*` or `pioneers.*` female appearance pool**.

The pool test is what keeps that rule honest.  "Wears a Terran body" alone
would also have caught `character_yki_f_diplomat_01_macro`, which is a Yaki
diplomat that happens to be issued the same suit -- replacing it would put
Ganyu in a Yaki faction slot that has nothing to do with Terrans.  What
decides is who selects the macro, not what it wears.

Prints the list; `make_mod.py` reads the JSON this writes.
"""

import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)

import ws_paths

#: every library that can declare an NPC macro, in load order
SOURCES = [
    ws_paths.vanilla('libraries', 'character_macros.xml'),
    ws_paths.dlc_terran('libraries', 'character_macros.xml'),
]

#: factions whose members are explicitly *not* the NPCs we are replacing
EXCLUDE_FACTIONS = {'player'}


def parse_library(path):
    """{macro name: {'ref','race','female','faction','models'}}"""
    if not os.path.exists(path):
        return {}
    text = open(path, encoding='utf-8').read()
    out = {}
    for m in re.finditer(r'<macro\s+name="([^"]+)"([^>]*)>(.*?)</macro>',
                         text, re.S):
        name, attrs, body = m.group(1), m.group(2), m.group(3)
        ref = re.search(r'ref="([^"]+)"', attrs)
        ident = re.search(r'<identification\b([^>]*)/?>', body)
        entry = {'ref': ref.group(1) if ref else None}
        if ident:
            for k in ('race', 'female', 'faction'):
                a = re.search(r'%s="([^"]*)"' % k, ident.group(1))
                if a:
                    entry[k] = a.group(1)
        models = re.search(r'<models>(.*?)</models>', body, re.S)
        entry['models'] = models.group(1) if models else None
        out[name] = entry                 # later libraries override
    return out


#: appearance-pool libraries, in load order
GROUP_SOURCES = [
    ws_paths.vanilla('libraries', 'charactergroups.xml'),
] + sorted(glob.glob(ws_paths.dlc_all('*', 'libraries', 'charactergroups.xml')))


def load_pools():
    """{pool name: [body xml, ...]} across every library."""
    pools = {}
    for path in GROUP_SOURCES:
        if not os.path.exists(path):
            continue
        text = open(path, encoding='utf-8').read()
        for m in re.finditer(r'<character name="([^"]+)">(.*?)</character>',
                             text, re.S):
            pools.setdefault(m.group(1), []).append(m.group(2))
    return pools


def terran_pool_macros(pools):
    """Every macro reachable from a `terran.*` / `pioneers.*` female pool.

    Pools reference either a macro directly or another pool, so this walks
    `character=` links too -- `terran.trader.female` points at
    `terran.manager.female`, which is where the real macros are.
    """
    out = set()
    for name in pools:
        if not (name.startswith(('terran.', 'pioneers.'))
                and name.endswith('.female')):
            continue
        seen, stack = set(), [name]
        while stack:
            cur = stack.pop()
            if cur in seen or cur not in pools:
                continue
            seen.add(cur)
            for body in pools[cur]:
                out.update(re.findall(r'<select macro="([^"]+)"', body))
                stack.extend(re.findall(r'<select character="([^"]+)"', body))
    return out


def resolve(macros, name, depth=0):
    """Effective (race, female, faction) for a macro, following `ref`."""
    e = macros.get(name)
    if e is None or depth > 8:
        return {}
    out = {}
    if e.get('ref'):
        out.update(resolve(macros, e['ref'], depth + 1))
    for k in ('race', 'female', 'faction'):
        if k in e:
            out[k] = e[k]                 # the child's own value wins
    return out


def main():
    macros = {}
    for p in SOURCES:
        macros.update(parse_library(p))
    print('macros loaded: %d' % len(macros))

    pools = load_pools()
    reachable = terran_pool_macros(pools)
    print('appearance pools: %d, of which reachable from a Terran/Pioneer '
          'female pool: %d macros' % (len(pools), len(reachable)))

    targets, skipped = [], []
    for name, e in sorted(macros.items()):
        eff = resolve(macros, name)
        if eff.get('female') != 'true':
            continue
        by_race = eff.get('race') == 'terran'
        by_pool = name in reachable
        if not (by_race or by_pool):
            continue
        if not e.get('models'):
            skipped.append((name, 'inherits models from %s' % e.get('ref')))
            continue
        if eff.get('faction') in EXCLUDE_FACTIONS:
            skipped.append((name, 'faction=%s' % eff.get('faction')))
            continue
        why = ('race=terran' if by_race else 'terran pool')
        if by_race and by_pool:
            why = 'race=terran + pool'
        targets.append((name, why))

    print('\n-- patched (%d) --' % len(targets))
    for n, why in targets:
        print('   %-58s %s' % (n, why))
    targets = [n for n, _ in targets]

    print('\n-- skipped (%d) --' % len(skipped))
    for n, why in skipped:
        print('   %-58s %s' % (n, why))

    # sanity: nothing male may have slipped in
    bad = []
    for n in targets:
        torsos = re.findall(r'ref="([^"]*bodies[^"]*)"', macros[n]['models'])
        male = [t for t in torsos if re.search(r'/[a-z_]*_m_', t)]
        if male:
            bad.append((n, male))
    if bad:
        print('\n!! targets whose torso assets look male:')
        for n, t in bad:
            print('   %-58s %s' % (n, [x.split('/')[-1] for x in t]))

    out = os.path.join(PROJ, 'work', 'terran_female_macros.json')
    with open(out, 'w', encoding='utf-8') as fh:
        json.dump(sorted(targets), fh, indent=1)
    print('\nwrote %s (%d names)' % (out, len(targets)))


if __name__ == '__main__':
    main()
