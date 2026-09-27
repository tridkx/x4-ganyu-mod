# -*- coding: utf-8 -*-
"""Where the workspace's shared pieces live, resolved from this file.

Hard-coding `D:\\dsh-x4\\work` broke the moment the workspace's `work/` was
reorganised, so the roots are derived from this file's own location and the
few layouts that have actually existed are probed in order:

    <ws>/x4-character-retarget/work    the workspace work tree (current)
    <ws>/work                          where it used to be

`<ws>` is two levels up from this file (`<ws>/ganyu/tools/ws_paths.py`), so
the project keeps working after being moved, renamed or copied to another
machine.  Every script that needs the unpacked game libraries, the converter
package or the retarget tools should import from here rather than spell the
paths out.
"""

import glob
import os

TOOLS = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(TOOLS)                       # <ws>/ganyu
WS = os.path.dirname(PROJ)                          # <ws>

SHARED = os.path.join(WS, 'shared')

#: the converter addon, unpacked next to its zip
CONVERTER = None
for _d in sorted(glob.glob(os.path.join(SHARED, 'X4CharacterConverter*'))):
    if os.path.isdir(_d) and os.path.exists(
            os.path.join(_d, 'X4CharacterConverter', '__init__.py')):
        CONVERTER = _d
        break

#: the retarget toolkit (sibling project)
RETARGET_TOOLS = os.path.join(WS, 'x4-character-retarget', 'tools')

#: unpacked game root used as the converter's `data_root`
X4_ROOT = os.path.join(SHARED, 'x4root')


def _find_gamedata():
    """The work tree holding the unpacked `vanilla/` + `dlc_all/` libraries."""
    candidates = [
        os.path.join(WS, 'x4-character-retarget', 'work'),
        os.path.join(WS, 'work'),
        os.path.join(PROJ, 'work'),
    ]
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'vanilla', 'libraries')):
            return c
    return candidates[0]


#: unpacked game libraries land here (`vanilla/`, `dlc_all/`, `dlc_terran/`)
GAMEDATA = _find_gamedata()


def vanilla(*parts):
    return os.path.join(GAMEDATA, 'vanilla', *parts)


def dlc_all(*parts):
    return os.path.join(GAMEDATA, 'dlc_all', *parts)


def dlc_terran(*parts):
    return os.path.join(GAMEDATA, 'dlc_terran', *parts)
