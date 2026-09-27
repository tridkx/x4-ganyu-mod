#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""查一组角色宏被"谁"引用 —— 判断一次替换会动到哪些 NPC。

为什么需要：改一个角色宏（或往外观池里加/换宏）之前，得先知道谁在用这个名字。
"被外观池 <select> 选中" = 普通随机 NPC；"被 md/*.xml 里 create_cue_actor 引用"
= 剧情/任务 NPC（唯一角色，不走池子）。名字看不出来，只能查数据。

    python tools/check_macro_refs.py                      # 默认读 replace 版的宏清单
    python tools/check_macro_refs.py --mod work/x4_ganyu_terran_add
    python tools/check_macro_refs.py --macro character_terran_female_plot_secret_service_contact_macro

实现要点（踩过的坑，别退回旧写法）：
  * 游戏根目录的 0N.cat 只装本体，DLC 在 extensions/<dlc>/ext_0N.cat —— 两个都要索引；
  * **不要按路径去重**：DLC 的 libraries/charactergroups.xml 是 diff 文件，
    与本体同名，去重会把本体那份顶掉，于是所有池引用都查不到（v1/v2 的假"死宏"）。
"""

import argparse
import collections
import glob
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
WS = os.path.dirname(PROJ)
sys.path.insert(0, os.path.join(WS, 'x4-anim-preview', 'tools'))
from pathlib import Path  # noqa: E402

from x4game import _parse_cat, find_game_root  # noqa: E402

#: 会引用角色宏的目录（其余是模型/贴图/音效）
SCAN = ('libraries/', 'md/', 'maps/', 'index/', 'aiscripts/', 'cutscenes/', 't/')
NAME_RE = re.compile(r'character_[a-z0-9_]*_macro')


def default_macros():
    """replace 版改写的宏清单（从产物 XML 里读，不写死）。"""
    xml = os.path.join(PROJ, 'work', 'x4_ganyu_terran_replace', 'libraries', 'character_macros.xml')
    if not os.path.exists(xml):
        return []
    return re.findall(r"macro\[@name='([^']+)'\]", open(xml, encoding='utf-8').read())


def macros_from_mod(mod_dir):
    out = []
    for xml in glob.glob(os.path.join(mod_dir, 'libraries', '*.xml')):
        text = open(xml, encoding='utf-8').read()
        out += re.findall(r"<replace sel=\"/macros/macro\[@name='([^']+)'\]", text)
        out += re.findall(r'<macro name="([^"]+)"', text)
    return sorted(set(out))


def build_targets(root):
    cats = sorted(glob.glob(str(root / '0[0-9].cat')))
    cats += sorted(glob.glob(str(root / 'extensions' / '*' / 'ext_[0-9][0-9].cat')))
    targets = []
    for cat in cats:
        cat_p = Path(cat)
        dat_p = cat_p.with_suffix('.dat')
        tag = cat_p.parent.name + '/' + cat_p.name
        offset = 0
        for name, size, _mtime, _md5 in _parse_cat(cat_p):
            key = name.lower()
            if key.endswith('.xml') and (key.startswith(SCAN) or any('/' + d in key for d in SCAN)):
                targets.append((str(dat_p), offset, size, key, tag))
            offset += size
    return sorted(targets, key=lambda t: (t[0], t[1]))


def classify(key, line, macro):
    if re.search(r'<macro\s+name="%s"' % re.escape(macro), line):
        return 'define定义'
    if '<select' in line:
        return 'pool外观池'
    if 'ref="%s"' % macro in line:
        return 'ref被继承'
    if key.startswith('md/') or '/md/' in key:
        return 'md任务脚本'
    if key.startswith('maps/') or '/maps/' in key:
        return 'maps开局'
    if '/cutscenes/' in key:
        return 'cutscene过场'
    return 'other其他'


def main():
    ap = argparse.ArgumentParser(description="查角色宏被谁引用")
    ap.add_argument('--macro', action='append', default=[], help="宏名，可重复")
    ap.add_argument('--mod', help="从某个产物目录的 libraries/*.xml 里取宏名")
    ap.add_argument('--family', help="从游戏包里枚举所有以该前缀开头的角色宏，例如 character_argon_female")
    args = ap.parse_args()

    macros = args.macro or (macros_from_mod(args.mod) if args.mod else default_macros())
    if args.family:
        root = find_game_root()
        found = set()
        for cat in sorted(glob.glob(str(root / '0[0-9].cat'))) + sorted(
                glob.glob(str(root / 'extensions' / '*' / 'ext_[0-9][0-9].cat'))):
            dat = Path(cat).with_suffix('.dat')
            offset = 0
            for name, size, _m, _d in _parse_cat(Path(cat)):
                if name.lower().endswith('character_macros.xml'):
                    with open(dat, 'rb') as fh:
                        fh.seek(offset)
                        found |= set(re.findall(r'<macro name="(%s[^"]*_macro)"' % re.escape(args.family),
                                                fh.read(size).decode('utf-8', 'replace')))
                offset += size
        macros = sorted(found)
        print(f"家族 {args.family}: {len(macros)} 个宏")
    if not macros:
        print("没给宏名，也读不到默认清单；用 --macro 或 --mod 指定")
        return 2
    want = set(macros)
    print(f"待查宏 {len(macros)} 个")

    root = find_game_root()
    targets = build_targets(root)
    print(f"扫描 {len(targets)} 个脚本类 xml，共 {sum(t[2] for t in targets)/1e6:.1f} MB", flush=True)

    hits = collections.defaultdict(list)
    cur_dat, fh = None, None
    for dat, off, size, key, tag in targets:
        if dat != cur_dat:
            if fh:
                fh.close()
            fh = open(dat, 'rb')
            cur_dat = dat
        fh.seek(off)
        try:
            body = fh.read(size).decode('utf-8')
        except UnicodeDecodeError:
            continue
        present = want.intersection(NAME_RE.findall(body))
        if not present:
            continue
        for line in body.splitlines():
            for macro in present:
                if macro in line:
                    hits[macro].append((key, classify(key, line, macro), line.strip()[:200], tag))
    if fh:
        fh.close()

    summary = collections.Counter()
    for macro in macros:
        entries = hits.get(macro, [])
        by_kind = collections.defaultdict(list)
        for f, k, l, c in entries:
            by_kind[k].append((f, l, c))
        plot = [k for k in by_kind if k in ('md任务脚本', 'maps开局', 'cutscene过场')]
        tag = '剧情/任务   ' if plot else ('普通 NPC    ' if 'pool外观池' in by_kind else '无人引用    ')
        print(f"\n{tag} {macro}")
        for kind in ('define定义', 'ref被继承', 'pool外观池', 'md任务脚本', 'maps开局', 'cutscene过场', 'other其他'):
            items = by_kind.get(kind, [])
            if not items:
                continue
            files = sorted({f for f, _, _ in items})
            print(f"    {kind:10s} x{len(items):3d}  {', '.join(files[:5])}{' ...' if len(files) > 5 else ''}")
            if kind in ('md任务脚本', 'maps开局', 'cutscene过场'):
                for f, l, c in items[:4]:
                    print(f"        [{c}:{f}] {l}")
        summary[tag.strip()] += 1
    print("\n汇总:", dict(summary))
    return 0


if __name__ == '__main__':
    sys.exit(main())
