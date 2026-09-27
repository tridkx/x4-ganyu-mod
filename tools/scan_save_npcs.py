#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扫描 X4 存档，统计有多少 NPC 挂着某个角色宏。

为什么需要这个：X4 把 NPC 的宏名写进存档（`<component class="npc" macro="...">`），
加载存档时宏找不到就整条组件跳过（X4.exe 里的原话：
"Macro '%s' in savegame does not exist. Skipping component(s)."）。
所以**新增宏**的外观 mod（`<add sel="/macros">`，本项目 `--mode add`）卸载以后，
那些抽中它的 NPC 会消失；而只改原版宏 `<models>` 的 mod（`--mode replace`）不会。
这个脚本用来量化前者：卸载前先看看存档里到底有多少个 NPC 会受影响。

用法（存档是 gzip 的 XML，流式读取，不落盘）：

    python tools/scan_save_npcs.py                          # 自动找最新存档，扫 "ganyu"
    python tools/scan_save_npcs.py --macro character_terran_female
    python tools/scan_save_npcs.py --macro ganyu --names 20
    python tools/scan_save_npcs.py --save "D:/some/save_006.xml.gz"

退出码：0 = 没扫到；1 = 扫到了（也就是"卸载会让这些 NPC 消失"）。
"""

import argparse
import collections
import datetime
import glob
import gzip
import os
import re
import sys

try:  # 中文输出别栽在控制台码页上
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

NPC_RE = re.compile(rb'<component class="npc" macro="([^"]+)"')
PERSON_RE = re.compile(rb'<person macro="([^"]+)" role="([^"]+)"')
ATTR_RE = re.compile(rb'(\w+)="([^"]*)"')


def save_candidates():
    """不硬编码盘符：用户主目录 -> Documents/Egosoft/X4/<profile>/save。"""
    docs = os.path.join(os.path.expanduser("~"), "Documents", "Egosoft", "X4")
    return sorted(glob.glob(os.path.join(docs, "*", "save")))


def pick_save(save_dir):
    files = glob.glob(os.path.join(save_dir, "*.xml.gz"))
    if not files:
        return None
    # 存档名带序号，取最近修改的那个
    return max(files, key=os.path.getmtime)


def open_maybe_gz(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def decode_name(raw):
    for codec in ("utf-8", "gbk"):
        try:
            return raw.decode(codec)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


def scan(path, needle, want_samples=0):
    needle_b = needle.encode("utf-8").lower()
    total_npc = 0
    total_person = 0
    hits = collections.Counter()          # 命中的宏名 -> NPC 数
    hit_person = collections.Counter()    # 命中的宏名 -> 招募列表条目数
    owners = collections.Counter()
    samples = []
    patches = []
    in_patches = False
    with open_maybe_gz(path) as fh:
        for line in fh:
            if b"<patches>" in line or b"<history>" in line:
                in_patches = True
            elif in_patches and (b"</patches>" in line or b"</history>" in line):
                in_patches = False
            elif in_patches and b"<patch " in line:
                m = re.search(rb'extension="([^"]+)"', line)
                if m:
                    patches.append(m.group(1).decode("ascii", "replace"))

            for macro in NPC_RE.findall(line):
                total_npc += 1
                if needle_b in macro.lower():
                    name = macro.decode("ascii", "replace")
                    hits[name] += 1
                    attrs = dict(ATTR_RE.findall(line))
                    owner = attrs.get(b"owner", b"?").decode("ascii", "replace")
                    owners[owner] += 1
                    if len(samples) < want_samples and attrs.get(b"name"):
                        samples.append(
                            (decode_name(attrs[b"name"]),
                             attrs.get(b"code", b"").decode("ascii", "replace"),
                             owner, name)
                        )
            for macro, role in PERSON_RE.findall(line):
                total_person += 1
                if needle_b in macro.lower():
                    hit_person[macro.decode("ascii", "replace") + " (" + role.decode() + ")"] += 1
    return dict(total_npc=total_npc, total_person=total_person, hits=hits,
                hit_person=hit_person, owners=owners, samples=samples, patches=patches)


def main():
    ap = argparse.ArgumentParser(description="统计存档里挂某个角色宏的 NPC 数量")
    ap.add_argument("--macro", default="ganyu", help="宏名子串（默认 ganyu）")
    ap.add_argument("--save", help="存档文件；不给就自动找最新的")
    ap.add_argument("--names", type=int, default=0, help="打印多少个命中的 NPC 名字")
    args = ap.parse_args()

    path = args.save
    if not path:
        dirs = save_candidates()
        if not dirs:
            print("找不到存档目录（~/Documents/Egosoft/X4/*/save），用 --save 指定")
            return 2
        path = pick_save(dirs[0])
        if not path:
            print(f"{dirs[0]} 里没有 .xml.gz 存档，用 --save 指定")
            return 2

    stamp = datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    print(f"存档: {path}  ({stamp})")
    res = scan(path, args.macro, want_samples=max(0, args.names))

    print(f"NPC 组件总数: {res['total_npc']}   招募列表条目总数: {res['total_person']}")
    print(f"存档登记的扩展 (<patches>/<history>): "
          f"{', '.join(sorted(set(res['patches']))) or '(无)'}")

    hit_n = sum(res["hits"].values()) + sum(res["hit_person"].values())
    print(f"\n命中 '{args.macro}' 的 NPC: {hit_n}"
          f"（在场 {sum(res['hits'].values())} + 招募列表 {sum(res['hit_person'].values())}）")
    for macro, n in res["hits"].most_common():
        print(f"  {n:6d}  {macro}")
    for macro, n in res["hit_person"].most_common():
        print(f"  {n:6d}  {macro}")
    if res["owners"]:
        print("  所属势力: " + ", ".join(f"{k}={v}" for k, v in res["owners"].most_common(6)))
    for name, code, owner, macro in res["samples"][: max(0, args.names)]:
        print(f"    - {name} [{code}] ({owner}) {macro}")

    if hit_n:
        print("\n卸载后这些 NPC 会被引擎跳过（Macro ... does not exist. Skipping component(s).）")
        return 1
    print("\n没有 NPC 引用它：现在卸载不会让任何 NPC 消失。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
