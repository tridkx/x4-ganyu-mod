# 甘雨 Ganyu → X4: Foundations **Argon 女性** NPC

把《原神》的**甘雨**（MMD/PMX，275 骨，模型改造：观海 / BiliBili 观海子）
移植成《X4：基石》里的 **Argon 女性** NPC 外观，替换头部与躯干网格。

```bash
python tools/build_all.py --mode add        # 一条命令：重定向 → 贴图 → 导出 → 组装 → 打包
python tools/verify_mod.py  --mode add      # 发版前自检
```

## 两种形态（装之前二选一）

| 形态 | 你会得到什么 | 什么时候用 |
|---|---|---|
| **仅增加** `--mode add` | 新增一条 macro `character_argon_female_ganyu_macro`，并往 **6 个 Argon 女性外观池**各加一条 `<select>`。**原版 macro 一条不动**，她只是随机出现的其中一种；其余女性保持自己的脸/名字/语音。**剧情/任务 NPC 不走外观池，保持原版。** | 正式游玩、发布 |
| **全量替换** `--mode replace` | 把 **122 个** Argon 女性 macro 的 `<models>` 逐个改写成甘雨，含不走池的剧情 NPC —— 所有 Argon 女性都是她 | 想看模型出现在所有地方 / 调试 |

两种形态共用同一个扩展 id（`x4_ganyu_mod`）与同一套资产路径，**同时只能装一个**。
发布包已经分开：

```
work/dist/x4_ganyu_argon_add_v1.0.zip       ← 仅增加（推荐）
work/dist/x4_ganyu_argon_replace_v1.0.zip   ← 全量替换
```

解压到 `X4 Foundations/extensions/` 下即可（压缩包里已经是 `x4_ganyu_mod/` 一层），
然后在游戏的 `扩展 / Extensions` 菜单里启用。

## 管线做了什么

```
甘雨.pmx（MMD，含权重）  →  逐骨绑定姿态转移  →  ganyu_x4_stage1.blend
                        →  X4CharacterConverter 导出  →  .xac + DDS + xml
                        →  XRCatTool 打包  →  ext_01.cat / ext_01.dat
```

核心是第一步：**X4 的 NPC 替换是「换网格、留骨架」**。
`character_components.xml` 里的共享 component `character_argon_female_01`
拥有骨骼与 1100+ 条动画，macro 只挑 head/torso/props 三个**网格**槽位 ——
所以替换物必须带上**逐字节相同的 91 骨骼 Biped 骨架**。本 mod 的两个 `.xac`
都是 91/91 骨记录与 vanilla 逐字节相同（`verify_mod.py` 会验证）。

| 脚本 | 作用 |
|---|---|
| `tools/paths.py` | 所有路径解析（项目相对 + 环境变量覆盖，无硬编码盘符） |
| `tools/ganyu_src.py` | 源模型位置、材质名 → ascii 词干、丢弃项、"不可见面"判据 |
| `tools/ganyu_mmd_to_x4.py` | 骨骼映射表（CORE / 折叠链 / 横向阻尼） |
| `tools/build_ganyu_x4.py` | 阶段 1：Blender 里重定向网格、分 head/torso、删不可见面 |
| `tools/prepare_textures_ganyu.py` | 4 张 atlas → DDS（BC1/BC3）+ 材质 manifest |
| `tools/build_ganyu_mod.py` | 阶段 2：填进 vanilla 宿主的网格槽、导出 `.xac` |
| `tools/make_mod.py` | 组装 mod 树（`--mode add` / `--mode replace`） |
| `tools/find_female_macros.py` | 枚举「有效 race=argon 且 female」的 macro → `work/argon_female_macros.json` |
| `tools/verify_mod.py` | 发版自检：树 / 材质 / **骨架逐字节** / XML 语义 |
| `tools/build_release.py` | 打成发布 zip |
| `tools/diag_*.py` `measure_ganyu_geometry.py` | 各项判据的**测量**脚本（结论都从它们来） |

## 复现

```bash
python   tools/find_female_macros.py --race argon          # 生成 122 个 macro 清单（replace 用）
python   tools/build_all.py --mode add                     # 全量重建（add 形态）
python   tools/build_all.py --mode replace                 # 全量重建（replace 形态）
python   tools/verify_mod.py --mode add                    # 自检，参数必须与产物一致
python   tools/build_release.py --version 1.0              # 发布 zip
python   tools/build_all.py --mode add --deploy            # 可选：构建并安装进游戏
python   tools/deploy.py --mode none                       # 卸载
```

`build_all.py` **默认只构建、不安装**（`deploy` 不在默认步骤里）：它会往游戏的
`extensions/` 写目录，这种事不该由"构建"顺手做掉。两个形态共用一个扩展 id，
装另一个就等于替换掉当前这个。

环境依赖：Blender 5.2、系统 Python（Pillow / numpy / scipy）、
[X Tools](https://www.egosoft.com/download/x4/bonus_en.php)（`XRCatTool.exe`）。
`tools/paths.py` 会自己探测它们，找不到时用环境变量覆盖
（`GANYU_SRC` / `X4_GAME` / `XRCAT_TOOL` / `X4_WORKSPACE`）。

## 与工作区另外两个 Argon 工程的差异

| | 萝丝 / 艾梅莉埃 / 荧 | 孟柏汝 BoRu | **甘雨** |
|---|---|---|---|
| 来源 | RE Engine / MMD | UE4 `.psk` | **MMD/PMX（与荧同系，不同作者）** |
| 骨数 | 203 / 445 / 272 | 80 | **275（187 带权）** |
| 拟合 scale | 9.625 / 9.625 / 9.7746 | — | **9.6267 cm/unit** |
| 材质数 | — | — | **19 → 18**（丢 1） |
| 特殊点 | — | — | **衣服与身体 100% 重合**；作者的"隐藏面"要用贴图 alpha 删；宽袖长发的关节撕裂天然偏高 |

甘雨与荧同属"原神 MMD"来源，所以**骨架映射表、坐标手性、UV 的 v 轴处理
全部重新量过后与荧一致**（结论一致不等于可以照抄 —— 材质名表就是完全不同的，
见 `docs/甘雨移植进展.md` §1）。

## 已验证

| 项 | 结果 |
|---|---|
| 骨架 | head / torso 各 **91/91** bind 记录与 vanilla 逐字节相同 |
| 骨骼映射 | 187 根带权骨 → 53 直接 + **134 折叠 + 0 未映射** |
| 坐标手性 | `(x, y, z) → (x, −z, y)`，`det(R) = +1.0000`，**保持绕序** |
| 顶点预算 | head 5403 vs vanilla 5002（**1.08×**）；torso 10584 vs 4600（**2.30×**），上限 6× |
| 贴图 | 4 张 atlas → 18 个材质的 DDS，全部在 `ext_01.cat` 内可解析 |
| 自检 | `verify_mod.py` 两种形态全过（0 警告） |
| 动画预览 | 见下表 |

### 动画预览（`x4-anim-preview/tools/ai_check.py`，与荧同机同动画对照）

| 指标 | 甘雨 | 荧（已发布版） | vanilla |
|---|---|---|---|
| 关节撕裂 | **1.87–1.88×** | 2.03–2.08× | 1.00× |
| 两脚间距 | **0.87–0.90×** | 0.86–0.89× | 1.00× |
| 坐姿脚底离地 | **−7.0 cm** | −9.6 cm | +1.1 cm |
| 硬伤（骨架/绕序/顶点爆炸） | 无 | 无 | — |

指标是**比值**，且服装类几何（甘雨的宽袖、裙摆、长发）天然偏高；
逐张对比图已确认无破面、无错位、贴图正确。**甘雨的每一项都不差于已发布的荧版本。**

## 已知问题

1. **关节处的残余撕裂**（肩、肘、脚踝），根因是源骨架与 X4 骨架的脊柱/腿长比例
   不同，"给骨链共享位移"的修法会引入更明显的弯折，已在荧的项目里试过并回退。
   甘雨的实测比值（1.87×）比荧（2.03×）更接近 vanilla。
2. **坐姿时脚底比 vanilla 低 7 cm**：源模型是穿高跟鞋的姿态（脚背拱起），
   而 X4 的脚骨是平底鞋姿态。站姿已对齐（脚底 −0.5 cm），坐姿会略陷。
   荧同样存在（−9.6 cm）。

## 版权

模型版权归 **miHoYo**；模型改造者是 **观海（BiliBili：观海子）**，其为二次创作，
其 readme 明确 **请勿二次配布、请勿商用**。
本仓库**只包含工具与文档**，不含模型、贴图或游戏资产。
