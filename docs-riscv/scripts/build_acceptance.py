# -*- coding: utf-8 -*-
"""
由 4 份证据生成《TileLang 算子验收表.xlsx》。

输入（本地 evidence/ 与根目录）：
  - ops_list.json                  : Excel 216 行算子清单（语言/名称/来源/链接/负责人）
  - evidence/operator_inventory.json : 项目官方 operator_inventory.py 产出（205+48 条）
  - evidence/riscv_results.xml      : pytest --junitxml（823 testcase）
  - evidence/riscv_results.log      : 合并测试日志（含 823 passed）

输出：
  - alias_map.json                 : Excel 名 -> manifest/工厂名 映射（含置信度与匹配方式）
  - TileLang_算子验收表.xlsx        : 6 个 sheet

口径（不美化）：
  「通过(数值)」= 该算子关联的 testcase 至少一个 numerical passed 且无 failed/xfail/skip
  「仅组件证据」「未映射」「spec-only」「known_gaps」单独标注，不算通过
  「族级覆盖」= 一个 Excel 名扇出到多条 manifest，明确不声称逐个验证
  引用级证据会重叠，见 inventory 自带 scope 声明
"""
from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVID = ROOT / "evidence"

# xlsx skill 的统一样式
SKILL = Path(r"C:\Users\Lenovo\.zcode\cli\plugins\cache\zcode-plugins-official"
             r"\document-skills\0.1.4\skills\xlsx")
sys.path.insert(0, str(SKILL / "templates"))
import base as X  # noqa: E402
from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side  # noqa: E402
from openpyxl.utils import get_column_letter  # noqa: E402


# ---------------------------------------------------------------- 名称归一化
ACRONYM = {
    "groupedqueryattention": "gqa",
    "multiheadlatentattention": "mla",
    "multiheadattention": "mha",
    "deepseeksparseattention": "dsa",
    "nonsemanticattention": "nsa",
    "gateddeltanet": "gdn",
    "deltanet": "deltanet",
    "fusedmoe": "moe", "moe": "moe",
    "statespaceduality": "ssd", "ssd": "ssd",
    "gatedlinearattention": "gla", "gla": "gla",
    "mhc": "mhc",
    "rope": "rope",
    "kvcache": "kvcache",
    "fp8": "fp8",
}
TOKEN_FIX = {
    "forward": "fwd", "backward": "bwd",
    "decode": "decode", "prefill": "prefill",
    "quant": "quant", "lighting": "lightning",
}
STOP = {"op", "fwd", "bwd", "kernel", "with", "and", "the", "no", "to", "of"}


def norm(name: str) -> str:
    """小写化，展开缩写，去掉 Op/Kernel 等噪声。"""
    s = name.lower()
    for long, short in sorted(ACRONYM.items(), key=lambda kv: -len(kv[0])):
        s = s.replace(long, short)
    return s


def camel_tokens(name: str) -> set[str]:
    """把驼峰/下划线名切成小写词元集合。"""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s = s.replace("-", "_")
    toks = [t.lower() for t in re.split(r"[_]+", s) if t]
    out = set()
    for t in toks:
        out.add(t)
        for k, v in TOKEN_FIX.items():
            if t == k:
                out.add(v)
    # 数字/维度词（d256 / 3wg / c2c）保留但降权
    return out


def sim(a: str, b: str) -> float:
    """词元 Jaccard + 归一化子串包含加分。"""
    ta, tb = camel_tokens(a), camel_tokens(b)
    ta = {t for t in ta if t not in STOP}
    tb = {t for t in tb if t not in STOP}
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    na, nb = norm(a), norm(b)
    contain = 0.0
    if na in nb or nb in na:
        contain = 0.25
    return min(1.0, j + contain)


# ---------------------------------------------------------------- 载入证据
ops = json.load(open(ROOT / "ops_list.json", encoding="utf-8"))
inv = json.load(open(EVID / "operator_inventory.json", encoding="utf-8"))
tree = ET.parse(EVID / "riscv_results.xml")
cases = list(tree.getroot().iter("testcase"))

to_entries = inv["libraries"]["TileOps"]["entries"]
tk_entries = inv["libraries"]["TileKernels"]["entries"]
to_names = [e["name"] for e in to_entries]
tk_names = [e["name"] for e in tk_entries]
to_by_name = {e["name"]: e for e in to_entries}
tk_by_name = {e["name"]: e for e in tk_entries}


def outcome(c) -> str:
    if c.find("failure") is not None or c.find("error") is not None:
        return "failed"
    sk = c.find("skipped")
    if sk is not None:
        return "expected-failure" if sk.get("type") == "pytest.xfail" else "skipped"
    return "passed"


def scope(name: str) -> str:
    if "rejects_cuda_" in name:
        return "rejection"
    if "lowers_to_riscv" in name:
        return "compile-only"
    if name.startswith("test_public_"):
        return "public-api-numerical"
    return "kernel-or-component-numerical"


# 按 classname 后缀建 <测试模块> -> testcase 索引
by_module: dict[str, list] = defaultdict(list)
for c in cases:
    cn = c.get("classname", "")
    by_module[cn.rsplit(".", 1)[-1]].append(c)

NUM_SCOPES = {"kernel-or-component-numerical", "public-api-numerical"}


def evidence_for(ref_files: list[str]) -> dict:
    """按 inventory 的 ref（test_x.py 或 test_x.py::selector）聚合 testcase 结果。"""
    cnt: Counter = Counter()
    scopes: Counter = Counter()
    files: set[str] = set()
    for ref in ref_files:
        file, _, sel = ref.partition("::")
        mod = Path(file).stem
        files.add(file)
        for c in by_module.get(mod, []):
            if sel and not c.get("name", "").startswith(sel.split("[")[0]):
                continue
            o = outcome(c)
            cnt[o] += 1
            if o == "passed":
                scopes[scope(c.get("name", ""))] += 1
    return {"outcomes": dict(cnt), "scopes": dict(scopes), "files": sorted(files)}


# ---------------------------------------------------------------- 别名匹配
# 人工确认的高置信映射（Excel 全称 -> 项目真实名；1:N 用列表）
EXPLICIT_ALIAS: dict[str, list[str]] = {
    # TileKernels：Excel 用文件名，inventory 用 factory 名
    "expand_kernel": ["expand_to_mhc_fwd_tl", "expand_to_mhc_bwd_tl"],
    "norm_fn_kernel": ["_mhc_fn_normw_merge_fwd", "_mhc_fn_normw_merge_bwd"],
    "post_kernel": ["_mhc_post_fwd", "_mhc_post_bwd"],
    "sinkhorn_kernel": ["_mhc_sinkhorn_fwd", "_mhc_sinkhorn_bwd"],
    # TileOps：全称 -> 缩写族
    "DeepSeekSparseAttentionDecodeWithKVCacheFwdOp": ["DSADecodeWithKVCacheFwdOp"],
    "MultiHeadLatentAttentionDecodeWithKVCacheFwdOp": ["MLADecodeWithKVCacheFwdOp"],
    "MeanPoolingForwardOp": ["AvgPool1dFwdOp"],
    "NSACmpFwdVarlenOp": ["NSACompressedVarlenFwdOp"],
    "NSAFwdVarlenOp": ["NSAVarlenFwdOp"],
    "NSATopkVarlenOp": ["NSATopKVarlenFwdOp"],
    "GroupedQueryAttentionBwdOp": ["GQABwdOp"],
    "GroupedQueryAttentionDecodePagedWithKVCacheFwdOp": ["GQAPagedFwdOp"],
    "GroupedQueryAttentionDecodeWithKVCacheFwdOp": ["GQADenseFwdOp"],
    "GroupedQueryAttentionFwdOp": ["GQADenseFwdOp", "GQAVarlenFwdOp"],
    "GroupedQueryAttentionPrefillFP8TensorCoreFwdOp": ["GQADenseFwdOp"],
    "GroupedQueryAttentionPrefillFwdOp": ["GQAVarlenFwdOp"],
    "GroupedQueryAttentionPrefillPagedWithFP8KVCacheFwdOp": ["GQAPrefillPagedWithKVCacheFwdOp"],
    "GroupedQueryAttentionPrefillPagedWithKVCacheFwdOp": ["GQAPrefillPagedWithKVCacheFwdOp"],
    "GroupedQueryAttentionPrefillVarlenFwdOp": ["GQAVarlenFwdOp"],
    "GroupedQueryAttentionPrefillWithKVCacheFwdOp": ["GQAPrefillPagedWithKVCacheFwdOp"],
    # 下列 Excel 名在 manifest 中无 1:1 声明，保持空映射 = 如实标未映射，不硬塞假名
    "GroupedQueryAttentionSlidingWindowFwdOp": [],
    "GroupedQueryAttentionSlidingWindowVarlenFwdOp": [],
    "MultiHeadAttentionBwdOp": [],
    "MultiHeadAttentionDecodePagedWithKVCacheFwdOp": ["MHADecodePagedWithKVCacheFwdOp"],
    "MultiHeadAttentionDecodeWithKVCacheFwdOp": ["MHADecodePagedWithKVCacheFwdOp"],
    "MultiHeadAttentionFwdOp": [],
    "Conv1dBiasFwdOp": ["Conv1dFwdOp"],
    "Conv2dBiasFwdOp": ["Conv2dFwdOp"],
    "Conv3dBiasFwdOp": ["Conv3dFwdOp"],
    "DaCumsumFwdOp": ["CumsumFwdOp"],
    "DeltaNetBwdOp": ["DeltaNetChunkBwdOp"],
    "DeltaNetOp": ["DeltaNetFwdOp"],
    "DeltaNetDecodeOp": ["DeltaNetRecurrentFwdOp"],
    "DropoutOp": ["DropoutFwdOp"],
    "LerpFwdOp": ["LerpScalarFwdOp", "LerpTensorFwdOp"],
    "ClampFwdOp": ["ClampTensorFwdOp"],
    "ClampMaxFwdOp": ["ClampScalarFwdOp"],
    "ClampMinFwdOp": ["ClampScalarFwdOp"],
    "MaskedFillFwdOp": ["MaskedFillTensorFwdOp"],
    "EngramDecodeOp": ["EngramDecodeFwdOp"],
    "FFTC2COp": ["FFTC2CFwdOp"],
    "FP8LightingIndexerOp": ["FP8LightningIndexerFwdOp"],
    "FP8QuantOp": ["FP8QuantFwdOp"],
    "GatedDeltaNetBwdOp": [],
    "GatedDeltaNetDecodeOp": [],
    "GatedDeltaNetFwdOp": ["GDNFwdOp"],
    "GatedDeltaNetOp": ["GDNFwdOp"],
    "GLADecodeOp": ["GLAFwdOp"],
    "GLABwdOp": ["GLAChunkBwdOp"],
    "GemmOp": ["GemmFwdOp"],
    "GroupedGemmOp": ["GroupedGemmFwdOp"],
    "MHCPostOp": ["MHCPostFwdOp"],
    "MHCPreOp": ["MHCPreFwdOp"],
    "FusedMoeFwdCbFwdOp": ["FusedMoEFwdOp"],
    "FusedMoeFwdOp": ["FusedMoEFwdOp"],
    "FusedTopKOp": ["FusedTopKFwdOp"],
    "MoePermuteAlignFwdOp": ["MoEPermuteAlignFwdOp"],
    "FusedMoEExpertsNopadPersistent3WGFwdOp": ["FusedMoEExpertsFwdOp"],
    "MoeGroupedGemmNopadFwdOp": ["MoEGroupedGemmFwdOp"],
    "MoeGroupedGemmNopad3WGFusedActFwdOp": ["MoEGroupedGemmFwdOp"],
    "MoePermuteNopadFwdOp": ["MoEPrePermuteFwdOp"],
    "MoeUnpermuteFwdOp": ["MoEPostPermuteFwdOp"],
    "GroupNormFwdOpNoAffine": ["GroupNormFwdOp"],
    "InstanceNormFwdOpNoAffine": ["InstanceNormFwdOp"],
    "InfNormFwdOp": ["VectorNormFwdOp"],
    "L1NormFwdOp": ["VectorNormFwdOp"],
    "L2NormFwdOp": ["VectorNormFwdOp"],
    "RopeLlama31Op": ["RoPELlama31FwdOp"],
    "RopeLongRopeOp": ["LongRoPEFwdOp"],
    "RopeNeoxOp": ["RoPENeoxPositionIdsFwdOp"],
    "RopeNeoxPositionIdsOp": ["RoPENeoxPositionIdsFwdOp"],
    "RopeNonNeoxOp": ["RoPEFwdOp"],
    "RopeYarnOp": ["RoPEFwdOp"],
    "SSDDecodeOp": ["SSDRecurrentFwdOp"],
    "TopkSelectorOp": ["TopKSelectFwdOp"],
}


def match_one(excel_name: str, src: str) -> dict:
    pool = to_names if src == "TileOPs" else tk_names
    # 1) 精确
    if excel_name in pool:
        return {"matched": [excel_name], "method": "exact", "conf": 1.0}
    # 2) 人工确认别名（只保留真实存在于 pool 的目标，标 manual）
    if excel_name in EXPLICIT_ALIAS:
        declared = EXPLICIT_ALIAS[excel_name]
        if not declared:  # 显式声明：manifest 中无 1:1 对应，不再模糊匹配造假
            return {"matched": [], "method": "no-declaration", "conf": 0.0}
        valid = [n for n in declared if n in pool]
        if valid:
            return {"matched": valid, "method": "manual", "conf": 0.9}
    # 3) TileKernels 的 get_ / _mhc_ / _tl 命名
    if src == "TileKernels":
        cands = [n for n in pool if norm(excel_name) in norm(n) or norm(n) in norm(excel_name)]
        if not cands:
            scored = [(sim(excel_name, n), n) for n in pool]
            cands = [n for s, n in scored if s >= 0.6]
        if cands:
            return {"matched": cands, "method": "rule", "conf": 0.85}
    # 4) TileOps 词元相似度
    if src == "TileOPs":
        nn = norm(excel_name)
        scored = []
        for n in pool:
            s = sim(excel_name, n)
            if nn in norm(n) or norm(n) in nn:
                s = min(1.0, s + 0.3)
            scored.append((s, n))
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0.0
        if best >= 0.75:
            got = [n for s, n in scored if s >= best - 0.06]
            return {"matched": got, "method": "rule", "conf": round(best, 2)}
        if best >= 0.28:
            got = [n for s, n in scored[:3] if s >= 0.28]
            return {"matched": got, "method": "fuzzy", "conf": round(best, 2)}
    return {"matched": [], "method": "unmapped", "conf": 0.0}


alias_map = []
for o in ops:
    m = match_one(o["name"], o["src"])
    alias_map.append({
        "excel_name": o["name"], "src": o["src"],
        "matched": m["matched"], "method": m["method"], "confidence": m["conf"],
    })
json.dump(alias_map, open(ROOT / "alias_map.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

# ---------------------------------------------------------------- verdict
def verdict_for(entry: dict, ev: dict, is_kernel: bool) -> str:
    outs = ev["outcomes"]
    scopes = ev["scopes"]
    if outs.get("failed") or outs.get("expected-failure") or outs.get("skipped"):
        return "失败/跳过"
    if not any(scopes.get(s) for s in NUM_SCOPES):
        # 有 rejection/compile-only 但无数值
        if any(scopes.values()):
            return "仅非数值证据"
        return "无用例证据"
    if is_kernel:
        return "通过(数值)"
    if entry.get("upstream_status") == "spec-only":
        return "通过(数值, spec-only)"
    if entry.get("known_gaps"):
        return "通过(数值, 有 known_gaps)"
    return "通过(数值)"


# ---------------------------------------------------------------- 建表
wb = Workbook()
wb.properties.creator = "Z.ai"
THIN = Side(style="thin", color=X.NEUTRAL_200)
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def header(ws, row, cols):
    for j, v in enumerate(cols, 1):
        c = ws.cell(row=row, column=j, value=v)
        c.font = X.font_header()
        c.fill = X.fill_header()
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER


def body(ws, row, vals, idx):
    for j, v in enumerate(vals, 1):
        c = ws.cell(row=row, column=j, value=v)
        c.font = X.font_body()
        c.alignment = Alignment(vertical="top", wrap_text=isinstance(v, str) and len(str(v)) > 30)
        c.border = BORDER
        if idx % 2:
            c.fill = PatternFill("solid", fgColor=X.NEUTRAL_100)


def widths(ws, ws_widths):
    for i, w in enumerate(ws_widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


VERDICT_FILL = {
    "通过(数值)": X.ACCENT_POSITIVE,
    "通过(数值, spec-only)": X.ACCENT_WARNING,
    "通过(数值, 有 known_gaps)": X.ACCENT_WARNING,
    "仅组件证据": X.ACCENT_WARNING,
    "仅非数值证据": X.ACCENT_WARNING,
    "无用例证据": X.ACCENT_NEGATIVE,
    "未映射": X.NEUTRAL_600,
    "失败/跳过": X.ACCENT_NEGATIVE,
}


def color_verdict(ws, col, first_row, last_row):
    for r in range(first_row, last_row + 1):
        c = ws.cell(row=r, column=col)
        v = str(c.value or "")
        for key, color in VERDICT_FILL.items():
            if v.startswith(key):
                c.font = Font(name=X.FONT_NAME, size=11, bold=True, color=color)
                break


# ---- Sheet 1 口径说明 ----
ws = wb.active
ws.title = "口径说明"
ws["A1"] = "TileLang-RISC-V 算子验收表 · 口径与统计"
ws["A1"].font = X.font_title()
ws.merge_cells("A1:D1")
notes = [
    ("项目", "RuyiAI-Stack/tilelang-riscv @ 32fce5ed"),
    ("机器", "openeuler-riscv64 (64 核, glibc 2.38, Python 3.11.6, torch 2.4.1, numpy 1.26.4)"),
    ("测试命令", "pytest -q testing/python/riscv/{test_riscv_*.py,codegen_ops,TileKernels,TileOps} --junitxml=riscv_results.xml"),
    ("测试结果", "823 passed, 0 failed（108 核心 + 252 codegen_ops + 48 TileKernels + 415 TileOps）"),
    ("证据文件", "riscv_results.xml / operator_inventory.json / riscv_results.log（本目录 evidence/）"),
    ("", ""),
    ("【计数口径】", ""),
    ("Excel 算子", "216 行 = TileKernels 37 + TileOPs 179（用户提供的清单）"),
    ("项目枚举", "253 条 = 48 JIT factory (TileKernels) + 205 manifest 算子 (TileOPs)"),
    ("两者不等", "枚举粒度与命名体系不同，不能 1:1 外连接；本表用别名表桥接并如实标注"),
    ("48 / 415", "是参数化后的 pytest 用例数，不是算子数（项目 README 明确声明）"),
    ("", ""),
    ("【verdict 定义】", ""),
    ("通过(数值)", "关联用例中至少一个 numerical passed 且无 failed/xfail/skip"),
    ("通过(数值, spec-only)", "上游 manifest 标 spec-only，但本地有数值证据"),
    ("通过(数值, 有 known_gaps)", "有数值证据，但项目记录了已知限制（如 FP16/BF16 __fdividef）"),
    ("仅组件证据", "inventory 的 evidence = component-test-references，非逐算子数值"),
    ("仅非数值证据", "仅有 rejection / compile-only 用例，不算数值通过"),
    ("无用例证据", "有映射但 JUnit 里找不到对应用例"),
    ("未映射", "Excel 名无对应：unmapped=匹配不上；no-declaration=项目 manifest 里本就没有该算子的 1:1 声明，未硬塞近似名"),
    ("匹配方式", "exact=精确 / manual=人工确认别名 / rule=规则或相似度 / fuzzy=低置信待复核 / no-declaration / unmapped"),
    ("族级覆盖", "一个 Excel 名扇出到多条 manifest，只证明该族被覆盖，不声称逐个验证"),
    ("", ""),
    ("【不美化原则】", ""),
    ("引用会重叠", "inventory 自带声明: Counts overlap across entries; 一个 case 可计给多个算子"),
    ("unmapped", "20 条无 test 引用；component-only 9 条；spec-only 14 条；known_gaps 21 项 —— 均如实标注"),
    ("--require-tileops-numerical", "按设计必然失败，未用作验收闸门"),
    ("fp8 补丁", "test_paged_kv_cache_write.py 4 处 torch.full→fp32 转换（PyTorch CPU 不支持 fp8 fill），属测试层改动"),
]
r = 3
for k, v in notes:
    a = ws.cell(row=r, column=1, value=k)
    b = ws.cell(row=r, column=2, value=v)
    a.font = X.font_subheader() if k.startswith("【") else X.font_body()
    b.font = X.font_body()
    b.alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=4)
    r += 1

# 统计块
stat = [
    ("别名匹配统计", dict(Counter(a["method"] for a in alias_map))),
]
excel_verdicts = Counter()

# ---- Sheet 2 测试套件验收（按 test 文件）----
ws = wb.create_sheet("测试套件验收")
cols = ["库", "测试文件", "用例数", "通过", "失败", "跳过", "数值用例", "源"]
header(ws, 1, cols)
mod_stat = defaultdict(lambda: Counter())
for c in cases:
    cn = c.get("classname", "")
    lib = next((k for k in ("TileOps", "TileKernels", "codegen_ops") if f".{k}." in cn), "core")
    mod = cn.rsplit(".", 1)[-1]
    key = (lib, mod + ".py")
    o = outcome(c)
    mod_stat[key][o] += 1
    mod_stat[key]["total"] += 1
    if o == "passed" and scope(c.get("name", "")) in NUM_SCOPES:
        mod_stat[key]["numerical"] += 1
for i, (k, s) in enumerate(sorted(mod_stat.items()), 2):
    lib, f = k
    body(ws, i, [lib, f, s["total"], s["passed"], s["failed"],
                 s["skipped"] + s["expected-failure"], s["numerical"], "riscv_results.xml"], i)
    excel_verdicts[f] = s["total"]
ws.freeze_panes = "A2"
widths(ws, [14, 58, 10, 10, 10, 10, 12, 20])

# ---- Sheet 3 算子验收_Excel216 ----
ws = wb.create_sheet("算子验收_Excel216")
cols = ["语言", "算子名称(Excel)", "来源", "来源链接", "负责人", "匹配方式", "置信度",
        "映射到的项目名", "测试文件", "用例通过", "verdict", "说明"]
header(ws, 1, cols)
amap = {a["excel_name"]: a for a in alias_map}
for i, o in enumerate(ops, 2):
    a = amap[o["name"]]
    matched = a["matched"]
    note = ""
    if a["method"] in ("unmapped", "no-declaration"):
        verdict = "未映射"
        note = ("项目 manifest 中无 1:1 对应声明" if a["method"] == "no-declaration"
                else "命名体系不一致，需人工确认")
        files, npass = [], 0
    else:
        agg = Counter()
        allfiles = set()
        for mn in matched:
            e = to_by_name.get(mn) or tk_by_name.get(mn)
            if not e:
                continue
            ev = evidence_for((e.get("tests") or []) + (e.get("public_api_tests") or []))
            for k, v in ev["outcomes"].items():
                agg[k] += v
            allfiles.update(ev["files"])
            if e.get("upstream_status") == "spec-only":
                note = "上游 spec-only"
            if e.get("known_gaps"):
                note = (note + "; " if note else "") + "known_gaps: " + "; ".join(e["known_gaps"])[:80]
        npass = agg["passed"]
        files = sorted(allfiles)
        ev2 = {"outcomes": dict(agg), "scopes": {}}
        # 用第一个 entry 的 scope 摘要重算
        if matched:
            e0 = to_by_name.get(matched[0]) or tk_by_name.get(matched[0])
            if e0:
                ev2 = evidence_for((e0.get("tests") or []) + (e0.get("public_api_tests") or []))
                ev2["outcomes"] = dict(agg)
        if agg["failed"] or agg["expected-failure"] or agg["skipped"]:
            verdict = "失败/跳过"
        elif not files:
            verdict = "无用例证据"
        elif not any(v for k, v in ev2["scopes"].items() if k in NUM_SCOPES):
            verdict = "仅非数值证据"
        elif len(matched) > 1:
            verdict = "通过(数值, 族级覆盖)"
            note = (note + "; " if note else "") + f"1:N 扇出 {len(matched)} 条，仅证明族级覆盖"
        else:
            verdict = "通过(数值)"
        if a["method"] == "fuzzy":
            note = (note + "; " if note else "") + "低置信模糊匹配，需人工复核"
    excel_verdicts["V:" + verdict] += 1
    body(ws, i, [o["lang"], o["name"], o["src"], o["link"], o["owner"] or "",
                 a["method"], a["confidence"], "; ".join(matched)[:120],
                 "; ".join(files)[:120], npass, verdict, note], i)
ws.freeze_panes = "A2"
color_verdict(ws, 11, 2, len(ops) + 1)
widths(ws, [10, 46, 14, 52, 10, 12, 9, 44, 40, 10, 26, 46])

# ---- Sheet 4 算子验收_manifest253 ----
ws = wb.create_sheet("算子验收_manifest253")
cols = ["库", "项目名", "family", "upstream_status", "evidence", "测试引用",
        "JUnit 结果", "数值 scope", "known_gaps", "源"]
header(ws, 1, cols)
row = 2
for lib, entries in (("TileOps", to_entries), ("TileKernels", tk_entries)):
    for e in entries:
        ev = evidence_for((e.get("tests") or []) + (e.get("public_api_tests") or []))
        body(ws, row, [
            lib, e["name"], e.get("family", ""), e.get("upstream_status", ""),
            e.get("evidence", ""),
            "; ".join((e.get("tests") or []) + (e.get("public_api_tests") or []))[:150],
            json.dumps(ev["outcomes"], ensure_ascii=False),
            json.dumps(ev["scopes"], ensure_ascii=False),
            "; ".join(e.get("known_gaps") or [])[:120],
            e.get("manifest") or e.get("source", ""),
        ], row)
        row += 1
ws.freeze_panes = "A2"
widths(ws, [14, 44, 14, 16, 26, 52, 24, 34, 40, 56])

# ---- Sheet 5 别名表 ----
ws = wb.create_sheet("别名表")
cols = ["Excel 算子名", "来源", "匹配方式", "置信度", "映射到的项目名", "备注"]
header(ws, 1, cols)
for i, a in enumerate(alias_map, 2):
    remark = ""
    if a["method"] == "fuzzy":
        remark = "低置信，需人工复核"
    elif a["method"] == "no-declaration":
        remark = "项目 manifest 中无 1:1 对应声明（未硬塞近似名）"
    elif a["method"] == "unmapped":
        remark = "命名体系不一致或该名在仓库中不存在"
    elif len(a["matched"]) > 1:
        remark = f"1:N 扇出 {len(a['matched'])} 条"
    body(ws, i, [a["excel_name"], a["src"], a["method"], a["confidence"],
                 "; ".join(a["matched"]), remark], i)
ws.freeze_panes = "A2"
widths(ws, [50, 14, 12, 10, 60, 40])

# ---- Sheet 6 未映射与已知限制 ----
ws = wb.create_sheet("未映射与已知限制")
cols = ["类别", "库", "名称", "详情", "来源"]
header(ws, 1, cols)
gaps_rows = []
# 别名未映射
for a in alias_map:
    if a["method"] in ("unmapped", "fuzzy", "no-declaration"):
        gaps_rows.append(("别名" + a["method"], a["src"], a["excel_name"],
                          f"conf={a['confidence']}; matched={a['matched']}", "alias_map.json"))
# inventory unmapped / component / spec-only / known_gaps
for lib, entries in (("TileOps", to_entries), ("TileKernels", tk_entries)):
    for e in entries:
        if e.get("evidence") == "unmapped":
            gaps_rows.append(("无测试引用(unmapped)", lib, e["name"],
                              f"upstream={e.get('upstream_status')}", "operator_inventory.json"))
        if e.get("evidence") == "component-test-references":
            gaps_rows.append(("仅组件证据", lib, e["name"],
                              "; ".join(e.get("tests") or [])[:100], "operator_inventory.json"))
        if e.get("upstream_status") == "spec-only":
            gaps_rows.append(("上游 spec-only", lib, e["name"],
                              "上游无完全合规实现", "operator_inventory.json"))
        if e.get("known_gaps"):
            gaps_rows.append(("known_gaps", lib, e["name"],
                              "; ".join(e["known_gaps"])[:160], "operator_inventory.json"))
for i, g in enumerate(gaps_rows, 2):
    body(ws, i, list(g), i)
ws.freeze_panes = "A2"
widths(ws, [26, 14, 50, 76, 26])

# 回填口径说明的统计
ws1 = wb["口径说明"]
stat_row = r + 1
ws1.cell(row=stat_row, column=1, value="【匹配统计】").font = X.font_subheader()
stat_row += 1
for k, v in [("别名匹配", dict(Counter(a["method"] for a in alias_map))),
             ("Excel verdict", {k[2:]: v for k, v in excel_verdicts.items() if k.startswith("V:")})]:
    ws1.cell(row=stat_row, column=1, value=k).font = X.font_body()
    ws1.cell(row=stat_row, column=2, value=json.dumps(v, ensure_ascii=False)).font = X.font_body()
    ws1.merge_cells(start_row=stat_row, start_column=2, end_row=stat_row, end_column=4)
    stat_row += 1
widths(ws1, [30, 40, 30, 30])
ws1.freeze_panes = "A3"

out = ROOT / "TileLang_算子验收表.xlsx"
wb.save(out)
print("saved:", out)
print("sheets:", wb.sheetnames)
print("alias method:", dict(Counter(a["method"] for a in alias_map)))
print("Excel216 verdict:", {k[2:]: v for k, v in excel_verdicts.items() if k.startswith("V:")})
