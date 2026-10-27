"""Explain field-count diagnostics and cross-model variation from audited snapshots.

No embedding or source-model data is changed. Run this script after the optional
export_focus_evidence.py export to rebuild the self-contained report and figures.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import html
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
INPUT = HERE / "dataset_results"
OUT = HERE / "focus_results"
PLACEHOLDERS = {"notdefined", "not defined", "undefined", "unknown", "unspecified", "unset", "not set"}
BLUE, ORANGE, SLATE, PALE = "#176b91", "#ce7944", "#738591", "#dbe5eb"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def placeholder(v):
    return isinstance(v, str) and v.strip().lower() in PLACEHOLDERS


def retained(path, value):
    return path != "Name" and not path.endswith(".Reference") and not placeholder(value)


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"), map(float, np.quantile(a, [0, .1, .25, .5, .75, .9, 1]))))


def decomposition(catalog, node_rows):
    name = sum(r["non_empty_count"] for r in catalog if r["path"] == "Name")
    refs = sum(r["non_empty_count"] for r in catalog if r["path"].endswith(".Reference"))
    place = sum(r["placeholder_count"] for r in catalog if r["path"] != "Name" and not r["path"].endswith(".Reference"))
    remaining = sum(r["remaining_count"] for r in node_rows)
    raw = sum(r["leaf_count"] for r in node_rows)
    assert raw == name + refs + place + remaining
    return {"nodes": len(node_rows), "total": raw, "parts": {"remaining": remaining, "Name": name, "Reference": refs, "placeholder": place},
            "raw_median": float(np.median([r["leaf_count"] for r in node_rows])),
            "remaining_median": float(np.median([r["remaining_count"] for r in node_rows]))}


def calculate():
    current = read_json(INPUT / "current_semantic_statistics.json")
    with gzip.open(INPUT / "node_diagnostics.jsonl.gz", "rt", encoding="utf-8") as f:
        nodes = [json.loads(line) for line in f]
    assert len(nodes) == current["node_count"] == 93359
    assert len({r["model"] for r in nodes}) == current["model_count"] == 215
    catalog = current["catalog"]
    # Ensures the conservative rule does not double-subtract undefined References.
    assert sum(r["placeholder_count"] for r in catalog if r["path"] == "Name" or r["path"].endswith(".Reference")) == 0
    total = decomposition(catalog, nodes)
    duplicate = sum(r["duplicate_reference_count"] for r in nodes)
    conservative = [r["leaf_count"] - r["duplicate_reference_count"] - r["placeholder_count"] for r in nodes]
    assert all(v >= 0 for v in conservative)
    rules = [
        {"label": "全部非空字段", "sum": total["total"], "median": total["raw_median"]},
        {"label": "只去重复引用副本和明确占位", "sum": sum(conservative), "median": float(np.median(conservative))},
        {"label": "再去 Name 和全部 Reference 的诊断口径", "sum": total["parts"]["remaining"], "median": total["remaining_median"]},
    ]
    for r in rules:
        r["mean"] = r["sum"] / len(nodes)
        r["removed_fraction"] = 1 - r["sum"] / total["total"]
    classes = []
    for cls in current["classes"]:
        kind = cls["component_type"]
        fields = [r for r in catalog if r["component_type"] == kind]
        model_rows = [r for r in current["class_models"] if r["component_type"] == kind]
        common = [r for r in fields if r["coverage"] >= .5 and r["model_coverage"] >= .5]
        bins = Counter()
        for r in fields:
            n = r["models_with_attribute"]
            bins["one" if n == 1 else "two_to_ten" if n <= 10 else "half_or_more" if r["model_coverage"] >= .5 else "eleven_to_under_half"] += 1
        classes.append({"component_type": kind, "label": cls["label"], "fields": len(fields), "models": len(model_rows),
                        "decomposition": decomposition(fields, [r for r in nodes if r["component_type"] == kind]),
                        "model_median_distribution": quantiles([r["remaining"]["median"] for r in model_rows]),
                        "model_median_distribution_without_model6": quantiles([r["remaining"]["median"] for r in model_rows if r["model"] != "model_6"]),
                        "support_bins": {k: bins[k] for k in ("one", "two_to_ten", "eleven_to_under_half", "half_or_more")},
                        "common_fields": len(common),
                        "common_fields_without_name_reference": sum(r["path"] != "Name" and not r["path"].endswith(".Reference") for r in common)})

    evidence = read_json(HERE / "focus_model6_evidence.json")
    model6 = next(r for r in current["models"] if r["model"] == "model_6")
    assert evidence["source_csv_sha256"] == model6["source_csv_sha256"]
    pairs = {tuple(p) for p in evidence["class_path_pairs"]}
    support = {(r["component_type"], r["path"]): r["models_with_attribute"] for r in catalog}
    assert pairs <= support.keys()
    after = {p: n - (p in pairs) for p, n in support.items() if n - (p in pairs) > 0}
    all_paths = {p for c, p in support}
    remaining_paths = {p for c, p in after}
    concentration = {"model": "model_6", "nodes": model6["node_count"], "node_fraction": model6["node_count"] / len(nodes),
                     "all_unique_paths": len(all_paths), "model6_only_unique_paths": len(all_paths - remaining_paths), "paths_without_model6": len(remaining_paths),
                     "all_class_field_pairs": len(support), "single_model_pairs": sum(n == 1 for n in support.values()),
                     "model6_only_pairs": sum(support[p] == 1 for p in pairs), "pairs_without_model6": len(after)}
    assert concentration["model6_only_unique_paths"] == 2720
    assert concentration["single_model_pairs"] == 4141
    # Reuse the exact, already-published representative instances.
    report = (INPUT / "dataset_semantic_report.html").read_text(encoding="utf-8")
    examples = json.loads(re.findall(r"<script(?: [^>]*)?>([\s\S]*?)</script>", report)[0])["examples"]
    selected = [next(r for r in examples if r["model"] == m and r["component_type"] == c)
                for m, c in [("model_0", "IfcWall"), ("model_217", "IfcWall"), ("model_6", "IfcDoor")]]
    for r in selected:
        props = r["properties"]
        r["remaining_count"] = sum(retained(k, v) for k, v in props.items())
        r["same_as_field_name_in_remaining"] = sum(retained(k, v) and isinstance(v, str) and v == k.rsplit(".", 1)[-1] for k, v in props.items())
        r["property_groups"] = dict(Counter(k.split(".")[0] if "." in k else "基础字段" for k in props))
    template = next(r for r in selected if r["model"] == "model_217")
    assert (template["leaf_count"], template["remaining_count"], template["same_as_field_name_in_remaining"]) == (62, 59, 31)
    wall_fields = [r for r in catalog if r["component_type"] == "IfcWall"]
    coverage = [next(r for r in wall_fields if r["path"] == path) for path in
                ["BaseQuantities.Width", "Qto_WallBaseQuantities.Width", "Material", "Pset_WallCommon.IsExternal"]]
    result = {"snapshot_date": current["date"], "analysis_date": "2026-10-27", "model_count": 215, "node_count": len(nodes),
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [INPUT / "current_semantic_statistics.json", INPUT / "node_diagnostics.jsonl.gz", HERE / "focus_model6_evidence.json"]},
              "decomposition": total, "rules": rules, "duplicate_reference_extra_occurrences": duplicate,
              "duplicate_reference_nodes": current["duplicate_reference_nodes"], "classes": classes,
              "model_median_distribution": quantiles([r["remaining"]["median"] for r in current["models"]]),
              "concentration": concentration, "wall_coverage_examples": coverage, "examples": selected,
              "class_models": current["class_models"],
              "notes": ["Counts measure field occurrences, not independent semantic dimensions or correctness.",
                        "Model distributions give each model equal weight, after taking its within-class median.",
                        "Rare full paths may be useful special properties or unmerged synonyms, not automatically missing or useless.",
                        "Current parsed attributes may omit original IFC values; no downstream accuracy or causal software claim is made."]}
    assert sum(r["non_empty_count"] for r in catalog) == total["total"]
    assert sum(total["parts"].values()) == total["total"]
    return result


def style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    font = next((f for f in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei", "Arial Unicode MS"] if f in available), "DejaVu Sans")
    plt.rcParams.update({"font.family": font, "font.size": 11, "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "axes.edgecolor": "#bdcad2", "text.color": "#203d4e",
                         "axes.labelcolor": "#203d4e", "xtick.color": "#476476", "ytick.color": "#203d4e", "svg.fonttype": "none", "svg.hashsalt": "stage1-semantic-focus"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight", facecolor="white", metadata={"Date": "2026-10-27"})
    svg = OUT / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def figures(d):
    style()
    rows = [{"label": "全部构件", "decomposition": d["decomposition"]}] + d["classes"]
    fig, ax = plt.subplots(figsize=(11.8, 5.8), layout="constrained")
    y, left = np.arange(len(rows)), np.zeros(len(rows))
    for key, label, color in [("remaining", "按诊断规则保留", BLUE), ("Name", "名称 Name", PALE), ("Reference", "全部 Reference", SLATE), ("placeholder", "明确未定义占位", ORANGE)]:
        values = np.array([r["decomposition"]["parts"][key] / r["decomposition"]["nodes"] for r in rows])
        ax.barh(y, values, left=left, height=.58, color=color, label=label)
        left += values
    for i, value in enumerate(left):
        ax.text(value + .25, i, f"{value:.2f}", va="center", fontsize=10)
    ax.set_yticks(y, [r["label"] for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, max(left) + 3); ax.set_xlabel("平均每个构件的非空字段数量（个）")
    ax.set_title("图 1｜字段数量由什么组成？", loc="left", pad=39, fontsize=16)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=4, frameon=False, fontsize=10)
    ax.set_axisbelow(True); ax.xaxis.grid(color="#e3ebef", linewidth=.7)
    save(fig, "01_field_composition")

    fig, ax = plt.subplots(figsize=(11.8, 5.5), layout="constrained")
    for i, row in enumerate(d["classes"]):
        q = row["model_median_distribution"]
        ax.plot([q["p10"], q["p90"]], [i, i], color=PALE, linewidth=16, solid_capstyle="butt")
        ax.plot([q["p25"], q["p75"]], [i, i], color=BLUE, linewidth=7, solid_capstyle="butt")
        ax.scatter(q["median"], i, color=ORANGE, marker="D", s=65, zorder=3)
        ax.text(57, i, f'{q["p10"]:g}–{q["p90"]:g}  /  中央值 {q["median"]:g}', va="center", fontsize=11)
    ax.set_yticks(range(len(d["classes"])), [f'{r["label"]}（{r["models"]} 个模型）' for r in d["classes"]])
    ax.invert_yaxis(); ax.set_xlim(0, 84); ax.set_xticks(range(0, 56, 10))
    ax.set_xlabel("各模型该类构件的字段数量中位数（去 Name、全部 Reference 和明确占位后）")
    ax.set_title("图 2｜只比较同一种构件，模型之间仍有明显差异", loc="left", fontsize=16, pad=20)
    ax.text(.01, 1.01, "浅带：中间 80% 的模型　深带：中央一半的模型　◆：各模型中位数再取中位数", transform=ax.transAxes, fontsize=10)
    ax.set_axisbelow(True); ax.xaxis.grid(color="#e3ebef", linewidth=.7)
    save(fig, "02_same_class_model_variation")

    fig, axes = plt.subplots(2, 1, figsize=(11.8, 7.1), gridspec_kw={"height_ratios": [1, 2.3]}, layout="constrained")
    con = d["concentration"]
    ax = axes[0]
    ax.barh([0], [con["model6_only_unique_paths"]], color=ORANGE, height=.46, label="只出现在 model_6")
    ax.barh([0], [con["paths_without_model6"]], left=[con["model6_only_unique_paths"]], color=BLUE, height=.46, label="其他 214 个模型中也出现")
    ax.text(con["model6_only_unique_paths"] / 2, 0, f'{con["model6_only_unique_paths"]:,}（58.0%）', ha="center", va="center", color="white")
    ax.text(con["model6_only_unique_paths"] + con["paths_without_model6"] / 2, 0, f'{con["paths_without_model6"]:,}（42.0%）', ha="center", va="center", color="white")
    ax.set_yticks([0], ["4,688 个完整字段名"]); ax.set_xlim(0, 4688); ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("图 3｜字段词表很大，但高度集中在少数模型", loc="left", fontsize=16, pad=31)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2, frameon=False, fontsize=10)
    ax = axes[1]
    rows = d["classes"]
    left = np.zeros(len(rows))
    for key, label, color in [("one", "仅 1 个模型", ORANGE), ("two_to_ten", "2–10 个模型", SLATE),
                              ("eleven_to_under_half", "超过 10 个、不足一半模型", PALE), ("half_or_more", "至少一半模型", BLUE)]:
        values = np.array([100 * r["support_bins"][key] / r["fields"] for r in rows])
        ax.barh(range(len(rows)), values, left=left, color=color, label=label, height=.57)
        if key == "one":
            for i, v in enumerate(values):
                if v >= 12: ax.text(v / 2, i, f"{v:.1f}%", va="center", ha="center", color="white", fontsize=10)
        left += values
    ax.set_yticks(range(len(rows)), [f'{r["label"]}（{r["fields"]:,} 个字段）' for r in rows]); ax.invert_yaxis()
    ax.set_xlim(0, 100); ax.set_xlabel("该类字段中，分别有多少比例出现在这些数量的模型里（%）")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -.2), ncol=2, frameon=False, fontsize=10)
    save(fig, "03_field_vocabulary_concentration")

    fig, ax = plt.subplots(figsize=(11.8, 5.3), layout="constrained")
    y = np.arange(4)
    values = d["wall_coverage_examples"]
    labels = ["宽度：BaseQuantities.Width", "宽度：Qto_WallBaseQuantities.Width", "材料：Material", "是否外部：Pset_WallCommon.IsExternal"]
    ax.barh(y - .17, [100 * r["coverage"] for r in values], color=BLUE, height=.28, label="全部墙中，有值的墙占比")
    ax.barh(y + .17, [100 * r["model_coverage"] for r in values], color=ORANGE, height=.28, label="215 个模型中，出现过该字段的模型占比")
    for i, r in enumerate(values):
        for delta, key in [(-.17, "coverage"), (.17, "model_coverage")]:
            ax.text(100 * r[key] + .8, i + delta, f'{100*r[key]:.1f}%', va="center", fontsize=10)
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlim(0, 112); ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("比例（%）；两种颜色的分母不同，分别是 28,313 面墙和 215 个模型")
    ax.set_title("图 4｜在很多构件上出现，不代表在很多模型中通用", loc="left", fontsize=16, pad=65)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=1, frameon=False, fontsize=10)
    ax.set_axisbelow(True); ax.xaxis.grid(color="#e3ebef", linewidth=.7)
    save(fig, "04_object_vs_model_coverage")


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>' + ''.join(f'<th>{html.escape(str(h))}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def render(d):
    template = (HERE / "focus_semantic_report.html").read_text(encoding="utf-8")
    parts = d["decomposition"]["parts"]
    def image(name, alt):
        data = base64.b64encode((OUT / (name + ".png")).read_bytes()).decode("ascii")
        return f'<img class="chart" src="data:image/png;base64,{data}" alt="{html.escape(alt)}" loading="lazy">'
    replacements = {
        "__FIGURE_1__": image("01_field_composition", "各类构件的平均字段数，分成诊断保留、Name、Reference和明确占位四部分"),
        "__FIGURE_2__": image("02_same_class_model_variation", "控制构件类别，各模型典型字段数仍有较宽分布，墙的中间80%模型区间为4至40个字段"),
        "__FIGURE_3__": image("03_field_vocabulary_concentration", "4688个完整字段名中2720个只出现在model_6；按构件类别展示字段的模型分布"),
        "__FIGURE_4__": image("04_object_vs_model_coverage", "对比墙的字段在构件中和在模型中出现的比例，说明不同分母的差别"),
        "__COMPOSITION_TABLE__": table(["字段组成", "全部构件中累计出现次数", "占全部非空字段的比例"],
                                      [(label, f"{parts[key]:,}", f'{parts[key]/d["decomposition"]["total"]:.2%}') for key, label in
                                       [("remaining", "诊断规则保留的字段"), ("Name", "Name 名称"), ("Reference", "全部 Reference 引用"), ("placeholder", "其他明确的未定义占位值")]]),
        "__RULES_TABLE__": table(["计数规则", "构件字段数中位数", "平均每构件字段数", "相对原始字段出现次数的减少比例"],
                                [(r["label"], f'{r["median"]:g}', f'{r["mean"]:.2f}', f'{r["removed_fraction"]:.2%}') for r in d["rules"]]),
        "__CLASS_TABLE__": table(["构件", "含该类的模型数", "中间80%模型的典型字段数量范围", "各模型典型数量再取中位数", "常见字段数量", "再去 Name/Reference 后"],
                                [(r["label"], r["models"], f'{r["model_median_distribution"]["p10"]:g}–{r["model_median_distribution"]["p90"]:g}',
                                  f'{r["model_median_distribution"]["median"]:g}', r["common_fields"], r["common_fields_without_name_reference"]) for r in d["classes"]]),
    }
    cases = []
    for row in d["examples"]:
        label = next(c["label"] for c in d["classes"] if c["component_type"] == row["component_type"])
        cases.append(f'<details><summary>{row["model"]} / {label} / node {row["node_id"]}：{row["leaf_count"]} 个字段，诊断口径剩 {row["remaining_count"]} 个</summary>'
                     + table(["字段来源组", "本构件中字段数"], sorted(row["property_groups"].items(), key=lambda x: -x[1]))
                     + '<pre>' + html.escape(json.dumps(row["properties"], ensure_ascii=False, indent=2)) + '</pre></details>')
    replacements["__CASES__"] = ''.join(cases)
    replacements["__MODEL_ROWS__"] = ''.join(f'<tr data-class="{r["component_type"]}"><td>{r["model"]}</td><td>{next(c["label"] for c in d["classes"] if c["component_type"] == r["component_type"])}</td><td>{r["object_count"]}</td><td>{r["leaf"]["median"]:g}</td><td>{r["remaining"]["median"]:g}</td></tr>' for r in sorted(d["class_models"], key=lambda x: (int(x["model"].split("_")[1]), x["component_type"])))
    for key, value in replacements.items():
        assert key in template, key
        template = template.replace(key, value)
    assert not re.search(r"__[A-Z_0-9]+__", template)
    (OUT / "semantic_focus_report.html").write_text(template, encoding="utf-8", newline="\n")


def main():
    OUT.mkdir(exist_ok=True)
    result = calculate()
    write_json(OUT / "semantic_focus_statistics.json", result)
    with (OUT / "model_class_counts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["model", "component_type", "object_count", "raw_field_median", "diagnostic_remaining_median"])
        for r in result["class_models"]:
            writer.writerow([r["model"], r["component_type"], r["object_count"], r["leaf"]["median"], r["remaining"]["median"]])
    figures(result)
    render(result)
    print(f'PASS: {result["model_count"]} models, {result["node_count"]:,} nodes; four figures and a self-contained report in {OUT}')


if __name__ == "__main__":
    main()
