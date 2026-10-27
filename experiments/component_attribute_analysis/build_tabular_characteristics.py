"""Reproduce three-panel, publication-scale BIM attribute structure figures.

Inputs are the published audit snapshot, per-object counts and hash-verified
model_6 field-presence evidence. No original models or embeddings are modified.
The results describe observed full paths before comprehensive key harmonization.
"""
from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "dataset_results"
OUT = HERE / "tabular_results"
BLUE, GRAY, INK, GRID = "#247493", "#718592", "#203b49", "#e0e7eb"
LABELS = {"IfcWall": ("墙", "Wall"), "IfcSlab": ("板", "Slab"), "IfcColumn": ("柱", "Column"),
          "IfcDoor": ("门", "Door"), "IfcWindow": ("窗", "Window"), "IfcBeam": ("梁", "Beam"), "IfcSpace": ("空间", "Space")}
BIN_LABELS = ["1", "2–5", "6–10", "11–50", "51–100", ">100"]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def support_bin(n):
    return 0 if n == 1 else 1 if n <= 5 else 2 if n <= 10 else 3 if n <= 50 else 4 if n <= 100 else 5


def percent_label(value):
    return "<0.1" if 0 < value < .1 else f"{value:.1f}"


def support_distribution(fields):
    counts = [0] * len(BIN_LABELS)
    for f in fields:
        counts[support_bin(f["support"])] += 1
    assert sum(counts) == len(fields)
    return [{"label": label, "count": n, "percent": n / len(fields) * 100} for label, n in zip(BIN_LABELS, counts)]


def summarize_counts(values):
    a = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(a, [.25, .5, .75])
    iqr = q3 - q1
    low = a[a >= q1 - 1.5 * iqr].min()
    high = a[a <= q3 + 1.5 * iqr].max()
    return {"min": int(a.min()), "p10": float(np.quantile(a, .1)), "q1": float(q1), "median": float(median),
            "q3": float(q3), "p90": float(np.quantile(a, .9)), "max": int(a.max()), "mean": float(a.mean()),
            "whisker_low": float(low), "whisker_high": float(high), "outlier_count": int(((a < low) | (a > high)).sum())}


def analyze(current, nodes, membership, exclude_model6=False):
    rows = [r for r in nodes if not exclude_model6 or r["model"] != "model_6"]
    fields = []
    for r in current["catalog"]:
        count = r["models_with_attribute"] - (exclude_model6 and (r["component_type"], r["path"]) in membership)
        if count:
            fields.append({"component_type": r["component_type"], "path": r["path"], "support": count})
    result = {"model_count": len({r["model"] for r in rows}), "node_count": len(rows),
              "exclude_model6": exclude_model6, "vocabulary": "refitted to retained models", "scopes": []}
    for kind in ["ALL"] + list(LABELS):
        subset = [r for r in rows if kind == "ALL" or r["component_type"] == kind]
        subset_fields = [r for r in fields if kind == "ALL" or r["component_type"] == kind]
        paths = {r["path"] for r in subset_fields}
        values = [r["leaf_count"] for r in subset]
        total = sum(values)
        cells = len(subset) * len(paths)
        original_p = current["total_unique_paths"] if kind == "ALL" else next(c["distinct_paths"] for c in current["classes"] if c["component_type"] == kind)
        models = len({r["model"] for r in subset})
        assert paths and subset and 0 < total <= cells
        assert all(1 <= f["support"] <= models for f in subset_fields)
        result["scopes"].append({"scope": kind, "objects": len(subset), "eligible_models": models, "attributes": len(paths),
                                 "observed_cells": total, "potential_cells": cells,
                                 "occupancy_percent": total / cells * 100, "structural_blank_percent": 100 - total / cells * 100,
                                 "occupancy_with_baseline_vocabulary_percent": total / (len(subset) * original_p) * 100,
                                 "attribute_class_pairs": len(subset_fields), "count_distribution": summarize_counts(values),
                                 "model_support_bins": support_distribution(subset_fields)})
    assert result["scopes"][0]["objects"] == sum(r["objects"] for r in result["scopes"][1:])
    assert result["scopes"][0]["observed_cells"] == sum(r["observed_cells"] for r in result["scopes"][1:])
    return result, rows


def configure(zh):
    fonts = {f.name for f in font_manager.fontManager.ttflist}
    chinese = next((name for name in ["Microsoft YaHei", "Noto Sans CJK SC", "SimHei"] if name in fonts), "DejaVu Sans")
    plt.rcParams.update({"font.family": chinese if zh else "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 10,
                         "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.edgecolor": "#b7c5cd", "axes.unicode_minus": False, "svg.fonttype": "none",
                         "svg.hashsalt": "stage1-tabular-characteristics"})


def check_layout(fig):
    """Reject clipped titles and overlapping axis tick labels before export."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        assert box.x0 >= -1 and box.x1 <= fig.bbox.x1 + 1, f"Clipped figure text: {text.get_text()}"
    for ax in fig.axes:
        for texts in [ax.get_xticklabels(), ax.get_yticklabels()]:
            boxes = [t.get_window_extent(renderer) for t in texts if t.get_visible() and t.get_text()]
            for i, first in enumerate(boxes):
                assert not any(first.overlaps(second) for second in boxes[i + 1:]), "Overlapping axis tick labels"


def figure(stats, rows, zh=False):
    configure(zh)
    fig = plt.figure(figsize=(7.48, 7.1), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.38, 1], hspace=.10, wspace=.06)
    axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    scopes = stats["scopes"]
    labels = [("全部构件" if zh else "All objects") if r["scope"] == "ALL" else LABELS[r["scope"]][0 if zh else 1] for r in scopes]
    y = np.arange(len(scopes))
    # A uses each scope's observed attribute union; it is not semantic missingness.
    occupancy = [r["occupancy_percent"] for r in scopes]
    axa.barh(y, occupancy, height=.58, color=BLUE)
    for i, value in enumerate(occupancy):
        axa.text(value + .13, i, f"{value:.2f}%", va="center", fontsize=8)
    axa.set_yticks(y, labels); axa.invert_yaxis(); axa.set_xlim(0, 10.8); axa.set_xticks([0, 2, 4, 6, 8, 10])
    axa.set_xlabel("非空单元格比例（%）" if zh else "Non-empty cells (%)")
    axa.set_title("A  固定列表格的非空比例" if zh else "A  Fixed-column occupancy", loc="left", pad=12)
    axa.set_axisbelow(True); axa.xaxis.grid(color=GRID, linewidth=.6)
    # B is a standard Tukey boxplot of objects, not model medians. All fliers shown.
    boxes = []
    for s in scopes:
        a = np.asarray([r["leaf_count"] for r in rows if s["scope"] == "ALL" or r["component_type"] == s["scope"]])
        q = s["count_distribution"]
        boxes.append({"q1": q["q1"], "med": q["median"], "q3": q["q3"], "whislo": q["whisker_low"], "whishi": q["whisker_high"],
                      "fliers": a[(a < q["whisker_low"]) | (a > q["whisker_high"])]})
    axb.bxp(boxes, positions=y, orientation="horizontal", patch_artist=True, widths=.52,
            boxprops={"facecolor": "#d4e5ee", "edgecolor": BLUE, "linewidth": .8},
            medianprops={"color": INK, "linewidth": 1.3}, whiskerprops={"color": GRAY, "linewidth": .7},
            capprops={"color": GRAY, "linewidth": .7},
            flierprops={"marker": "o", "markersize": 1.7, "markerfacecolor": GRAY, "markeredgecolor": "none", "alpha": .18, "rasterized": True},
            manage_ticks=False)
    axb.set_yticks(y, labels); axb.invert_yaxis(); axb.set_xscale("log", base=2)
    axb.set_xlim(.8, 512); axb.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    axb.set_xlabel("每构件属性数（对数坐标）" if zh else "Attributes per object (log scale)")
    axb.set_title("B  每构件属性数量的分布" if zh else "B  Attributes per object", loc="left", pad=12)
    axb.set_axisbelow(True); axb.xaxis.grid(color=GRID, linewidth=.6)
    # C pools class-field pairs; no pooling of distinct full names is implied.
    dist = scopes[0]["model_support_bins"]
    percentages = [r["percent"] for r in dist]
    bars = axc.bar(range(len(dist)), percentages, color=[BLUE] + [GRAY] * (len(dist) - 1), width=.58)
    for bar, r in zip(bars, dist):
        axc.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.1, f'{percent_label(r["percent"])}%\n(n={r["count"]:,})',
                 ha="center", va="bottom", fontsize=8)
    axc.set_xticks(range(len(dist)), BIN_LABELS)
    axc.set_ylim(0, 97); axc.set_yticks([0, 20, 40, 60, 80])
    axc.set_ylabel("类别—属性组合占比（%）" if zh else "Attribute–class pairs (%)")
    axc.set_xlabel("某属性出现过的模型数量" if zh else "Number of models in which an attribute occurs")
    axc.set_title("C  属性在模型之间的共享程度" if zh else "C  Attribute prevalence across models", loc="left", pad=14)
    axc.set_axisbelow(True); axc.yaxis.grid(color=GRID, linewidth=.6)
    if zh:
        title = f'BIM 属性的表格结构特征 · {stats["model_count"]} 个模型，{stats["node_count"]:,} 个构件'
        note = "当前审计快照；尚未全面统一同义字段\n" + ("排除 model_6 后重新确定字段并集" if stats["exclude_model6"] else "全部可用模型；使用观察到的字段并集")
    else:
        title = f'BIM attribute structure | {stats["model_count"]} models; {stats["node_count"]:,} objects'
        note = "Audited snapshot before full key harmonization\n" + ("model_6 excluded; vocabulary refitted" if stats["exclude_model6"] else "All available models; observed attribute vocabulary")
    fig.suptitle(title + "\n" + note, fontsize=9.5, linespacing=1.4)
    check_layout(fig)
    stem = ("tabular_characteristics_without_model6" if stats["exclude_model6"] else "tabular_characteristics") + ("_zh" if zh else "_en")
    fig.savefig(OUT / f"{stem}.png", dpi=400, facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white", metadata={"Date": "2026-10-27"})
    svg = OUT / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    return stem


def prevalence_facets(stats, zh=False):
    configure(zh)
    fig, axes = plt.subplots(4, 2, figsize=(7.48, 8.4), sharex=True, sharey=True, layout="constrained")
    for ax, scope in zip(axes.flat, stats["scopes"]):
        dist = scope["model_support_bins"]
        vals = [r["percent"] for r in dist]
        ax.bar(range(len(dist)), vals, color=[BLUE] + [GRAY] * 5, width=.62)
        for i, value in enumerate(vals):
            if value > 0:
                ax.text(i, value + 1.4, percent_label(value), ha="center", fontsize=7)
        name = ("全部类别合计" if zh else "All classes pooled") if scope["scope"] == "ALL" else LABELS[scope["scope"]][0 if zh else 1]
        numbers = f'K={scope["attribute_class_pairs"]:,}' if scope["scope"] == "ALL" else f'M={scope["eligible_models"]}, K={scope["attribute_class_pairs"]:,}'
        ax.set_title(f'{name} | {numbers}', loc="left", fontsize=9)
        ax.set_xticks(range(6), BIN_LABELS); ax.set_ylim(0, 106); ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_axisbelow(True); ax.yaxis.grid(color=GRID, linewidth=.5)
    fig.supxlabel("属性出现过的模型数量" if zh else "Number of models in which an attribute occurs", fontsize=9)
    fig.supylabel("各范围内类别—属性组合占比（%）" if zh else "Within-scope attribute–class pairs (%)", fontsize=9)
    fig.suptitle("补充图：各构件类别的属性共享程度" if zh else "Supplement: attribute prevalence by object class", fontsize=11)
    check_layout(fig)
    stem = "attribute_prevalence_by_class_" + ("zh" if zh else "en")
    fig.savefig(OUT / f"{stem}.png", dpi=400, facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", facecolor="white", metadata={"Date": "2026-10-27"})
    svg = OUT / f"{stem}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def export_csv(result):
    with (OUT / "tabular_characteristics_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["variant", "scope", "objects", "eligible_models", "attribute_union_size", "observed_cells", "potential_cells", "occupancy_percent", "blank_percent", "attribute_count_median", "q1", "q3", "min", "max", "attribute_class_pairs", "single_model_pairs"])
        for label in ["all_models", "without_model6"]:
            for s in result[label]["scopes"]:
                q = s["count_distribution"]
                writer.writerow([label, s["scope"], s["objects"], s["eligible_models"], s["attributes"], s["observed_cells"], s["potential_cells"], s["occupancy_percent"], s["structural_blank_percent"], q["median"], q["q1"], q["q3"], q["min"], q["max"], s["attribute_class_pairs"], s["model_support_bins"][0]["count"]])
    with (OUT / "attribute_model_support.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["variant", "scope", "model_count_bin", "class_field_pair_count", "percent"])
        for label in ["all_models", "without_model6"]:
            for s in result[label]["scopes"]:
                for b in s["model_support_bins"]:
                    writer.writerow([label, s["scope"], b["label"], b["count"], b["percent"]])


def captions(stats, excluded):
    return f'''Figure. Tabular structure of observed BIM object attributes in the Stage1 audit snapshot (2026-10-14; {stats['model_count']} models and {stats['node_count']:,} graph objects). (A) Non-empty cell percentage after expanding the observed attribute union into fixed columns. The first row uses the global attribute union; each class-specific row uses the union observed in that class. Occupancy is the number of observed non-empty key–value occurrences divided by the number of objects multiplied by the corresponding union size. Structural blanks include inapplicable and unprovided attributes and do not measure semantic incompleteness. (B) Object-level distributions of non-empty attribute counts. Boxes span the first and third quartiles, the central line denotes the median, and whiskers extend to the most extreme observed values within 1.5 interquartile ranges from the box. Dots show observations beyond the whiskers; overlapping points may coincide. The horizontal axis is logarithmic (base 2), and all observations remain represented. (C) Distribution of the number of models supporting each object-class–attribute pair, pooled across the seven object classes. Each pair is counted once, and support requires at least one object with a non-empty value in a model; the eligible model count varies by class. Counts and percentages are shown above the bars. These panels describe observed full property paths before comprehensive synonym or schema harmonization, not a final cleaned WP1 table, semantic quality, or downstream model performance. Zero and false are non-empty values. The results provide motivation for flexible attribute representations, not comparative validation of a tokenization method.

Sensitivity figure. The same definitions, axes and boxplot conventions are used after excluding model_6. The retained {excluded['model_count']} models contain {excluded['node_count']:,} objects. Attribute unions and model support are recomputed from retained models; columns found only in model_6 are removed. Consequently, this is a refitted-vocabulary sensitivity analysis, not deletion of rows under a frozen global schema. CSV and JSON also report occupancy under the frozen baseline vocabulary for comparison.
'''


def render(result):
    a, b = result["all_models"], result["without_model6"]
    template = (HERE / "tabular_characteristics_report.html").read_text(encoding="utf-8")
    for marker, stem in [("__MAIN_FIGURE__", "tabular_characteristics_zh"), ("__SENSITIVITY_FIGURE__", "tabular_characteristics_without_model6_zh"), ("__PREVALENCE_FIGURE__", "attribute_prevalence_by_class_zh")]:
        png = base64.b64encode((OUT / (stem + ".png")).read_bytes()).decode("ascii")
        template = template.replace(marker, f'<img class="figure" src="data:image/png;base64,{png}" alt="{stem}" loading="lazy">')
    rows = []
    for sa, sb in zip(a["scopes"], b["scopes"]):
        name = "全部构件（全局字段）" if sa["scope"] == "ALL" else LABELS[sa["scope"]][0]
        rows.append('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in [name, f'{sa["objects"]:,}', f'{sa["attributes"]:,}', f'{sa["occupancy_percent"]:.2f}%', f'{sb["objects"]:,}', f'{sb["attributes"]:,}', f'{sb["occupancy_percent"]:.2f}%']) + '</tr>')
    template = template.replace("__STAT_ROWS__", ''.join(rows))
    files = ["tabular_characteristics_en.png", "tabular_characteristics_en.svg", "tabular_characteristics_zh.png", "tabular_characteristics_zh.svg",
             "tabular_characteristics_without_model6_en.png", "tabular_characteristics_without_model6_en.svg", "tabular_characteristics_without_model6_zh.png", "tabular_characteristics_without_model6_zh.svg",
             "attribute_prevalence_by_class_en.png", "attribute_prevalence_by_class_en.svg", "attribute_prevalence_by_class_zh.png", "attribute_prevalence_by_class_zh.svg",
             "figure_captions_en.txt", "tabular_characteristics_summary.csv", "attribute_model_support.csv", "tabular_characteristics_statistics.json"]
    template = template.replace("__DOWNLOADS__", ''.join(f'<li><a href="{name}" download>{name}</a></li>' for name in files))
    template = template.replace("__EN_CAPTION__", html.escape((OUT / "figure_captions_en.txt").read_text(encoding="utf-8")))
    assert "__MAIN_" not in template and "__STAT_" not in template and "__DOWNLOADS__" not in template
    (OUT / "tabular_characteristics_report.html").write_text(template, encoding="utf-8", newline="\n")


def main():
    OUT.mkdir(exist_ok=True)
    current_path = SOURCE / "current_semantic_statistics.json"
    nodes_path = SOURCE / "node_diagnostics.jsonl.gz"
    evidence_path = HERE / "focus_model6_evidence.json"
    current, evidence = read_json(current_path), read_json(evidence_path)
    with gzip.open(nodes_path, "rt", encoding="utf-8") as f:
        nodes = [json.loads(line) for line in f]
    meta = next(m for m in current["models"] if m["model"] == "model_6")
    assert evidence["source_csv_sha256"] == meta["source_csv_sha256"]
    membership = set(map(tuple, evidence["class_path_pairs"]))
    assert len(nodes) == current["node_count"] == 93359
    assert sum(r["leaf_count"] for r in nodes) == sum(r["non_empty_count"] for r in current["catalog"])
    assert membership <= {(r["component_type"], r["path"]) for r in current["catalog"]}
    all_stats, all_rows = analyze(current, nodes, membership)
    excluded_stats, excluded_rows = analyze(current, nodes, membership, exclude_model6=True)
    assert all_stats["scopes"][0]["attributes"] == 4688
    assert excluded_stats["scopes"][0]["attributes"] == 1968
    assert all_stats["scopes"][0]["model_support_bins"][0]["count"] == 4141
    result = {"snapshot_date": current["date"], "figure_date": "2026-10-27", "data_stage": "audited observed attributes before comprehensive key harmonization",
              "methods": {"population": current["method"]["population"], "nulls": current["method"]["nulls"],
                          "keys": "Full flattened property paths; synonyms not comprehensively merged; a path in different classes is counted once per class in panel C.",
                          "occupancy": "Observed non-empty cells / (objects * observed attribute-union size); structural blanks are not semantic incompleteness.",
                          "boxplot": "Object-weighted standard Tukey boxplot; log2 horizontal count axis; all fliers shown; no model-level averaging.",
                          "support": "Number of eligible models with at least one object holding the field; eligibility restricted to models containing the object class.",
                          "sensitivity": "Exclude model_6 then refit vocabulary, denominators and model support; frozen-vocabulary density also retained.",
                          "scope": "Descriptive census of the observed snapshot; no inferential error bars or causal claim."},
              "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [current_path, nodes_path, evidence_path]},
              "all_models": all_stats, "without_model6": excluded_stats}
    write_json(OUT / "tabular_characteristics_statistics.json", result)
    export_csv(result)
    for zh in [False, True]:
        figure(all_stats, all_rows, zh)
        figure(excluded_stats, excluded_rows, zh)
        prevalence_facets(all_stats, zh)
    (OUT / "figure_captions_en.txt").write_text(captions(all_stats, excluded_stats), encoding="utf-8", newline="\n")
    render(result)
    print(f'PASS: {len(nodes):,} objects; three-panel figures in EN/ZH; exclusion sensitivity; class-specific prevalence; source statistics and captions.')


if __name__ == "__main__":
    main()
