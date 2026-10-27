"""Build the English A/B figure requested for the model_6-excluded cohort.

Uses the existing audited-count pipeline; only rendering and cohort selection
change. Exports 180 mm-wide artwork, Arial 8–11 pt, RGB TIFF/PNG at 600 dpi,
editable SVG, a separate caption, and a self-contained review page.
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
from PIL import Image

from build_tabular_characteristics import LABELS, SOURCE, analyze, check_layout, read_json, write_json

HERE = Path(__file__).resolve().parent
OUT = HERE / "journal_results"
INK, BLUE, GRAY, GRID = "#202020", "#286b89", "#777777", "#e6e6e6"


def selected_data():
    paths = [SOURCE / "current_semantic_statistics.json", SOURCE / "node_diagnostics.jsonl.gz", HERE / "focus_model6_evidence.json"]
    current, evidence = read_json(paths[0]), read_json(paths[2])
    with gzip.open(paths[1], "rt", encoding="utf-8") as handle:
        nodes = [json.loads(line) for line in handle]
    model6 = next(m for m in current["models"] if m["model"] == "model_6")
    assert evidence["source_csv_sha256"] == model6["source_csv_sha256"]
    membership = set(map(tuple, evidence["class_path_pairs"]))
    assert membership <= {(r["component_type"], r["path"]) for r in current["catalog"]}
    assert len(nodes) == current["node_count"] == 93359
    stats, retained = analyze(current, nodes, membership, exclude_model6=True)
    assert stats["model_count"] == 214 and stats["node_count"] == 92338
    assert stats["scopes"][0]["attributes"] == 1968
    assert not any(r["model"] == "model_6" for r in retained)
    return {"snapshot_date": current["date"], "figure_date": "2026-10-27", "selected_cohort": "model_6 excluded",
            "data_stage": "observed full paths before comprehensive synonym harmonization",
            "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            "figure_specification": {"panels": ["A", "B"], "language": "English", "width_mm": 180, "height_mm": 90,
                                     "font": "Arial", "font_size_pt": [8, 11], "raster_dpi": 600,
                                     "panel_b_scale": "linear", "vector": "SVG with editable text and vector marks"},
            "statistics": stats}, retained


def draw(result, rows):
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" not in available:
        raise RuntimeError("Arial is required to reproduce the journal artwork; install Arial and rerun.")
    plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8.5, "axes.titlesize": 9,
                         "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
                         "axes.edgecolor": "#777777", "axes.linewidth": .6,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "svg.fonttype": "none", "svg.hashsalt": "stage1-journal-ab-model6-excluded"})
    fig, axes = plt.subplots(1, 2, figsize=(180 / 25.4, 90 / 25.4), layout="constrained")
    fig.get_layout_engine().set(w_pad=3 / 72, h_pad=4 / 72, wspace=.10)
    scopes = result["statistics"]["scopes"]
    labels = ["All objects" if s["scope"] == "ALL" else LABELS[s["scope"]][1] for s in scopes]
    y = np.arange(len(scopes))
    a, b = axes
    occupancy = [s["occupancy_percent"] for s in scopes]
    a.barh(y, occupancy, height=.56, color=BLUE, edgecolor="none")
    for position, value in zip(y, occupancy):
        a.text(value + .13, position, f"{value:.2f}%", va="center", fontsize=8)
    a.set_xlim(0, 10.8); a.set_xticks([0, 2, 4, 6, 8, 10])
    a.set_xlabel("Non-empty cells (%)")
    a.set_title("Attribute occupancy", loc="left", pad=11)
    boxes = []
    for scope in scopes:
        values = np.asarray([r["leaf_count"] for r in rows if scope["scope"] == "ALL" or r["component_type"] == scope["scope"]])
        q = scope["count_distribution"]
        boxes.append({"q1": q["q1"], "q3": q["q3"], "med": q["median"], "whislo": q["whisker_low"], "whishi": q["whisker_high"],
                      "fliers": values[(values < q["whisker_low"]) | (values > q["whisker_high"])]})
    b.bxp(boxes, positions=y, orientation="horizontal", manage_ticks=False, widths=.50, patch_artist=True,
          boxprops={"facecolor": "#dbe8ee", "edgecolor": BLUE, "linewidth": .75},
          medianprops={"color": INK, "linewidth": 1.0}, whiskerprops={"color": GRAY, "linewidth": .65},
          capprops={"color": GRAY, "linewidth": .65},
          flierprops={"marker": "o", "markersize": 1.6, "markeredgecolor": "none", "markerfacecolor": GRAY, "alpha": .24})
    b.set_xlim(0, 112); b.set_xticks([0, 20, 40, 60, 80, 100])
    b.set_xlabel("Number of attributes per object")
    b.set_title("Attribute-count distribution", loc="left", pad=11)
    for ax, panel in zip(axes, ["A", "B"]):
        ax.set_yticks(y, labels); ax.set_ylim(len(scopes) - .5, -.5)
        ax.set_axisbelow(True); ax.xaxis.grid(color=GRID, linewidth=.5)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.tick_params(axis="x", length=3, width=.6)
        ax.text(-.17, 1.055, panel, transform=ax.transAxes, ha="left", va="bottom", fontsize=11, fontweight="bold")
    check_layout(fig)
    renderer = fig.canvas.get_renderer()
    all_texts = list(fig.texts)
    for ax in axes:
        all_texts += [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.texts + ax.get_xticklabels() + ax.get_yticklabels()
    for text in all_texts:
        if not text.get_visible() or not text.get_text():
            continue
        assert all(ord(char) < 128 for char in text.get_text()), text.get_text()
        box = text.get_window_extent(renderer)
        assert box.x0 >= -.5 and box.y0 >= -.5 and box.x1 <= fig.bbox.x1 + .5 and box.y1 <= fig.bbox.y1 + .5, f"Clipped: {text.get_text()}"
    # Capture directly from the plot renderer as RGB, without an alpha channel.
    fig.set_dpi(600)
    fig.canvas.draw()
    rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())
    rgb.save(OUT / "Fig1.png", dpi=(600, 600))
    rgb.save(OUT / "Fig1.tif", compression="tiff_lzw", dpi=(600, 600))
    fig.savefig(OUT / "Fig1.svg", format="svg", facecolor="white", metadata={"Date": "2026-10-27"})
    svg = OUT / "Fig1.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)
    for path in [OUT / "Fig1.png", OUT / "Fig1.tif"]:
        with Image.open(path) as im:
            assert im.mode == "RGB" and all(abs(float(v) - 600) < .1 for v in im.info["dpi"])
            assert abs(im.width / 600 * 25.4 - 180) < .1 and abs(im.height / 600 * 25.4 - 90) < .1
        assert path.stat().st_size < 10_000_000


def caption(result):
    stats = result["statistics"]
    return (f"Figure 1. Structural sparsity and attribute-count variability of BIM object semantics. "
            f"The analysis includes {stats['node_count']:,} graph objects from {stats['model_count']} models after excluding model_6; "
            "attribute vocabularies are recomputed from the retained models. "
            "(A) Percentage of non-empty cells in a fixed-column representation. The All objects row uses the global union of 1,968 property paths; "
            "each class-specific row uses the property-path union observed within that class. Occupancy equals the observed non-empty key-value occurrences "
            "divided by the number of objects multiplied by the corresponding vocabulary size. Structural blanks can reflect inapplicable or unprovided attributes. "
            "(B) Object-level distributions of non-empty attribute counts on a linear scale. Boxes indicate the interquartile range; central lines indicate medians. "
            "Whiskers extend to the most extreme observations within 1.5 interquartile ranges from each box. Points indicate observations beyond the whiskers; "
            "overlapping points may coincide and do not denote data errors. Zero and false are counted as non-empty. "
            "The figure describes observed full property paths before comprehensive synonym harmonization; it is not a semantic-quality or model-performance assessment.\n")


def export(result):
    write_json(OUT / "figure_statistics.json", result)
    with (OUT / "figure_statistics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["scope", "objects", "eligible_models", "attributes", "non_empty_cells", "potential_cells", "occupancy_percent", "min_count", "q1", "median", "q3", "max_count", "whisker_low", "whisker_high", "outlier_count"])
        for s in result["statistics"]["scopes"]:
            q = s["count_distribution"]
            writer.writerow([s["scope"], s["objects"], s["eligible_models"], s["attributes"], s["observed_cells"], s["potential_cells"], s["occupancy_percent"],
                             q["min"], q["q1"], q["median"], q["q3"], q["max"], q["whisker_low"], q["whisker_high"], q["outlier_count"]])
    (OUT / "Fig1_caption.txt").write_text(caption(result), encoding="utf-8", newline="\n")
    (OUT / "figure_specification.txt").write_text(
        "Selected cohort: model_6 excluded; 214 models and 92,338 objects.\n"
        "Panels: A (occupancy) and B (attribute-count distribution); no panel C.\n"
        "Artwork size: 180 x 90 mm; RGB; Arial 8-11 pt; white background.\n"
        "Raster: 600 dpi PNG and LZW-compressed TIFF; no alpha channel; below 10 MB each.\n"
        "Vector: SVG with editable text and vector marks; no embedded raster images.\n"
        "Caption and cohort details are outside the artwork. B uses a linear scale.\n"
        "Formatting reference: PLOS figure specifications (Arial 8-12 pt; TIFF; 300-600 dpi; separate captions).\n"
        "https://journals.plos.org/plosone/s/figures\n"
        "General vector/legibility reference: Nature figure guide. Journal-specific font sizes differ.\n"
        "https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/\n"
        "No target journal was specified; check its final submission requirements.\n", encoding="utf-8", newline="\n")


def render(result):
    template = (HERE / "journal_figure_report.html").read_text(encoding="utf-8")
    png = base64.b64encode((OUT / "Fig1.png").read_bytes()).decode("ascii")
    template = template.replace("__FIGURE__", f'<a href="Fig1.png"><img class="figure" src="data:image/png;base64,{png}" alt="English panels A and B: attribute occupancy and attribute-count distributions after excluding model_6"></a>')
    rows = []
    for scope in result["statistics"]["scopes"]:
        label = "All objects" if scope["scope"] == "ALL" else LABELS[scope["scope"]][1]
        q = scope["count_distribution"]
        values = [label, f'{scope["objects"]:,}', f'{scope["attributes"]:,}', f'{scope["occupancy_percent"]:.2f}%', f'{q["median"]:g}', f'{q["q1"]:g}-{q["q3"]:g}']
        rows.append('<tr>' + ''.join(f'<td>{html.escape(value)}</td>' for value in values) + '</tr>')
    template = template.replace("__ROWS__", ''.join(rows))
    template = template.replace("__CAPTION__", html.escape(caption(result)))
    (OUT / "tabular_characteristics_report.html").write_text(template, encoding="utf-8", newline="\n")


def main():
    OUT.mkdir(exist_ok=True)
    result, rows = selected_data()
    draw(result, rows)
    export(result)
    render(result)
    print("PASS: model_6 excluded; 214 models; 92,338 objects; English A/B only; 180 x 90 mm; Arial; 600 dpi RGB PNG/TIFF; editable SVG.")


if __name__ == "__main__":
    main()
