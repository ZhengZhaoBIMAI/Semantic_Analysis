"""Copy self-contained reports into the GitHub Pages publication directory."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "_site"
REPORTS = {
    "index.html": "experiments/component_attribute_analysis/dataset_results/dataset_semantic_report.html",
    "model_0.html": "experiments/component_attribute_analysis/single_model_results/model_0/semantic_audit.html",
    "semantic_focus.html": "experiments/component_attribute_analysis/focus_results/semantic_focus_report.html",
    "tabular_characteristics.html": "experiments/component_attribute_analysis/journal_results/tabular_characteristics_report.html",
}

OUTPUT.mkdir(exist_ok=True)
for destination, source in REPORTS.items():
    shutil.copyfile(ROOT / source, OUTPUT / destination)
report = OUTPUT / "semantic_focus.html"
text = report.read_text(encoding="utf-8")
for name in ["semantic_focus_statistics.json", "model_class_counts.csv"]:
    text = text.replace(f'href="{name}"', f'href="focus/{name}"')
report.write_text(text, encoding="utf-8", newline="\n")
focus = OUTPUT / "focus"
focus.mkdir(exist_ok=True)
for name in ["semantic_focus_statistics.json", "model_class_counts.csv"]:
    shutil.copyfile(ROOT / "experiments/component_attribute_analysis/focus_results" / name, focus / name)
tabular_source = ROOT / "experiments/component_attribute_analysis/journal_results"
tabular_site = OUTPUT / "tabular"
tabular_site.mkdir(exist_ok=True)
tabular_report = OUTPUT / "tabular_characteristics.html"
tabular_text = tabular_report.read_text(encoding="utf-8")
for source in tabular_source.iterdir():
    if source.is_file() and source.suffix in {".png", ".svg", ".tif", ".json", ".csv", ".txt"}:
        shutil.copyfile(source, tabular_site / source.name)
        tabular_text = tabular_text.replace(f'href="{source.name}"', f'href="tabular/{source.name}"')
tabular_report.write_text(tabular_text, encoding="utf-8", newline="\n")
(OUTPUT / ".nojekyll").touch()
print(f"Prepared {len(REPORTS)} reports in {OUTPUT}")
