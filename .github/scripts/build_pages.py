"""Copy self-contained reports into the GitHub Pages publication directory."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "_site"
REPORTS = {
    "index.html": "experiments/component_attribute_analysis/dataset_results/dataset_semantic_report.html",
    "model_0.html": "experiments/component_attribute_analysis/single_model_results/model_0/semantic_audit.html",
}

OUTPUT.mkdir(exist_ok=True)
for destination, source in REPORTS.items():
    shutil.copyfile(ROOT / source, OUTPUT / destination)
(OUTPUT / ".nojekyll").touch()
print(f"Prepared {len(REPORTS)} reports in {OUTPUT}")
