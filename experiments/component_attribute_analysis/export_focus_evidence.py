"""Export hash-verified model_6 field-presence evidence for the focused analysis.

Only the resulting field-name list is published. The external Stage1 project is
read without changing its model data, embeddings, or analysis outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.component_attribute_analysis import audit_dataset_semantics as audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root", type=Path, required=True,
        help="External Stage1 project containing data/processed/model_6.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("focus_model6_evidence.json"),
        help="Destination JSON; defaults to focus_model6_evidence.json beside this script.",
    )
    args = parser.parse_args()

    model = "model_6"
    current_file = Path(__file__).with_name("dataset_results") / "current_semantic_statistics.json"
    current = json.loads(current_file.read_text(encoding="utf-8"))
    metadata = next(record for record in current["models"] if record["model"] == model)
    project_root = args.project_root.resolve()
    csv_file = project_root / "data" / "processed" / model / "target_attribute.csv"
    actual_hash = hashlib.sha256(csv_file.read_bytes()).hexdigest()
    if actual_hash != metadata["source_csv_sha256"]:
        raise ValueError("model_6 input CSV differs from the published analysis; evidence export stopped.")

    audit.ROOT = project_root
    frame, leaves, _ = audit.read_model(model)
    if len(frame) != metadata["node_count"]:
        raise ValueError("model_6 node count differs from the published analysis.")
    pairs = {
        (record.node_type, path)
        for record in frame.itertuples()
        for path, value in leaves[int(record.node_id)].items()
        if audit.one.nonempty(value)
    }
    catalog_pairs = {(field["component_type"], field["path"]) for field in current["catalog"]}
    if not pairs <= catalog_pairs:
        raise ValueError("Exported fields include entries absent from the published catalog.")

    evidence = {
        "evidence_version": 1,
        "model": model,
        "source_csv_sha256": actual_hash,
        "model_nodes": len(frame),
        "class_path_pairs": [list(pair) for pair in sorted(pairs)],
        "method": (
            "SHA-256 of the external processed model_6 target_attribute.csv must match "
            "the published current_semantic_statistics.json source_csv_sha256. "
            "Use audit_dataset_semantics.read_model to verify GUID/node-index alignment "
            "and flatten the current semantic input with the same parsing rules as the "
            "published analysis. Retain nonempty leaf values; 0 and false remain valid. "
            "Store each distinct [component_type, full_property_path] once. "
            "No property values, GUIDs, external file paths, or embeddings are included. "
            "Subtract one from catalog models_with_attribute for each exported pair "
            "to derive field support after removing model_6."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Exported {len(pairs)} distinct class-field pairs for {len(frame)} nodes; input SHA-256 matches publication.")


if __name__ == "__main__":
    main()
