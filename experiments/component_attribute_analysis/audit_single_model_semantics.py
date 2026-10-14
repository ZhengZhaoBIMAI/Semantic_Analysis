"""Audit one Stage1 model without changing its source data or embeddings.

Reconstructs property text with the current embedding code, profiles every leaf,
and compares matched original IFC instances and their types when available.
Run with Python 3.14 (IfcOpenShell is installed in that interpreter here).
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_processing import semantics_Embed as semantic

LABELS = {
    "IfcWall": "墙", "IfcSlab": "板", "IfcColumn": "柱", "IfcDoor": "门",
    "IfcWindow": "窗", "IfcBeam": "梁", "IfcSpace": "空间",
}
ORDER = ["IfcWall", "IfcSlab", "IfcColumn", "IfcDoor", "IfcWindow", "IfcBeam", "IfcSpace"]
NULLS = {"", "none", "null", "nan", "n/a", "na", "{}", "[]"}
MEANINGS = {
    "Name": "构件名称（常含族名、类型名及实例编号）",
    "Description": "说明", "Tag": "实例编号", "PredefinedType": "IFC 预定义类型",
    "GlobalId": "构件唯一标识", "node_id": "图节点编号", "type": "原始 IFC 类别",
    "has_geometry": "是否有几何", "Material": "材料名称或材料层集名称",
    "OverallHeight": "整体高度", "OverallWidth": "整体宽度", "OperationType": "门的开启方式",
    "PartitioningType": "窗的分格方式", "LongName": "空间完整名称",
    "CompositionType": "空间组成类型", "ElevationWithFlooring": "含地面完成面的标高",
    "Reference": "类型引用/规格名称", "IsExternal": "是否外部构件",
    "LoadBearing": "是否承重", "ExtendToStructure": "是否延伸至结构",
    "Span": "跨度", "Slope": "坡度", "PitchAngle": "倾角",
    "ThermalTransmittance": "传热系数", "FireRating": "耐火等级",
    "AcousticRating": "隔声等级", "ObjectType": "构件对象类型",
    "ConstructionType": "构造类型", "GlazingAreaFraction": "玻璃面积比例",
    "ThermalResistance": "热阻", "Infiltration": "渗透/漏风参数",
    "ElementType": "类型定义中的构件类型", "Category": "材料类别",
    "LayerThickness": "材料层厚度", "LayerCount": "材料层数",
    "PanelOperation": "门扇开启方式", "PanelPosition": "门扇位置",
    "Roll": "梁截面转角", "ContainerName": "所在楼层名称",
    "ContainerType": "空间容器类别", "Elevation": "楼层标高",
    "Manufacturer": "制造商（仅为文件中的声明值）",
    "ConstituentMaterialNames": "组成材料名称列表", "ConstituentNames": "材料组成部分名称",
    "ConstituentFractions": "材料组成比例", "ConstituentCount": "材料组成部分数量",
    "ConstituentCategories": "组成材料类别", "ConstituentSetName": "材料组成集名称",
}


def nonempty(value):
    if value is None:
        return False
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() not in NULLS
    if isinstance(value, (dict, list, tuple)):
        return bool(value)
    return True


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if hasattr(value, "is_a"):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def leaves(value, prefix):
    if isinstance(value, dict):
        for key, val in value.items():
            yield from leaves(val, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        yield prefix, list(value)
    else:
        yield prefix, plain(value)


def group(path):
    if path.startswith("Pset_"):
        return "Pset 属性集"
    if path.startswith("Qto_") or path.startswith("BaseQuantities"):
        return "工程量"
    if path == "Material" or path.startswith("Material."):
        return "材料"
    return "IFC 基础属性"


def semantic_path(source_path):
    return {"Material.Name": "Material", "Type.PredefinedType": "PredefinedType",
            "Type.OperationType": "OperationType", "Type.PartitioningType": "PartitioningType"}.get(source_path, source_path)


def same_value(a, b):
    if isinstance(a, (float, int)) and not isinstance(a, bool) and isinstance(b, (float, int)) and not isinstance(b, bool):
        return abs(a - b) <= 0.000501  # Existing extractor rounds to 3 decimals.
    return a == b


def meaning(path):
    return MEANINGS.get(path.rsplit(".", 1)[-1], "")


def display(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def numeric(value):
    if isinstance(value, bool):
        return None
    try:
        val = float(value)
        return val if math.isfinite(val) else None
    except (ValueError, TypeError):
        return None


def summarize_catalog(records, counts):
    out = []
    for (kind, path), values in sorted(records.items(), key=lambda x: (ORDER.index(x[0][0]), x[0][1])):
        populated = [v for v in values.values() if nonempty(v)]
        value_counts = Counter(display(v) for v in populated)
        numbers = [numeric(v) for v in populated]
        numbers = [v for v in numbers if v is not None]
        categories = sorted({"布尔" if isinstance(v, bool) else "数值" if numeric(v) is not None else "文本" if isinstance(v, str) else "列表" for v in populated})
        out.append({
            "component_type": kind, "component_label": LABELS[kind], "path": path,
            "meaning": meaning(path), "group": group(path), "object_count": counts[kind],
            "present_count": len(values), "non_empty_count": len(populated),
            "coverage": len(populated) / counts[kind],
            "missing_count": counts[kind] - len(populated),
            "null_when_present_count": len(values) - len(populated),
            "unique_values": len(value_counts), "value_type": "/".join(categories),
            "top_values": [{"value": json.loads(v), "count": n} for v, n in value_counts.most_common(8)],
            "min": min(numbers) if numbers else None,
            "median": float(np.median(numbers)) if numbers else None,
            "max": max(numbers) if numbers else None,
            "semantic_included": path.split(".")[0] not in semantic.PROPERTY_IGNORE,
            "constant_in_this_class": len(value_counts) == 1,
        })
    return out


def audit_ifc(path, frame, semantic_by_id):
    if path is None or not path.exists():
        return {"available": False, "note": "原始 IFC 不可用，只统计当前 CSV，无法区分源文件缺失与提取遗漏。"}
    import ifcopenshell
    import ifcopenshell.util.element as element

    model = ifcopenshell.open(str(path))
    counts = Counter(frame["node_type"])
    fields = defaultdict(dict)
    origin = defaultdict(set)
    occurrence_origins = defaultdict(Counter)
    original_examples = {}
    raw_counts = Counter()
    gaps = []
    material_classes = Counter()
    matched = 0
    units = []
    for u in model.by_type("IfcUnitAssignment")[0].Units:
        if u.is_a("IfcSIUnit"):
            units.append({"unit_type": u.UnitType, "name": u.Name, "prefix": u.Prefix})
        else:
            units.append({"unit_type": getattr(u, "UnitType", None), "name": getattr(u, "Name", None)})

    for row in frame.to_dict("records"):
        guid, kind, node_id = row["GlobalId"], row["node_type"], int(row["node_id"])
        obj = model.by_guid(guid)
        if obj is None:
            raise ValueError(f"Original IFC missing target GUID {guid}")
        matched += 1
        source = {}
        source_origins = {}

        def add(key, val, where):
            val = plain(val)
            if nonempty(val):
                source[key] = val
                source_origins[key] = where
                fields[(kind, key)][node_id] = val
                origin[(kind, key)].add(where)
                occurrence_origins[(kind, key)][where] += 1

        info = obj.get_info()
        for key, val in info.items():
            if key in {"id", "type", "GlobalId", "OwnerHistory", "ObjectPlacement", "Representation", "Tag"}:
                continue
            if isinstance(val, (str, float, int, bool)):
                add(key, val, "IFC 实例基础属性")
        own_psets = element.get_psets(obj, should_inherit=False)
        all_psets = element.get_psets(obj, should_inherit=True)
        for pset, props in all_psets.items():
            definition = model.by_id(props["id"]) if props.get("id") else None
            prefix = definition.is_a() if definition is not None and not definition.is_a("IfcPropertySet") and not definition.is_a("IfcElementQuantity") else pset
            for key, val in props.items():
                if key == "id":
                    continue
                where = "实例 Pset/Qto" if key in own_psets.get(pset, {}) else "类型继承 Pset/Qto"
                for subpath, leaf in leaves(val, f"{prefix}.{key}"):
                    add(subpath, leaf, where)
        typ = element.get_type(obj)
        if typ is not None:
            for key in ("Name", "Description", "PredefinedType", "ElementType", "ConstructionType", "OperationType", "PartitioningType"):
                add(f"Type.{key}", getattr(typ, key, None), "IFC 类型定义")
        mat = element.get_material(obj, should_inherit=True)
        if mat is not None:
            material_classes[mat.is_a()] += 1
            if not mat.is_a("IfcMaterial"):
                components = element.get_materials(obj, should_inherit=True)
                add("Material.ComponentNames", [c.Name for c in components if c and nonempty(c.Name)], "IFC 材料定义")
            if mat.is_a("IfcMaterial"):
                for key in ("Name", "Category", "Description"):
                    add(f"Material.{key}", getattr(mat, key, None), "IFC 材料定义")
            if mat.is_a("IfcMaterialConstituentSet"):
                constituents = mat.MaterialConstituents or []
                add("Material.ConstituentCount", len(constituents), "IFC 材料组成集")
                add("Material.ConstituentSetName", mat.Name, "IFC 材料组成集")
                add("Material.ConstituentMaterialNames", [c.Material.Name for c in constituents if c.Material and nonempty(c.Material.Name)], "IFC 材料组成集")
                add("Material.ConstituentNames", [c.Name for c in constituents if nonempty(c.Name)], "IFC 材料组成集")
                add("Material.ConstituentFractions", [c.Fraction for c in constituents if nonempty(c.Fraction)], "IFC 材料组成集")
                add("Material.ConstituentCategories", [c.Material.Category for c in constituents if c.Material and nonempty(c.Material.Category)], "IFC 材料组成集")
            layer_set = mat.ForLayerSet if mat.is_a("IfcMaterialLayerSetUsage") else mat if mat.is_a("IfcMaterialLayerSet") else None
            if layer_set is not None:
                add("Material.LayerCount", len(layer_set.MaterialLayers), "IFC 材料层")
                add("Material.Layers", [{"name": getattr(layer.Material, "Name", None), "thickness": layer.LayerThickness} for layer in layer_set.MaterialLayers], "IFC 材料层")
        container = element.get_container(obj)
        if container is not None:
            add("Spatial.ContainerName", container.Name, "IFC 空间关系")
            add("Spatial.ContainerType", container.is_a(), "IFC 空间关系")
            add("Spatial.Elevation", getattr(container, "Elevation", None), "IFC 空间关系")
        for key, val in source.items():
            if not nonempty(semantic_by_id[node_id].get(semantic_path(key))):
                gaps.append({"node_id": node_id, "guid": guid, "component_type": kind, "path": key, "value": val, "origin": source_origins[key]})
        original_examples.setdefault(kind, {"node_id": node_id, "guid": guid, "original_fields": source})

    for obj in model.by_type("IfcProduct"):
        kind = obj.is_a()
        if kind == "IfcWallStandardCase":
            kind = "IfcWall"
        if kind == "IfcSlabStandardCase":
            kind = "IfcSlab"
        if kind in LABELS:
            raw_counts[kind] += 1
    catalog = summarize_catalog(fields, counts)
    missing_origins = defaultdict(Counter)
    for gap in gaps:
        missing_origins[(gap["component_type"], gap["path"])][gap["origin"]] += 1
    for item in catalog:
        key = (item["component_type"], item["path"])
        item["origin"] = sorted(origin[key])
        item["source_count_by_origin"] = dict(occurrence_origins[key])
        item["missing_count_by_origin"] = dict(missing_origins[key])
        item["semantic_comparison_path"] = semantic_path(item["path"])
        item["semantic_nonempty_count"] = sum(nonempty(semantic_by_id[i].get(semantic_path(item["path"]))) for i in fields[key])
        item["missing_from_semantic_count"] = item["non_empty_count"] - item["semantic_nonempty_count"]
        item["different_value_count"] = sum(nonempty(semantic_by_id[i].get(semantic_path(item["path"]))) and not same_value(v, semantic_by_id[i][semantic_path(item["path"])]) for i, v in fields[key].items())
        current_counts = Counter(display(semantic_by_id[i][semantic_path(item["path"])]) for i in fields[key] if nonempty(semantic_by_id[i].get(semantic_path(item["path"]))))
        item["semantic_top_values"] = [{"value": json.loads(v), "count": n} for v, n in current_counts.most_common(8)]
        item["semantic_coverage"] = item["semantic_nonempty_count"] / counts[item["component_type"]]
        item["representation_note"] = (
            "Material.Name 已以 Material 字段保留" if item["path"] == "Material.Name" else
            "类型名称常已部分出现在 Name/Reference，未单列" if item["path"] in {"Type.Name", "ObjectType"} else
            "空间关系单独审查，不等同节点属性" if item["path"].startswith("Spatial.") else ""
        )
    type_pset_losses = [r for r in catalog if r["missing_count_by_origin"].get("类型继承 Pset/Qto", 0)]
    return {
        "available": True, "audit_version": 2, "path": str(path), "schema": model.schema,
        "target_guids_matched": matched, "units": units, "catalog": catalog,
        "type_pset_losses": type_pset_losses, "material_representation_counts": dict(material_classes),
        "source_target_class_counts": dict(raw_counts), "examples": original_examples,
        "quantity_set_count": len(model.by_type("IfcElementQuantity")),
        "material_layer_count": len(model.by_type("IfcMaterialLayer")),
        "material_constituent_count": len(model.by_type("IfcMaterialConstituent")) if model.schema != "IFC2X3" else 0,
        "gap_instances": gaps,
    }


def analyze(model_name, ifc_path):
    folder = ROOT / "data" / "processed" / model_name
    frame = pd.read_csv(folder / "target_attribute.csv")
    index = pd.read_csv(folder / "node_index.csv")
    assert not frame["node_id"].duplicated().any()
    assert not frame["GlobalId"].duplicated().any()
    assert len(frame) == len(index)
    types = semantic.load_node_type_series(folder, frame)
    properties = semantic.drop_ignored_property_columns(frame)
    counts = Counter(types)
    records = defaultdict(dict)
    semantic_by_id = {}
    node_records = []
    text_by_id = {}
    for position, (_, row) in enumerate(frame.iterrows()):
        node_id, kind = int(row["node_id"]), types.iloc[position]
        text = semantic.combine_properties(properties.iloc[position])
        content = json.loads(text)
        text_by_id[node_id] = text
        leaf_content = dict(leaves(content, ""))
        semantic_by_id[node_id] = leaf_content
        for key, value in row.items():
            if not nonempty(value):
                continue
            parsed = semantic._parse_property_value(value)
            for path, leaf in leaves(parsed, key):
                records[(kind, path)][node_id] = leaf
        meaningful = {key: val for key, val in leaf_content.items() if nonempty(val)}
        node_records.append({
            "node_id": node_id, "guid": row["GlobalId"], "component_type": kind,
            "name": row["Name"], "semantic_leaf_count": len(meaningful),
            "semantic_text_chars": len(text), "estimated_tokens": semantic.estimate_tokens(text),
            "would_truncate_at_default_limit": semantic.estimate_tokens(text) > semantic.DEFAULT_MAX_INPUT_TOKENS,
            "semantic_properties": content, "semantic_text": text,
        })
    catalog = summarize_catalog(records, counts)
    summaries = []
    for kind in ORDER:
        selected = [r for r in catalog if r["component_type"] == kind and r["semantic_included"] and r["non_empty_count"]]
        nodes = [r for r in node_records if r["component_type"] == kind]
        if not nodes:
            continue
        meaningful_without_name = []
        pset_references = []
        duplicate_groups = defaultdict(list)
        for item in selected:
            signature = tuple((i, display(v)) for i, v in sorted(records[(kind, item["path"])].items()) if nonempty(v))
            duplicate_groups[signature].append(item["path"])
        duplicate_fields = [v for v in duplicate_groups.values() if len(v) > 1]
        for node in nodes:
            leaves_now = {k: v for k, v in semantic_by_id[node["node_id"]].items() if nonempty(v)}
            meaningful_without_name.append(display({k: v for k, v in leaves_now.items() if k != "Name"}))
            refs = [v for k, v in leaves_now.items() if k.endswith(".Reference")]
            pset_references.append(len(refs) - len(set(map(display, refs))))
        summaries.append({
            "component_type": kind, "label": LABELS[kind], "object_count": len(nodes),
            "semantic_distinct_leaf_paths": len(selected),
            "varying_leaf_paths": sum(r["unique_values"] > 1 for r in selected),
            "constant_leaf_paths": sum(r["unique_values"] == 1 for r in selected),
            "per_object_leaf_min": min(n["semantic_leaf_count"] for n in nodes),
            "per_object_leaf_median": float(np.median([n["semantic_leaf_count"] for n in nodes])),
            "per_object_leaf_max": max(n["semantic_leaf_count"] for n in nodes),
            "semantic_text_unique_count": len(set(n["semantic_text"] for n in nodes)),
            "property_combinations_excluding_name": len(set(meaningful_without_name)),
            "duplicate_leaf_groups": duplicate_fields,
            "repeated_reference_values_total": sum(pset_references),
            "attribute_paths": [r["path"] for r in selected],
        })
    embeddings = pd.read_csv(folder / "semantic_embedding.csv")
    assert len(embeddings) == len(frame)
    classes = semantic.DEFAULT_NODE_TYPE_CLASSES
    expected_labels = semantic.build_node_type_one_hot(types.tolist(), classes, model_name=model_name)
    sizes = Counter()
    property_tail_norms = []
    for position, embedding_row in embeddings.iterrows():
        assert int(embedding_row["node_id"]) == int(frame.iloc[position]["node_id"])
        assert embedding_row["guid"] == frame.iloc[position]["GlobalId"]
        vector = json.loads(embedding_row["semantic_embedding"])
        assert np.isfinite(vector).all()
        assert vector[:len(classes)] == expected_labels[position]
        sizes[len(vector)] += 1
        property_tail_norms.append(float(np.linalg.norm(vector[len(classes):])))
    joined = frame.copy()
    joined["node_type"] = types
    original = audit_ifc(ifc_path, joined, semantic_by_id)
    meta = json.loads((ROOT / "data" / "graphs" / model_name / "metadata.json").read_text(encoding="utf-8"))
    return {
        "model": model_name, "report_date": "2026-10-14", "total_nodes": len(frame),
        "source_csv_column_count": len(frame.columns),
        "active_semantic_top_level_columns": sorted({k.split(".")[0] for content in semantic_by_id.values() for k, v in content.items() if nonempty(v)}),
        "semantic_unique_leaf_path_count": len({r["path"] for r in catalog if r["semantic_included"] and r["non_empty_count"]}),
        "class_attribute_combination_count": sum(r["semantic_included"] and bool(r["non_empty_count"]) for r in catalog),
        "ignored_columns_present": [c for c in frame.columns if c in semantic.PROPERTY_IGNORE],
        "all_empty_columns": [c for c in frame.columns if not frame[c].map(nonempty).any()],
        "summary": summaries, "catalog": catalog, "nodes": node_records, "original_ifc": original,
        "semantic_config": {"type_classes": classes, "type_dimension": len(classes),
            "property_dimension": semantic.DEFAULT_PROPERTIES_DIMENSION,
            "embedding_model_default_in_current_code": semantic.DEFAULT_MODEL,
            "max_input_tokens_default": semantic.DEFAULT_MAX_INPUT_TOKENS,
            "truncation_count_at_current_default": sum(n["would_truncate_at_default_limit"] for n in node_records),
            "max_estimated_tokens": max(n["estimated_tokens"] for n in node_records)},
        "embedding_validation": {"dimension_counts": dict(sizes), "one_hot_matches_all_nodes": True,
            "guid_alignment_matches_all_nodes": True, "property_norm_min": min(property_tail_norms),
            "property_norm_max": max(property_tail_norms),
            "limitation": "属性文本按当前代码重建。历史属性向量若曾复用，不能仅由向量反推出当时的完整输入。"},
        "graph_metadata": meta,
        "sources": [str(folder / "target_attribute.csv"), str(folder / "node_index.csv"),
            str(folder / "semantic_embedding.csv"), str(ROOT / "data_processing" / "semantics_Embed.py"),
            str(ROOT / "utils" / "attributes.py"), str(ifc_path) if ifc_path else ""],
    }


def render_report(data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False, default=plain).replace("<", "\\u003c")
    template = Path(__file__).with_name("single_model_semantic_report.html")
    return template.read_text(encoding="utf-8").replace("__AUDIT_DATA__", payload)


def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT, help="External Stage1 dataset directory")
    parser.add_argument("--model", default="model_0")
    parser.add_argument("--ifc", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    ROOT = args.project_root.resolve()
    ifc_path = args.ifc
    if ifc_path is None:
        log = ROOT / "data" / "raw" / args.model / "ifc_parser.log"
        if log.exists():
            match = re.search(r"Starting IFC parsing: (.+)", log.read_text(encoding="utf-8"))
            if match:
                ifc_path = Path(match.group(1).strip())
    out = args.output_dir or Path(__file__).parent / "single_model_results" / args.model
    out.mkdir(parents=True, exist_ok=True)
    data = analyze(args.model, ifc_path)
    # Checks are exact and based on matched records, not the old multi-model report.
    assert sum(s["object_count"] for s in data["summary"]) == data["total_nodes"]
    assert all(0 <= r["non_empty_count"] <= r["object_count"] for r in data["catalog"])
    assert data["embedding_validation"]["dimension_counts"] == {1007: data["total_nodes"]}
    paths = [ROOT / "data" / "processed" / args.model / f for f in ("target_attribute.csv", "node_index.csv", "semantic_embedding.csv")]
    data["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (out / "semantic_audit.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=plain), encoding="utf-8")
    (out / "semantic_audit.html").write_text(render_report(data), encoding="utf-8")
    print(json.dumps({"output": str(out), "nodes": data["total_nodes"],
        "summary": [{k: r[k] for k in ("component_type", "object_count", "semantic_distinct_leaf_paths", "per_object_leaf_min", "per_object_leaf_median", "per_object_leaf_max", "constant_leaf_paths", "property_combinations_excluding_name")} for r in data["summary"]],
        "active_columns": data["active_semantic_top_level_columns"],
        "unique_paths": data["semantic_unique_leaf_path_count"],
        "ifc_type_pset_loss_paths": len(data["original_ifc"].get("type_pset_losses", [])),
        "quantity_sets": data["original_ifc"].get("quantity_set_count")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
