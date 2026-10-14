"""Profile existing semantic inputs and original IFC files across Stage1 models.

Writes new analysis artifacts only. No model data, embeddings, or training files
are modified. Python 3.14 supplies pandas, numpy, and IfcOpenShell locally.
"""
from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data_processing import semantics_Embed as sem
from experiments.component_attribute_analysis import audit_single_model_semantics as one

ORDER, LABELS = one.ORDER, one.LABELS
IFC_ROOT = None
PLACEHOLDERS = {"notdefined", "not defined", "undefined", "unknown", "unspecified", "unset", "not set"}
CATEGORY_ORDER = ["名称与引用", "分类与用途", "尺寸与工程量", "材料与构造", "功能与布尔", "性能与安全", "制造与管理", "其他自定义属性"]


def natural(p):
    return int(p.name.split("_")[-1])


def category(path):
    key = path.rsplit(".", 1)[-1].lower()
    low = path.lower()
    if key in {"name", "longname", "description", "reference", "objecttype", "tag", "guid", "globalid", "id"}:
        return "名称与引用"
    if any(x in low for x in ("thermal", "firerating", "acoustic", "sound", "smoke", "fireresistance", "flammab", "securityrating", "heattransfer", "heattransmitt")):
        return "性能与安全"
    if path.startswith(("Qto_", "BaseQuantities")) or key in {"overallheight", "overallwidth", "length", "height", "width", "depth", "thickness", "span", "slope", "roll", "pitchangle", "area", "volume", "perimeter", "weight"} or any(x in key for x in ("netarea", "grossarea", "netvolume", "grossvolume", "thickness")):
        return "尺寸与工程量"
    if low.startswith("material") or any(x in key for x in ("reinforcement", "constructiontype", "structuralmaterial")):
        return "材料与构造"
    if key in {"predefinedtype", "compositiontype", "operationtype", "partitioningtype", "elementtype", "occupancytype", "spaceusage", "category", "classification"}:
        return "分类与用途"
    if key in {"isexternal", "loadbearing", "extendtostructure", "handicapaccessible", "fireexit", "selfclosing", "isventilated", "isilluminated", "isairconditioned"} or key.startswith("is"):
        return "功能与布尔"
    if any(x in key for x in ("manufacturer", "modelreference", "production", "serialnumber", "cost", "warranty", "purchase", "installationdate")):
        return "制造与管理"
    return "其他自定义属性"


def placeholder(value):
    return isinstance(value, str) and value.strip().lower() in PLACEHOLDERS


def value_type(value):
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, (list, dict)):
        return "collection"
    return "text"


def canon(value):
    return json.dumps(one.plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def quantiles(values):
    if not values:
        return {}
    v = np.asarray(values, dtype=float)
    return {"min": float(v.min()), "p10": float(np.quantile(v, .1)), "median": float(np.median(v)), "p90": float(np.quantile(v, .9)), "max": float(v.max()), "mean": float(v.mean())}


def source_path(name):
    if IFC_ROOT is not None:
        direct = IFC_ROOT / f"{name}.ifc"
        if direct.exists():
            return direct
        nested = list((IFC_ROOT / name).glob("*.ifc"))
        if len(nested) == 1:
            return nested[0]
    log = ROOT / "data/raw" / name / "ifc_parser.log"
    if log.exists():
        m = re.search(r"Starting IFC parsing: (.+)", log.read_text(encoding="utf-8")[:5000])
        if m and Path(m.group(1).strip()).exists():
            return Path(m.group(1).strip())
    candidates = list((ROOT / "data_extended/ifc" / name).glob("*.ifc"))
    return candidates[0] if len(candidates) == 1 else None


def read_model(name):
    folder = ROOT / "data/processed" / name
    frame = pd.read_csv(folder / "target_attribute.csv", low_memory=False)
    idx = pd.read_csv(folder / "node_index.csv")
    assert len(idx) == len(frame) and not frame.node_id.duplicated().any() and not frame.GlobalId.duplicated().any(), name
    index = idx.set_index("node_id")
    assert set(frame.node_id) == set(idx.node_id), name
    assert all(str(g) == str(index.loc[i, "guid"]) for i, g in zip(frame.node_id, frame.GlobalId)), name
    kinds = [index.loc[i, "node_type"] for i in frame.node_id]
    assert set(kinds) <= set(ORDER), (name, set(kinds))
    prop = sem.drop_ignored_property_columns(frame)
    leaf_map, texts = {}, {}
    for node_id, values in zip(frame.node_id, prop.itertuples(index=False, name=None)):
        content = {k: sem._parse_property_value(v) for k, v in zip(prop.columns, values) if not sem._is_empty_value(v)}
        text = json.dumps(content, ensure_ascii=False, sort_keys=True)
        leaf_map[int(node_id)] = dict(one.leaves(content, ""))
        texts[int(node_id)] = text
    for pos in sorted({0, len(prop) // 2, len(prop) - 1}):
        assert texts[int(frame.iloc[pos].node_id)] == sem.combine_properties(prop.iloc[pos]), (name, pos)
    frame["node_type"] = kinds
    return frame, leaf_map, texts


def aggregate_current(model_dirs, out):
    stats = {}
    class_counts, class_models = Counter(), defaultdict(set)
    model_rows, class_model_rows, node_rows, category_hits = [], [], [], Counter()
    global_source_types = Counter()
    for number, folder in enumerate(model_dirs, 1):
        name = folder.name
        frame, leaf_map, texts = read_model(name)
        raw_meta = json.loads((ROOT / "data/raw" / name / "meta.json").read_text(encoding="utf-8")).get("metadata", {})
        application = raw_meta.get("application_name", "unknown")
        vendor = "Archicad" if "archicad" in application.lower() else "Revit" if "revit" in application.lower() else "Other"
        class_sizes = Counter(frame.node_type)
        model_nodes, model_combinations = [], defaultdict(set)
        global_source_types.update(frame["type"])
        for obj in frame[["node_id", "GlobalId", "node_type", "Name"]].to_dict("records"):
            node_id, kind = int(obj["node_id"]), obj["node_type"]
            class_counts[kind] += 1
            class_models[kind].add(name)
            actual = {p: v for p, v in leaf_map[node_id].items() if one.nonempty(v)}
            informative = {p: v for p, v in actual.items() if p != "Name" and not p.endswith(".Reference") and not placeholder(v)}
            ref_values = [canon(v) for p, v in actual.items() if p.endswith(".Reference")]
            duplicate_refs = len(ref_values) - len(set(ref_values))
            cats = set()
            for p, v in actual.items():
                cats.add(category(p))
                key = (kind, p)
                if key not in stats:
                    stats[key] = {"n": 0, "models": set(), "values": Counter(), "types": Counter(), "placeholder": 0, "per_model": Counter()}
                s = stats[key]
                s["n"] += 1
                s["models"].add(name)
                s["values"][canon(v)] += 1
                s["types"][value_type(v)] += 1
                s["placeholder"] += placeholder(v)
                s["per_model"][name] += 1
            for cat in cats:
                category_hits[(kind, cat)] += 1
            combo = canon({p: v for p, v in actual.items() if p != "Name"})
            model_combinations[kind].add(combo)
            tokens = sem.estimate_tokens(sem._clean_text(texts[node_id]))
            row = {"model": name, "node_id": node_id, "component_type": kind,
                   "leaf_count": len(actual), "remaining_count": len(informative),
                   "reference_count": len(ref_values), "duplicate_reference_count": duplicate_refs,
                   "placeholder_count": sum(placeholder(v) for v in actual.values()),
                   "name_ends_instance_number": bool(re.search(r":\d+$", str(obj["Name"]))),
                   "estimated_tokens": tokens, "would_truncate": tokens > sem.DEFAULT_MAX_INPUT_TOKENS,
                   "material_available": one.nonempty(actual.get("Material")),
                   "quantity_available": any(p.startswith(("Qto_", "BaseQuantities")) for p in actual)}
            node_rows.append(row)
            model_nodes.append(row)
        for kind in ORDER:
            nodes = [n for n in model_nodes if n["component_type"] == kind]
            if not nodes:
                continue
            class_model_rows.append({"model": name, "component_type": kind, "object_count": len(nodes),
                "leaf": quantiles([n["leaf_count"] for n in nodes]),
                "remaining": quantiles([n["remaining_count"] for n in nodes]),
                "material_coverage": sum(n["material_available"] for n in nodes)/len(nodes),
                "quantity_coverage": sum(n["quantity_available"] for n in nodes)/len(nodes),
                "combinations_without_name": len(model_combinations[kind]),
                "combination_fraction": len(model_combinations[kind])/len(nodes)})
        model_rows.append({"model": name, "application": application, "vendor": vendor,
            "schema": raw_meta.get("source", "unknown"), "source_ifc": str(source_path(name)),
            "source_ifc_size_mb": raw_meta.get("file_size_mb"), "node_count": len(frame),
            "component_counts": dict(class_sizes), "csv_columns": len(frame.columns) - 1,
            "leaf": quantiles([n["leaf_count"] for n in model_nodes]),
            "remaining": quantiles([n["remaining_count"] for n in model_nodes]),
            "material_coverage": sum(n["material_available"] for n in model_nodes)/len(model_nodes),
            "quantity_coverage": sum(n["quantity_available"] for n in model_nodes)/len(model_nodes),
            "duplicate_reference_nodes": sum(n["duplicate_reference_count"] > 0 for n in model_nodes),
            "name_number_nodes": sum(n["name_ends_instance_number"] for n in model_nodes),
            "placeholder_nodes": sum(n["placeholder_count"] > 0 for n in model_nodes),
            "truncation_nodes": sum(n["would_truncate"] for n in model_nodes),
            "source_csv_sha256": hashlib.sha256((folder / "target_attribute.csv").read_bytes()).hexdigest()})
        if number % 10 == 0 or number == len(model_dirs):
            print(f"Current {number}/{len(model_dirs)}: {name}, cumulative nodes={len(node_rows)}", flush=True)
    class_model_map = {(r["model"], r["component_type"]):r["object_count"] for r in class_model_rows}
    catalog = []
    for (kind, path), s in stats.items():
        model_mean = sum(s["per_model"].get(m, 0) / class_model_map[(m, kind)] for m in class_models[kind]) / len(class_models[kind])
        catalog.append({"component_type": kind, "path": path, "meaning": one.meaning(path),
            "category": category(path), "value_types": dict(s["types"]), "non_empty_count": s["n"],
            "object_count": class_counts[kind], "coverage": s["n"] / class_counts[kind],
            "models_with_attribute": len(s["models"]), "models_with_component": len(class_models[kind]),
            "model_coverage": len(s["models"])/len(class_models[kind]),
            "mean_within_model_coverage": model_mean, "unique_values": len(s["values"]),
            "placeholder_count": s["placeholder"],
            "top_values": [{"value":json.loads(v),"count":n} for v,n in s["values"].most_common(5)]})
    catalog.sort(key=lambda r:(ORDER.index(r["component_type"]),-r["coverage"],r["path"]))
    classes = []
    for kind in ORDER:
        nodes = [n for n in node_rows if n["component_type"]==kind]
        fields = [r for r in catalog if r["component_type"]==kind]
        classes.append({"component_type":kind,"label":LABELS[kind],"node_count":len(nodes),
            "model_count":len(class_models[kind]),"distinct_paths":len(fields),
            "leaf":quantiles([n["leaf_count"] for n in nodes]),
            "remaining":quantiles([n["remaining_count"] for n in nodes]),
            "material_coverage":sum(n["material_available"] for n in nodes)/len(nodes),
            "quantity_coverage":sum(n["quantity_available"] for n in nodes)/len(nodes),
            "duplicate_reference_coverage":sum(n["duplicate_reference_count"]>0 for n in nodes)/len(nodes),
            "placeholder_coverage":sum(n["placeholder_count"]>0 for n in nodes)/len(nodes),
            "name_number_coverage":sum(n["name_ends_instance_number"] for n in nodes)/len(nodes),
            "common_path_count":sum(r["coverage"]>=.5 and r["model_coverage"]>=.5 for r in fields),
            "category_coverage":{c:category_hits[(kind,c)]/len(nodes) for c in CATEGORY_ORDER}})
    result = {"date":"2026-10-14","model_count":len(model_rows),"node_count":len(node_rows),
        "models":model_rows,"classes":classes,"class_models":class_model_rows,"catalog":catalog,
        "category_order":CATEGORY_ORDER,"source_ifc_types":dict(global_source_types),
        "total_unique_paths":len({r["path"] for r in catalog}),"class_path_count":len(catalog),
        "leaf":quantiles([n["leaf_count"] for n in node_rows]),
        "remaining":quantiles([n["remaining_count"] for n in node_rows]),
        "value_type_counts":dict(sum((Counter(r["value_types"]) for r in catalog),Counter())),
        "sparse_object_paths":sum(r["coverage"]<.05 for r in catalog),
        "sparse_model_paths":sum(r["model_coverage"]<.05 for r in catalog),
        "duplicate_reference_nodes":sum(n["duplicate_reference_count"]>0 for n in node_rows),
        "duplicate_reference_occurrences":sum(n["duplicate_reference_count"] for n in node_rows),
        "name_number_nodes":sum(n["name_ends_instance_number"] for n in node_rows),
        "placeholder_nodes":sum(n["placeholder_count"]>0 for n in node_rows),
        "truncation_nodes":sum(n["would_truncate"] for n in node_rows),
        "max_estimated_tokens":max(n["estimated_tokens"] for n in node_rows),
        "method":{"population":"全部可用 processed 模型的图节点，非原始 IFC 所有对象",
           "remaining":"移除 Name、末级名为 Reference 的字段及 NOTDEFINED/UNKNOWN 等未定义占位值；不是独立信息量评分",
           "coverage":"对象覆盖率以该类节点为分母；模型覆盖率以含该类的模型为分母；模型均值覆盖率让每个模型等权",
           "categories":"按属性名规则归类，仅帮助浏览；中文或软件自定义同义字段未做全面语义合并",
           "nulls":"None、空容器、空字符串、NaN、NA/N/A/null 不计非空；0、false 保留",
           "history":"属性输入按当前编码代码重建，未重新生成向量，不能反推历史复用向量的完整输入",
           "units":"数值保留源单位，不跨不同单位合并物理量的最小/最大值",
           "placeholders":sorted(PLACEHOLDERS)}}
    (out/"current_semantic_statistics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    # Per-node diagnostic records are a machine-readable analysis intermediate.
    with (out/"node_diagnostics.jsonl").open("w",encoding="utf-8") as f:
        for row in node_rows:
            f.write(json.dumps(row,ensure_ascii=False)+"\n")
    return result


def select_sources(current, count):
    if count == 0:
        return [m["model"] for m in current["models"]]
    chosen = ["model_0"]
    strata = defaultdict(list)
    for m in current["models"]:
        strata[(m["vendor"],m["schema"])].append(m)
    for rows in strata.values():
        rows.sort(key=lambda r:r["leaf"]["median"])
        for i in sorted({0,len(rows)//2,len(rows)-1}):
            if rows[i]["model"] not in chosen:
                chosen.append(rows[i]["model"])
    return chosen[:count] if len(chosen)>count else chosen


def aggregate_original(current, names, out):
    details = out/"source_audits"
    details.mkdir(exist_ok=True)
    aggregate = {}
    model_rows, class_records, errors = [], [], []
    material_representations = Counter()
    for number,name in enumerate(names,1):
        started = time.perf_counter()
        checkpoint = details/f"{name}.json"
        try:
            audit = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else None
            if audit is None or audit.get("audit_version") != 2:
                frame, props, _ = read_model(name)
                audit = one.audit_ifc(source_path(name),frame,props)
                assert audit["available"] and audit["target_guids_matched"]==len(frame)
                checkpoint.write_text(json.dumps(audit,ensure_ascii=False,default=one.plain,allow_nan=False),encoding="utf-8")
                del frame,props
                gc.collect()
            material_representations.update(audit["material_representation_counts"])
            losses = audit["type_pset_losses"]
            gap_nodes = {r["node_id"] for r in audit["gap_instances"] if r.get("origin") == "类型继承 Pset/Qto"}
            row={"model":name,"matched_nodes":audit["target_guids_matched"],"schema":audit["schema"],
                "type_inheritance_loss_paths":len(losses),"type_inheritance_affected_nodes":len(gap_nodes),
                "quantity_set_count":audit["quantity_set_count"],"material_layer_count":audit["material_layer_count"],
                "material_representation_counts":audit["material_representation_counts"],"units":audit["units"]}
            model_rows.append(row)
            for r in audit["catalog"]:
                key=(r["component_type"],r["path"])
                if key not in aggregate:
                    aggregate[key]={"source_count":0,"current_count":0,"missing_count":0,"different_count":0,
                        "models":set(),"missing_models":set(),"origin":set(),"examples":[],"missing_by_origin":Counter(),"source_by_origin":Counter()}
                s=aggregate[key]
                s["source_count"]+=r["non_empty_count"]
                s["current_count"]+=r["semantic_nonempty_count"]
                s["missing_count"]+=r["missing_from_semantic_count"]
                s["different_count"]+=r["different_value_count"]
                s["models"].add(name)
                if r["missing_from_semantic_count"]:
                    s["missing_models"].add(name)
                s["origin"].update(r["origin"])
                s["missing_by_origin"].update(r["missing_count_by_origin"])
                s["source_by_origin"].update(r["source_count_by_origin"])
                if len(s["examples"])<4:
                    s["examples"].extend({"model":name,"value":v["value"],"count":v["count"]} for v in r["top_values"][:2])
            for kind in ORDER:
                rows=[r for r in audit["catalog"] if r["component_type"]==kind]
                if not rows:
                    continue
                class_records.append({"model":name,"component_type":kind,"node_count":rows[0]["object_count"],
                    "source_material_nodes":sum(r["non_empty_count"] for r in rows if r["path"] in {"Material.Name","Material.ComponentNames"}),
                    "source_material_single_nodes":sum(r["non_empty_count"] for r in rows if r["path"]=="Material.Name"),
                    "source_material_set_nodes":sum(r["non_empty_count"] for r in rows if r["path"]=="Material.ComponentNames"),
                    "source_material_layer_nodes":sum(r["non_empty_count"] for r in rows if r["path"]=="Material.Layers"),
                    "type_loss_field_occurrences":sum(r["missing_count_by_origin"].get("类型继承 Pset/Qto",0) for r in rows)})
        except Exception as exc:
            errors.append({"model":name,"error":repr(exc)})
            print(f"SOURCE ERROR {name}: {exc!r}",flush=True)
        print(f"Source {number}/{len(names)}: {name}, {time.perf_counter()-started:.1f}s",flush=True)
    denominators=Counter()
    for r in class_records:
        denominators[r["component_type"]]+=r["node_count"]
    catalog=[]
    for (kind,path),s in aggregate.items():
        catalog.append({"component_type":kind,"path":path,"meaning":one.meaning(path),"category":category(path),
            "source_nonempty_count":s["source_count"],"current_nonempty_count":s["current_count"],
            "missing_count":s["missing_count"],"different_value_count":s["different_count"],
            "audited_class_nodes":denominators[kind],"source_coverage":s["source_count"]/denominators[kind],
            "source_models":len(s["models"]),"missing_models":len(s["missing_models"]),
            "origin":sorted(s["origin"]),"examples":s["examples"]})
        catalog[-1]["missing_count_by_origin"] = dict(s["missing_by_origin"])
        catalog[-1]["source_count_by_origin"] = dict(s["source_by_origin"])
    catalog.sort(key=lambda r:(-r["missing_count"],r["component_type"],r["path"]))
    result={"requested_models":len(names),"audited_models":len(model_rows),
        "audited_nodes":sum(r["matched_nodes"] for r in model_rows),"models":model_rows,"class_models":class_records,
        "catalog":catalog,"material_representation_counts":dict(material_representations),"errors":errors,
        "type_inheritance_affected_nodes":sum(r["type_inheritance_affected_nodes"] for r in model_rows),
        "type_inheritance_affected_models":sum(r["type_inheritance_affected_nodes"]>0 for r in model_rows),
        "models_with_quantity_sets":sum(r["quantity_set_count"]>0 for r in model_rows),
        "models_with_material_layers":sum(r["material_layer_count"]>0 for r in model_rows)}
    (out/"original_ifc_statistics.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    return result


def report(current, original, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_path=Path("C:/Windows/Fonts/msyh.ttc")
    family=font_manager.FontProperties(fname=str(font_path)).get_name() if font_path.exists() else "sans-serif"
    plt.rcParams.update({"font.family":family,"font.sans-serif":["Microsoft YaHei","Noto Sans CJK SC","DejaVu Sans"],"axes.unicode_minus":False,"font.size":10})
    fig,(left,right)=plt.subplots(1,2,figsize=(15,5.5),gridspec_kw={"width_ratios":[1.1,1]})
    y=np.arange(len(current["classes"]))
    for i,r in enumerate(current["classes"]):
        left.plot([r["leaf"]["p10"],r["leaf"]["p90"]],[i,i],color="#b7d5db",linewidth=9,solid_capstyle="round")
        left.scatter(r["leaf"]["median"],i,s=52,color="#087b82",zorder=4)
        left.scatter(r["remaining"]["median"],i,s=40,facecolors="white",edgecolors="#235979",zorder=5)
        left.text(r["leaf"]["median"]+.8,i-.13,str(int(r["leaf"]["median"])),va="bottom",fontsize=9,color="#365367")
    left.set_yticks(y,[r["label"] for r in current["classes"]])
    left.invert_yaxis()
    left.set_ylim(len(y)-.4,-.8)
    left.set_xlabel("每个构件的属性字段数量")
    left.set_title("属性数量：P10–P90 区间与中位数",loc="left",fontweight="bold")
    left.plot([],[],color="#b7d5db",linewidth=7,label="P10–P90")
    left.scatter([],[],color="#087b82",label="非空字段中位数")
    left.scatter([],[],facecolors="white",edgecolors="#235979",label="去 Name/Reference 和占位值后的中位数")
    left.legend(loc="upper left",bbox_to_anchor=(0,-.19),fontsize=8,frameon=False)
    source_material=[]
    for c in current["classes"]:
        rows=[r for r in original["class_models"] if r["component_type"]==c["component_type"]]
        source_material.append(sum(r["source_material_nodes"] for r in rows)/sum(r["node_count"] for r in rows))
    right.barh(y-.17,[r["material_coverage"]*100 for r in current["classes"]],height=.32,color="#087b82",label="当前 Material 字段")
    right.barh(y+.17,np.asarray(source_material)*100,height=.32,color="#a7c8d6",label="源 IFC 有材料名称（含集合内部）")
    right.set_yticks(y,[r["label"] for r in current["classes"]])
    right.invert_yaxis()
    right.set_xlim(0,104)
    right.set_xlabel("该类节点中的覆盖率（%）")
    right.set_title("材料名称：当前输入与源 IFC",loc="left",fontweight="bold")
    right.text(3,len(y)-1,"空间材料通常不适用",va="center",fontsize=9,color="#637c87")
    right.legend(loc="upper left",bbox_to_anchor=(0,-.19),fontsize=8,frameon=False)
    for ax in (left,right):
        ax.grid(axis="x",alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[["top","right"]].set_visible(False)
    fig.suptitle(f"Stage1：{current['model_count']} 个模型，{current['node_count']:,} 个图节点",x=.06,ha="left",fontsize=14,fontweight="bold")
    fig.tight_layout(rect=[0,.15,1,.94])
    png=out/"semantic_richness_distribution.png"
    fig.savefig(png,dpi=180,bbox_inches="tight",facecolor="white")
    plt.close(fig)
    # Compact table payload; full statistics and per-model audits remain separate.
    examples=[]
    for name in select_sources(current,16):
        frame, props, _ = read_model(name)
        for kind in ORDER:
            rows=frame[frame.node_type==kind]
            if rows.empty:
                continue
            sizes={int(r.node_id):sum(one.nonempty(v) for v in props[int(r.node_id)].values()) for r in rows.itertuples()}
            median=float(np.median(list(sizes.values())))
            row=min(rows.itertuples(),key=lambda r:abs(sizes[int(r.node_id)]-median))
            examples.append({"model":name,"component_type":kind,"node_id":int(row.node_id),"guid":row.GlobalId,
                "leaf_count":sizes[int(row.node_id)],"properties":{p:v for p,v in props[int(row.node_id)].items() if one.nonempty(v)}})
    payload={"current":current,"original":original,"examples":examples}
    for r in payload["current"]["catalog"]:
        for v in r["top_values"]:
            text=canon(v["value"])
            if len(text)>220:
                v["value"]=text[:217]+"..."
    text=json.dumps(payload,ensure_ascii=False,allow_nan=False).replace("<","\\u003c")
    template=Path(__file__).with_name("dataset_semantic_report.html").read_text(encoding="utf-8")
    template=template.replace("__READING_GUIDE__",Path(__file__).with_name("dataset_report_glossary.html").read_text(encoding="utf-8"))
    template=template.replace("__CATEGORY_GUIDE__",Path(__file__).with_name("dataset_semantic_categories.html").read_text(encoding="utf-8"))
    template=template.replace('src="semantic_richness_distribution.png"','src="data:image/png;base64,'+base64.b64encode(png.read_bytes()).decode("ascii")+'"')
    (out/"dataset_semantic_report.html").write_text(template.replace("__DATA__",text),encoding="utf-8")


def main():
    global ROOT, IFC_ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root",type=Path,default=ROOT,help="External Stage1 directory containing data/processed and data/raw")
    parser.add_argument("--ifc-root",type=Path,help="Optional original IFC directory: model_0.ifc or model_0/*.ifc")
    parser.add_argument("--output-dir",type=Path,default=Path(__file__).parent/"dataset_results")
    parser.add_argument("--source-count",type=int,default=0,help="0 = all models; positive = stratified case selection")
    parser.add_argument("--reuse-current",action="store_true")
    parser.add_argument("--skip-render",action="store_true")
    args=parser.parse_args()
    ROOT=args.project_root.resolve()
    one.ROOT=ROOT
    IFC_ROOT=args.ifc_root.resolve() if args.ifc_root else None
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    dirs=sorted([p for p in (ROOT/"data/processed").glob("model_*") if p.is_dir() and (p/"target_attribute.csv").exists()],key=natural)
    if args.reuse_current:
        current=json.loads((out/"current_semantic_statistics.json").read_text(encoding="utf-8"))
    else:
        current=aggregate_current(dirs,out)
    original=aggregate_original(current,select_sources(current,args.source_count),out)
    assert sum(r["node_count"] for r in current["classes"])==current["node_count"]
    assert all(r["non_empty_count"]<=r["object_count"] for r in current["catalog"])
    if not args.skip_render:
        report(current,original,out)
    print(json.dumps({"models":current["model_count"],"nodes":current["node_count"],
        "paths":current["total_unique_paths"],"class_paths":current["class_path_count"],
        "leaf":current["leaf"],"remaining":current["remaining"],
        "source_models":original["audited_models"],"source_errors":original["errors"],
        "type_inheritance_affected_nodes":original["type_inheritance_affected_nodes"],
        "type_inheritance_affected_models":original["type_inheritance_affected_models"],
        "output":str(out)},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
