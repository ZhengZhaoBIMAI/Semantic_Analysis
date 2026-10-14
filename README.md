# Stage1 Semantic Analysis

对 Stage1 实验中构件属性的分析结果，覆盖 **215 个模型、93,359 个目标构件**，并逐一对照原始 IFC。分析日期为 **2026-10-14**。

## 查看结果

下载或克隆仓库后，用浏览器打开下面的 HTML。报告内嵌图表和数据，可离线筛选，并导出 CSV。

- [跨模型分析报告](experiments/component_attribute_analysis/dataset_results/dataset_semantic_report.html)：各类构件属性数量、内容类型、字段覆盖率、215 个模型对比、原始 IFC 遗漏、12 个代表模型的构件实例。
- [model_0 案例报告](experiments/component_attribute_analysis/single_model_results/model_0/semantic_audit.html)：单个模型的详细检查和 semantic 输入示例。
- [属性数量与材料对照图](experiments/component_attribute_analysis/dataset_results/semantic_richness_distribution.png)。

GitHub 的文件页面会展示 HTML 源码；在本地浏览器中打开下载的文件即可查看交互报告。

## 本次主要结果

| 构件 | 数量 | 每构件非空属性数中位数 | 当前 Material 字段有值的比例 |
|---|---:|---:|---:|
| 墙 | 28,313 | 17 | 75.6% |
| 板 | 5,563 | 14 | 56.4% |
| 柱 | 10,765 | 13 | 93.1% |
| 门 | 10,282 | 11 | 7.2% |
| 窗 | 17,554 | 12 | 1.3% |
| 梁 | 12,992 | 18 | 88.4% |
| 空间 | 7,890 | 14 | 通常不适用 |

- 全部构件的非空属性数中位数是 16。各模型的中位数从 2 到 85，差异较大。
- 共出现 4,688 个不同的完整属性名。按构件类别分别计数后，共有 5,392 个“类别与字段”组合，其中 93.6% 的对象覆盖率低于 5%。
- 79.8% 的构件有内容重复的 Reference 字段。
- 213 个模型、46,866 个构件至少缺少一项类型继承属性。原始 IFC 中 12,189 面墙有传热系数，当前对应字段只保留 4,359 面。
- 原始 IFC 中约 93.6% 的窗可找到材料名称，当前 Material 一列只有约 1.3%。其他自定义字段仍可能保留部分材料文字。
- 150 个原始 IFC 文件包含工程量集，31 个包含材料层定义。

## 统计定义与已知限制

- **非空属性数**：展开属性集后的字段数量。0、false 是有效值；NOTDEFINED 等文字也算非空。
- **剩余属性数**：去掉 Name、末级名为 Reference 的字段，以及明确未定义占位值后，再数剩下的字段。它不是独立有效信息量。
- **对象覆盖率**：某类构件中，填写某字段的构件比例。
- **模型覆盖率**：包含某类构件的模型中，至少有一个构件填写该字段的模型比例。
- **工程量覆盖率**：Qto_ 或 BaseQuantities 属性组里至少一项有值的构件比例，不代表全部工程量齐全。
- 内容类型按字段名初步分类，不是 IFC 官方分类。“其他自定义属性”还包含未被规则识别的尺寸、楼层、阶段等信息。
- 当前被归入“制造与管理”的 12 种字段全部来自 model_217，取值是重复字段名的模板式文字，例如 `Manufacturer="Manufacturer"`，不能当作真实厂商或保修信息。
- 属性输入按当前编码逻辑重建；未重新调用 embedding API，不能仅凭已有向量反推历史完整输入。
- 数值保留源单位，未将不同单位下的数值混合计算物理量范围。

报告内包含覆盖率、P10/P90、图表和遗漏对照各列的完整中文说明与真实例子。

## 完整统计与审计记录

文件位于 `experiments/component_attribute_analysis/dataset_results/`：

| 文件 | 内容 |
|---|---|
| `current_semantic_statistics.json` | 全部现有属性的统计、模型信息与输入 CSV 的 SHA-256 |
| `original_ifc_statistics.json` | 215 个原始 IFC 的汇总对照 |
| `source_audits.zip` | 215 份逐模型原始 IFC 审计 JSON，解压后为 `source_audits/model_*.json` |
| `node_diagnostics.jsonl.gz` | 93,359 条逐构件计数和诊断记录，每行一个 JSON |

`ARTIFACT_MANIFEST.json` 记录发布文件的大小和 SHA-256。公开结果中的本机绝对路径替换为 `STAGE1_PROJECT` 和 `IFC_SOURCES` 标识，不影响统计值及输入数据的哈希。

本仓库不包含原始 IFC、原始/处理后模型数据、训练文件、embedding 向量或本机环境文件。

## 复现分析

原始数据需自行保留在外部 Stage1 目录，至少包含：

```text
Stage1_experiment/
  data/processed/model_*/target_attribute.csv
  data/processed/model_*/node_index.csv
  data/raw/model_*/meta.json
  data/raw/model_*/ifc_parser.log
```

单模型完整审计还需要 `semantic_embedding.csv` 和 `data/graphs/<model>/metadata.json`。

本次分析环境为 Python 3.14，依赖版本见 `requirements.txt`。

```powershell
python -m pip install -r requirements.txt

python experiments/component_attribute_analysis/audit_dataset_semantics.py --project-root "C:\path\Stage1_experiment" --ifc-root "C:\path\ifc"
```

`--ifc-root` 支持 `model_0.ifc` 或 `model_0/*.ifc` 的目录布局；不指定时，程序尝试从解析日志和 Stage1 的 `data_extended/ifc` 目录定位源文件。

单模型审计：

```powershell
python experiments/component_attribute_analysis/audit_single_model_semantics.py --project-root "C:\path\Stage1_experiment" --model model_0 --ifc "C:\path\ifc\model_0.ifc"
```

分析只写结果文件，不修改外部模型数据。`data_processing/semantics_Embed.py` 是本次审计使用的属性组合逻辑快照，统计过程不调用 embedding API。重新生成中文图表时，系统需有可用的中文字体。

## 验证

已核对所有目标节点的 GUID、节点索引数量、源属性保留与缺失数量，以及分析前后输入 CSV 的哈希。

报告筛选、CSV 导出、实例展示及标签切换的 JavaScript 逻辑可用 Node.js 验证：

```powershell
node experiments/component_attribute_analysis/dataset_results/check_report_runtime.mjs
```

该检查验证脚本逻辑，不代替浏览器中的可视检查。
