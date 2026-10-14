"""Build semantic embeddings aligned to graph node indexes.

Input per model:
    data/processed/<model>/target_attribute.csv
    data/processed/<model>/node_index.csv

Output per model:
    data/processed/<model>/semantic_embedding.csv

Each row follows target_attribute.csv / node_index.csv order:
    node_id,guid,semantic_embedding

The semantic embedding is:
    node_type one-hot + property text embedding
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"
TARGET_ATTRIBUTE_FILENAME = "target_attribute.csv"
NODE_INDEX_FILENAME = "node_index.csv"
SEMANTIC_EMBEDDING_FILENAME = "semantic_embedding.csv"

DEFAULT_MODEL = "text-embedding-3-small"
DEFAULT_PROPERTIES_DIMENSION = 1000
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_BATCH_TOKENS = 250_000
DEFAULT_MAX_INPUT_TOKENS = 8_000
DEFAULT_NODE_TYPE_CLASSES = [
    "IfcBeam",
    "IfcColumn",
    "IfcDoor",
    "IfcSlab",
    "IfcSpace",
    "IfcWall",
    "IfcWindow",
]

PROPERTY_IGNORE = [
    "node_id",
    "GlobalId",
    "id",
    "type",
    "has_geometry",
    "OwnerHistory",
    "ObjectPlacement",
    "Representation",
    "RepresentationContexts",
    "UnitsInContext",
    "BuildingAddress",
    "SiteAddress",
    "Tag",
    "Creation Date",
    "Last Modified Date",
    "Owning User",
    "User organization",
    "Owning Application",
    "Change Type",
]


def normalize_l2(x: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        norm = np.linalg.norm(arr)
        return arr if norm == 0 else arr / norm

    norm = np.linalg.norm(arr, 2, axis=1, keepdims=True)
    return np.where(norm == 0, arr, arr / norm)


def _is_empty_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return stripped in {"", "none", "None", "null", "NULL", "{}"}
    return False


def _parse_property_value(value: object) -> object:
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def drop_ignored_property_columns(df: pd.DataFrame, ignore: Sequence[str] = PROPERTY_IGNORE) -> pd.DataFrame:
    """Remove non-semantic / bookkeeping columns before property embedding."""
    return df.drop(columns=[column for column in ignore if column in df.columns], errors="ignore")


def combine_properties(row: pd.Series, ignore: Sequence[str] = ()) -> str:
    """Preserve the original row-to-properties logic after ignored columns are removed."""
    combined = {}
    for key, value in row.to_dict().items():
        if key in ignore or _is_empty_value(value):
            continue
        combined[key] = _parse_property_value(value)

    return json.dumps(combined, ensure_ascii=False, sort_keys=True)


def _clean_text(text: object) -> str:
    cleaned = "" if text is None else str(text)
    return cleaned.replace("\n", " ").strip()


def parse_node_type_classes(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return list(DEFAULT_NODE_TYPE_CLASSES)

    classes = [_clean_text(item) for item in value.split(",")]
    classes = [item for item in classes if item]
    if not classes:
        raise ValueError("--node-type-classes must contain at least one non-empty class name.")
    if len(classes) != len(set(classes)):
        raise ValueError(f"--node-type-classes contains duplicates: {classes}")
    return classes


def load_node_type_series(processed_model_dir: Path, target_df: pd.DataFrame) -> pd.Series:
    node_index_csv = processed_model_dir / NODE_INDEX_FILENAME
    if not node_index_csv.exists():
        raise FileNotFoundError(f"Missing node index CSV: {node_index_csv}")

    node_index_df = pd.read_csv(node_index_csv)
    required_columns = {"node_id", "guid", "node_type"}
    missing_columns = required_columns - set(node_index_df.columns)
    if missing_columns:
        raise ValueError(
            f"{node_index_csv} is missing required columns: {sorted(missing_columns)}"
        )

    node_type_by_id = {
        int(row["node_id"]): _clean_text(row["node_type"])
        for _, row in node_index_df.iterrows()
    }
    guid_by_id = {
        int(row["node_id"]): _clean_text(row["guid"])
        for _, row in node_index_df.iterrows()
    }

    node_types: list[str] = []
    for _, row in target_df.iterrows():
        node_id = int(row["node_id"])
        if node_id not in node_type_by_id:
            raise ValueError(f"{processed_model_dir.name}: node_id {node_id} is missing from node_index.csv")

        target_guid = _clean_text(row["GlobalId"])
        index_guid = guid_by_id[node_id]
        if target_guid != index_guid:
            raise ValueError(
                f"{processed_model_dir.name}: node_id {node_id} GUID mismatch between "
                f"target_attribute.csv ({target_guid!r}) and node_index.csv ({index_guid!r})"
            )

        node_types.append(node_type_by_id[node_id])

    return pd.Series(node_types, index=target_df.index, name="node_type")


def build_node_type_one_hot(
    node_types: Sequence[str],
    node_type_classes: Sequence[str],
    *,
    model_name: str,
) -> list[list[float]]:
    class_to_index = {node_type: idx for idx, node_type in enumerate(node_type_classes)}
    unknown = sorted({node_type for node_type in node_types if node_type not in class_to_index})
    if unknown:
        raise ValueError(
            f"{model_name}: node_index.csv contains node_type values not in node type classes: {unknown}. "
            "Pass --node-type-classes with a comma-separated class list if these should be included."
        )

    embeddings: list[list[float]] = []
    for node_type in node_types:
        vector = [0.0] * len(node_type_classes)
        vector[class_to_index[node_type]] = 1.0
        embeddings.append(vector)
    return embeddings


def read_existing_property_embeddings(
    semantic_csv: Path,
    *,
    old_type_dimension: int,
    properties_dimension: int,
) -> list[list[float]]:
    if not semantic_csv.exists():
        raise FileNotFoundError(f"Missing existing semantic embedding CSV: {semantic_csv}")

    df = pd.read_csv(semantic_csv)
    required_columns = {"semantic_embedding"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{semantic_csv} is missing required columns: {sorted(missing_columns)}")

    property_embeddings: list[list[float]] = []
    for row_index, value in enumerate(df["semantic_embedding"].tolist()):
        embedding = json.loads(value)
        if len(embedding) < old_type_dimension + properties_dimension:
            raise ValueError(
                f"{semantic_csv} row {row_index} has embedding dim {len(embedding)}, "
                f"expected at least {old_type_dimension + properties_dimension}"
            )
        property_embeddings.append([float(x) for x in embedding[old_type_dimension:old_type_dimension + properties_dimension]])

    return property_embeddings


def estimate_tokens(text: str) -> int:
    """Cheap conservative-ish estimate to keep embedding batches below API limits."""
    return max(1, math.ceil(len(text) / 4))


def truncate_by_estimated_tokens(text: str, max_tokens: int) -> str:
    if estimate_tokens(text) <= max_tokens:
        return text
    return text[: max_tokens * 4]


def create_embeddings_batch(
    client,
    texts: Sequence[str],
    *,
    model: str,
    dimension: int,
    max_input_tokens: int,
) -> list[list[float]]:
    if not texts:
        return []

    prepared_texts = [
        truncate_by_estimated_tokens(_clean_text(text), max_input_tokens)
        for text in texts
    ]
    response = client.embeddings.create(
        model=model,
        input=prepared_texts,
        encoding_format="float",
    )
    vectors = [item.embedding[:dimension] for item in response.data]
    normalized = normalize_l2(np.asarray(vectors, dtype=np.float64))
    return normalized.tolist()


def get_cached_embeddings(
    client,
    texts: Sequence[str],
    cache: dict[str, list[float]],
    *,
    model: str,
    dimension: int,
    batch_size: int,
    max_batch_tokens: int,
    max_input_tokens: int,
) -> list[list[float]]:
    missing_texts = []
    seen_missing = set()

    for text in texts:
        if text not in cache and text not in seen_missing:
            missing_texts.append(text)
            seen_missing.add(text)

    batch: list[str] = []
    batch_tokens = 0

    def flush_batch() -> None:
        nonlocal batch, batch_tokens
        if not batch:
            return
        embeddings = create_embeddings_batch(
            client,
            batch,
            model=model,
            dimension=dimension,
            max_input_tokens=max_input_tokens,
        )
        for text, embedding in zip(batch, embeddings):
            cache[text] = embedding
        batch = []
        batch_tokens = 0

    for text in missing_texts:
        prepared = truncate_by_estimated_tokens(_clean_text(text), max_input_tokens)
        text_tokens = estimate_tokens(prepared)

        if batch and (len(batch) >= batch_size or batch_tokens + text_tokens > max_batch_tokens):
            flush_batch()

        batch.append(text)
        batch_tokens += text_tokens

    flush_batch()

    return [cache[text] for text in texts]


def build_semantic_embeddings_for_model(
    processed_model_dir: Path,
    *,
    client,
    model: str = DEFAULT_MODEL,
    node_type_classes: Sequence[str] = DEFAULT_NODE_TYPE_CLASSES,
    properties_dimension: int = DEFAULT_PROPERTIES_DIMENSION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    properties_cache: dict[str, list[float]] | None = None,
    reuse_existing_properties: bool = False,
    old_type_dimension: int = 80,
) -> Path:
    target_attribute_csv = processed_model_dir / TARGET_ATTRIBUTE_FILENAME
    if not target_attribute_csv.exists():
        raise FileNotFoundError(f"Missing target attribute CSV: {target_attribute_csv}")

    df = pd.read_csv(target_attribute_csv)
    required_columns = {"node_id", "GlobalId", "type"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{target_attribute_csv} is missing required columns: {sorted(missing_columns)}"
        )

    properties_cache = properties_cache if properties_cache is not None else {}

    node_types = load_node_type_series(processed_model_dir, df).tolist()
    type_embeddings = build_node_type_one_hot(
        node_types,
        node_type_classes,
        model_name=processed_model_dir.name,
    )

    output_path = processed_model_dir / SEMANTIC_EMBEDDING_FILENAME
    if reuse_existing_properties:
        properties_embeddings = read_existing_property_embeddings(
            output_path,
            old_type_dimension=old_type_dimension,
            properties_dimension=properties_dimension,
        )
    else:
        if client is None:
            raise ValueError("client is required when reuse_existing_properties=False")
        property_df = drop_ignored_property_columns(df)
        properties = [combine_properties(row) for _, row in property_df.iterrows()]
        properties_embeddings = get_cached_embeddings(
            client,
            properties,
            properties_cache,
            model=model,
            dimension=properties_dimension,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            max_input_tokens=max_input_tokens,
        )

    semantic_embeddings = [
        type_embedding + properties_embedding
        for type_embedding, properties_embedding in zip(type_embeddings, properties_embeddings)
    ]

    output_df = pd.DataFrame(
        {
            "node_id": df["node_id"].astype(int),
            "guid": df["GlobalId"].astype(str),
            "semantic_embedding": [json.dumps(value) for value in semantic_embeddings],
        }
    )

    output_df.to_csv(output_path, index=False)
    return output_path


def iter_model_dirs(processed_data_root: Path = PROCESSED_DATA_ROOT) -> Iterable[Path]:
    for child in sorted(processed_data_root.iterdir()):
        if child.is_dir():
            yield child


def build_semantic_embeddings_for_all_models(
    *,
    processed_data_root: Path = PROCESSED_DATA_ROOT,
    model_name: str | None = None,
    limit: int | None = None,
    embedding_model: str = DEFAULT_MODEL,
    node_type_classes: Sequence[str] = DEFAULT_NODE_TYPE_CLASSES,
    properties_dimension: int = DEFAULT_PROPERTIES_DIMENSION,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    start_from: str | None = None,
    skip_existing: bool = False,
    reuse_existing_properties: bool = False,
    old_type_dimension: int = 80,
) -> list[Path]:
    client = None
    if not reuse_existing_properties:
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The OpenAI Python package is required for semantic embeddings. "
                "Install it in the active environment with: pip install openai"
            ) from exc
        client = OpenAI()
    properties_cache: dict[str, list[float]] = {}

    model_dirs = list(iter_model_dirs(processed_data_root))
    if model_name is not None:
        model_dirs = [model_dir for model_dir in model_dirs if model_dir.name == model_name]
        if not model_dirs:
            raise FileNotFoundError(f"Missing processed model directory: {processed_data_root / model_name}")
    elif start_from is not None:
        start_index = next(
            (idx for idx, model_dir in enumerate(model_dirs) if model_dir.name == start_from),
            None,
        )
        if start_index is None:
            raise FileNotFoundError(f"Missing processed model directory: {processed_data_root / start_from}")
        model_dirs = model_dirs[start_index:]
    if limit is not None:
        model_dirs = model_dirs[:limit]

    output_paths: list[Path] = []
    for index, processed_model_dir in enumerate(model_dirs, start=1):
        output_path = processed_model_dir / SEMANTIC_EMBEDDING_FILENAME
        if skip_existing and output_path.exists():
            print(f"[{index}/{len(model_dirs)}] Skipping {processed_model_dir.name} (exists)")
            output_paths.append(output_path)
            continue

        started = time.perf_counter()
        print(f"[{index}/{len(model_dirs)}] Processing {processed_model_dir.name}")
        output_path = build_semantic_embeddings_for_model(
            processed_model_dir,
            client=client,
            model=embedding_model,
            node_type_classes=node_type_classes,
            properties_dimension=properties_dimension,
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            max_input_tokens=max_input_tokens,
            properties_cache=properties_cache,
            reuse_existing_properties=reuse_existing_properties,
            old_type_dimension=old_type_dimension,
        )
        elapsed = time.perf_counter() - started
        print(f"[OK] {output_path} ({elapsed:.1f}s)")
        output_paths.append(output_path)

    return output_paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Build node-aligned semantic embeddings.")
    parser.add_argument("--processed-root", type=Path, default=PROCESSED_DATA_ROOT)
    parser.add_argument("--model", dest="model_name", default=None, help="Only process one model folder.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N models.")
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--type-dimension",
        type=int,
        default=None,
        help="Deprecated compatibility argument. Node type is now encoded as one-hot.",
    )
    parser.add_argument(
        "--node-type-classes",
        type=str,
        default=",".join(DEFAULT_NODE_TYPE_CLASSES),
        help="Comma-separated node_type class order for one-hot encoding.",
    )
    parser.add_argument("--properties-dimension", type=int, default=DEFAULT_PROPERTIES_DIMENSION)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-batch-tokens", type=int, default=DEFAULT_MAX_BATCH_TOKENS)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--start-from", default=None, help="Start from this model folder name.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip models that already have semantic_embedding.csv.")
    parser.add_argument(
        "--reuse-existing-properties",
        action="store_true",
        help="Reuse the property embedding tail from existing semantic_embedding.csv and replace only the old type part with node_type one-hot.",
    )
    parser.add_argument(
        "--old-type-dimension",
        type=int,
        default=80,
        help="Number of leading old type-embedding dimensions to drop when --reuse-existing-properties is used.",
    )
    args = parser.parse_args()

    build_semantic_embeddings_for_all_models(
        processed_data_root=args.processed_root,
        model_name=args.model_name,
        limit=args.limit,
        embedding_model=args.embedding_model,
        node_type_classes=parse_node_type_classes(args.node_type_classes),
        properties_dimension=args.properties_dimension,
        batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        max_input_tokens=args.max_input_tokens,
        start_from=args.start_from,
        skip_existing=args.skip_existing,
        reuse_existing_properties=args.reuse_existing_properties,
        old_type_dimension=args.old_type_dimension,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
