#!/usr/bin/env python3
"""Build a subsampled OpenAI-format JSONL for NeMo RL SFT.

Default data source is `HuggingFaceTB/smoltalk` (SmolTalk SFT mix; see
https://huggingface.co/datasets/HuggingFaceTB/smoltalk and the paper https://arxiv.org/abs/2502.02737).

Example:
  uv run scripts/prepare_nemotron_sft.py --output-dir data/smoltalk_sft --total-samples 50000

  # Specific subset(s); comma-separated HuggingFace config names:
  uv run scripts/prepare_nemotron_sft.py --subsets smol-magpie-ultra,numina-cot-100k --total-samples 10000
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class SourceSpec:
    name: str
    weight: float
    split: str = "train"
    configs: tuple[str, ...] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a subsampled NeMo RL SFT dataset from HuggingFaceTB/smoltalk (or another "
            "messages-style dataset with the same load_dataset API)."
        )
    )
    parser.add_argument(
        "--dataset",
        default="HuggingFaceTB/smoltalk",
        help="Hugging Face dataset id (default: HuggingFaceTB/smoltalk).",
    )
    parser.add_argument(
        "--subsets",
        default="all",
        help=(
            "Comma-separated HuggingFace config names (Smoltalk subsets), e.g. "
            "'all' or 'smol-magpie-ultra,numina-cot-100k'. See dataset card for names."
        ),
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to stream (default: train).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/smoltalk_sft"),
        help="Output directory for train/val jsonl and metadata.",
    )
    parser.add_argument(
        "--total-samples",
        type=int,
        default=50000,
        help="Total number of examples (train + validation) to sample.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.02,
        help="Validation ratio split from sampled data.",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=25000,
        help="Streaming shuffle buffer size per source.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global RNG seed for deterministic sampling/splitting.",
    )
    return parser.parse_args()


def normalize_weights(sources: list[SourceSpec]) -> list[SourceSpec]:
    weight_sum = sum(source.weight for source in sources)
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be greater than zero.")
    return [
        SourceSpec(
            name=source.name,
            split=source.split,
            weight=source.weight / weight_sum,
            configs=source.configs,
        )
        for source in sources
    ]


def target_counts(total: int, sources: list[SourceSpec]) -> dict[str, int]:
    weighted = [(source.name, source.weight * total) for source in sources]
    counts = {name: int(value) for name, value in weighted}
    remainder = total - sum(counts.values())

    residuals = sorted(
        ((name, value - int(value)) for name, value in weighted),
        key=lambda x: x[1],
        reverse=True,
    )

    for name, _ in residuals[:remainder]:
        counts[name] += 1

    return counts


def normalize_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message.get("role")
    if not role:
        return None

    normalized: dict[str, Any] = {"role": role}

    if "content" in message:
        normalized["content"] = message["content"]
    else:
        normalized["content"] = ""

    if "tool_calls" in message:
        normalized["tool_calls"] = message["tool_calls"]
    if "tool_call_id" in message:
        normalized["tool_call_id"] = message["tool_call_id"]
    if "name" in message:
        normalized["name"] = message["name"]

    return normalized


def extract_messages(example: dict[str, Any]) -> list[dict[str, Any]] | None:
    if "messages" in example and isinstance(example["messages"], list):
        normalized = [
            msg for msg in (normalize_message(m) for m in example["messages"]) if msg is not None
        ]
        return normalized if normalized else None

    if "input" in example and "output" in example:
        return [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]

    return None


def is_valid_dialog(messages: list[dict[str, Any]]) -> bool:
    """Match NeMo RL ``OpenAIFormatDataset`` (last turn must be ``assistant``)."""
    roles = {msg.get("role") for msg in messages}
    if "user" not in roles or "assistant" not in roles:
        return False
    # Tool-calling or multi-turn rows may end with ``tool`` / ``user``; NeMo rejects those.
    return messages[-1].get("role") == "assistant"


def to_record(example: dict[str, Any], dataset_name: str) -> dict[str, Any] | None:
    messages = extract_messages(example)
    if messages is None or not is_valid_dialog(messages):
        return None

    record: dict[str, Any] = {
        "messages": messages,
        "source_dataset": dataset_name,
    }

    if "tools" in example and isinstance(example["tools"], list):
        record["tools"] = example["tools"]
    if "category" in example:
        record["category"] = example["category"]
    if "source" in example:
        record["source"] = example["source"]
    if "thinking" in example:
        record["thinking"] = example["thinking"]

    return record


def stream_subsample_from_single_config(
    dataset_name: str,
    config_name: str | None,
    split: str,
    n_samples: int,
    seed: int,
    shuffle_buffer_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config_name is None:
        dataset = load_dataset(dataset_name, split=split, streaming=True)
    else:
        dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    rows: list[dict[str, Any]] = []
    seen = 0
    kept = 0

    for example in dataset:
        seen += 1
        record = to_record(example, dataset_name)
        if record is None:
            continue
        rows.append(record)
        kept += 1
        if kept >= n_samples:
            break

    stats = {
        "dataset": dataset_name,
        "config": config_name,
        "requested": n_samples,
        "seen": seen,
        "kept": kept,
    }
    return rows, stats


def stream_subsample(
    source: SourceSpec,
    n_samples: int,
    seed: int,
    shuffle_buffer_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if source.configs is None:
        rows, stats = stream_subsample_from_single_config(
            dataset_name=source.name,
            config_name=None,
            split=source.split,
            n_samples=n_samples,
            seed=seed,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        return rows, {"dataset": source.name, "configs": [stats]}

    per_config = n_samples // len(source.configs)
    remainder = n_samples % len(source.configs)

    rows: list[dict[str, Any]] = []
    per_config_stats: list[dict[str, Any]] = []

    for idx, config_name in enumerate(source.configs):
        target = per_config + (1 if idx < remainder else 0)
        sampled, stats = stream_subsample_from_single_config(
            dataset_name=source.name,
            config_name=config_name,
            split=source.split,
            n_samples=target,
            seed=seed + idx,
            shuffle_buffer_size=shuffle_buffer_size,
        )
        rows.extend(sampled)
        per_config_stats.append(stats)

    return rows, {"dataset": source.name, "configs": per_config_stats}


def split_train_val(
    rows: list[dict[str, Any]],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    copied = list(rows)
    rng.shuffle(copied)

    val_count = int(len(copied) * val_ratio)
    if len(copied) > 1 and val_count == 0 and val_ratio > 0:
        val_count = 1

    val_rows = copied[:val_count]
    train_rows = copied[val_count:]
    return train_rows, val_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_nemo_config(path: Path, train_path: Path, val_path: Path) -> None:
    content = f"""# NeMo-RL SFT dataset config snippet (paths are absolute; adjust or use Hydra overrides)
data:
  _override_: true
  max_input_seq_length: ${{policy.max_total_sequence_length}}
  train:
    dataset_name: openai_format
    data_path: {train_path.as_posix()}
    chat_key: messages
    tool_key: tools
    split_validation_size: 0
  validation:
    dataset_name: openai_format
    data_path: {val_path.as_posix()}
    chat_key: messages
    tool_key: tools
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()

    if args.total_samples <= 0:
        raise ValueError("--total-samples must be > 0")
    if not (0 <= args.val_ratio < 1):
        raise ValueError("--val-ratio must be in [0, 1)")

    subset_tuple = tuple(s.strip() for s in args.subsets.split(",") if s.strip())
    if not subset_tuple:
        raise ValueError("--subsets must list at least one config name (e.g. 'all').")

    sources = normalize_weights(
        [
            SourceSpec(
                name=args.dataset,
                weight=1.0,
                configs=subset_tuple,
                split=args.split,
            ),
        ]
    )

    counts = target_counts(args.total_samples, sources)

    source_stats: list[dict[str, Any]] = []
    all_train: list[dict[str, Any]] = []
    all_val: list[dict[str, Any]] = []

    for idx, source in enumerate(sources):
        n_samples = counts[source.name]
        sampled, stats = stream_subsample(
            source=source,
            n_samples=n_samples,
            seed=args.seed + idx,
            shuffle_buffer_size=args.shuffle_buffer_size,
        )
        train_rows, val_rows = split_train_val(
            sampled,
            val_ratio=args.val_ratio,
            seed=args.seed + idx,
        )
        stats.update({"train": len(train_rows), "val": len(val_rows), "requested": n_samples})
        source_stats.append(stats)
        all_train.extend(train_rows)
        all_val.extend(val_rows)

    rng = random.Random(args.seed)
    rng.shuffle(all_train)
    rng.shuffle(all_val)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.openai.jsonl"
    val_path = output_dir / "val.openai.jsonl"
    metadata_path = output_dir / "dataset_metadata.json"
    nemo_cfg_path = output_dir / "nemo_rl_data_config.yaml"

    write_jsonl(train_path, all_train)
    write_jsonl(val_path, all_val)
    write_nemo_config(nemo_cfg_path, train_path, val_path)

    metadata = {
        "seed": args.seed,
        "dataset": args.dataset,
        "subsets": list(subset_tuple),
        "split": args.split,
        "total_samples_requested": args.total_samples,
        "val_ratio": args.val_ratio,
        "sources": source_stats,
        "output_files": {
            "train": train_path.as_posix(),
            "val": val_path.as_posix(),
            "nemo_config": nemo_cfg_path.as_posix(),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Created NeMo-RL SFT dataset:")
    print(f"  train: {train_path} ({len(all_train)} rows)")
    print(f"  val:   {val_path} ({len(all_val)} rows)")
    print(f"  cfg:   {nemo_cfg_path}")
    print(f"  meta:  {metadata_path}")


if __name__ == "__main__":
    main()
