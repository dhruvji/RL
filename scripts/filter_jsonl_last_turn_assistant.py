#!/usr/bin/env python3
"""Keep only JSONL rows where messages[-1]['role'] == 'assistant' (NeMo openai_format)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("input_jsonl", type=Path)
    p.add_argument("output_jsonl", type=Path)
    args = p.parse_args()

    kept = 0
    dropped = 0
    with args.input_jsonl.open(encoding="utf-8") as fin, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                msgs = row.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    dropped += 1
                    continue
                if msgs[-1].get("role") != "assistant":
                    dropped += 1
                    continue
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                kept += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                dropped += 1

    print(f"kept={kept} dropped={dropped} -> {args.output_jsonl}", file=sys.stderr)


if __name__ == "__main__":
    main()
