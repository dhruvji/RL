#!/usr/bin/env python3
"""Verify SFT assistant supervision spans (check #2: template splitting / non-empty labels).

Mirrors the NeMo RL SFT data path:
  sft_processor -> add_loss_mask_to_message_log(assistant) ->
  batched_message_log_to_flat_message

Reports how many assistant spans are empty, EOS-only, or truncated (loss_multiplier=0),
and shows a few decoded examples.

Example:
  uv run scripts/check_sft_assistant_spans.py data/smoltalk_sft/train.jsonl \\
    --tokenizer-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \\
    --chat-template default \\
    --max-seq-length 8192 \\
    --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.data.processors import sft_processor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsonl", type=Path, help="OpenAI-format JSONL (messages + optional task_name)")
    p.add_argument(
        "--tokenizer-name",
        required=True,
        help="HF tokenizer id (e.g. Nemotron BF16 repo if Base has no chat_template)",
    )
    p.add_argument(
        "--chat-template",
        default="default",
        help='Tokenizer chat_template mode: "default", "default" keeps HF template; '
        'use "none" for passthrough (NULL); or a .jinja path / raw jinja string.',
    )
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--limit", type=int, default=500, help="Max rows to scan (-1 = all)")
    p.add_argument("--add-bos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--add-eos", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--add-generation-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Must match training data_config.add_generation_prompt",
    )
    p.add_argument(
        "--make-sequence-length-divisible-by",
        type=int,
        default=1,
        help="Match policy.make_sequence_length_divisible_by (often tensor_parallel_size)",
    )
    p.add_argument("--verbose", type=int, default=3, help="Print this many full per-row reports")
    return p.parse_args()


def tokenizer_config_from_args(args: argparse.Namespace) -> dict:
    cfg: dict = {"name": args.tokenizer_name}
    ct = args.chat_template
    if ct.lower() == "none" or ct.lower() == "null":
        cfg["chat_template"] = None
    elif ct.lower() == "default":
        cfg["chat_template"] = "default"
    elif ct.endswith(".jinja"):
        cfg["chat_template"] = ct
    else:
        cfg["chat_template"] = ct
    cfg["chat_template_kwargs"] = None
    return cfg


def assistant_stats(
    message_log: list,
    tokenizer,
    loss_multiplier: float,
    flat_mask_sum: int,
) -> dict:
    """Per-row stats focused on assistant turns."""
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    assistants = [m for m in message_log if m.get("role") == "assistant"]
    last = assistants[-1] if assistants else None
    out: dict = {
        "n_assistant_turns": len(assistants),
        "last_assistant_len": 0,
        "last_assistant_non_eos_len": 0,
        "last_assistant_only_eos": False,
        "any_empty_assistant": False,
        "loss_multiplier": loss_multiplier,
        "flat_masked_tokens": flat_mask_sum,
    }
    if not last:
        out["any_empty_assistant"] = True
        return out
    ids = last["token_ids"]
    if hasattr(ids, "numel"):
        n = int(ids.numel())
        id_list = ids.tolist()
    else:
        id_list = list(ids)
        n = len(id_list)
    out["last_assistant_len"] = n
    if n == 0:
        out["any_empty_assistant"] = True
        return out
    non_eos_pad = [t for t in id_list if t not in (eos_id, pad_id) and t is not None]
    out["last_assistant_non_eos_len"] = len(non_eos_pad)
    # Single token and it is EOS -> model only learns to emit EOS from this span
    if n == 1 and id_list[0] == eos_id:
        out["last_assistant_only_eos"] = True
    elif n > 0 and all(t == eos_id for t in id_list):
        out["last_assistant_only_eos"] = True
    for m in assistants:
        tid = m["token_ids"]
        ln = int(tid.numel()) if hasattr(tid, "numel") else len(tid)
        if ln == 0:
            out["any_empty_assistant"] = True
    return out


def decode_preview(tokenizer, ids, max_chars: int = 240) -> str:
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not ids:
        return ""
    text = tokenizer.decode(ids, skip_special_tokens=False)
    if len(text) > max_chars:
        return text[:max_chars] + "…"
    return text


def main() -> None:
    args = parse_args()
    tok_cfg = tokenizer_config_from_args(args)
    tokenizer = get_tokenizer(tok_cfg)

    path = args.jsonl
    if not path.is_file():
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if args.limit >= 0 and len(rows) >= args.limit:
                break

    if not rows:
        print("error: no JSON rows loaded", file=sys.stderr)
        sys.exit(1)

    # Aggregate
    hist_last_len: list[int] = []
    hist_mask_sum: list[int] = []
    n_truncated = 0
    n_empty_last = 0
    n_only_eos = 0
    n_any_empty_assistant = 0
    n_no_assistant = 0

    print(f"Loaded {len(rows)} rows from {path}")
    print(f"Tokenizer: {args.tokenizer_name!r} chat_template={args.chat_template!r}")
    print(
        f"max_seq_length={args.max_seq_length} add_bos={args.add_bos} "
        f"add_eos={args.add_eos} add_generation_prompt={args.add_generation_prompt}"
    )
    print()

    for idx, row in enumerate(rows):
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            n_no_assistant += 1
            continue
        task_name = row.get("task_name") or "openai_format"
        spec = TaskDataSpec(task_name=str(task_name))
        datum = {"messages": messages, "task_name": spec.task_name}

        processed = sft_processor(
            datum,
            spec,
            tokenizer,
            args.max_seq_length,
            idx,
            add_bos=args.add_bos,
            add_eos=args.add_eos,
            add_generation_prompt=args.add_generation_prompt,
        )
        message_log = processed["message_log"]
        lm = float(processed["loss_multiplier"])

        if lm == 0.0:
            n_truncated += 1

        # Same as sft_train
        batch_ml = [message_log]
        add_loss_mask_to_message_log(batch_ml, roles_to_train_on=["assistant"])
        flat, lengths = batched_message_log_to_flat_message(
            batch_ml,
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
            make_sequence_length_divisible_by=args.make_sequence_length_divisible_by,
        )
        mask = flat["token_loss_mask"]
        flat_mask_sum = int(mask.sum().item())

        st = assistant_stats(message_log, tokenizer, lm, flat_mask_sum)
        hist_last_len.append(st["last_assistant_len"])
        hist_mask_sum.append(flat_mask_sum)

        if st["n_assistant_turns"] == 0:
            n_no_assistant += 1
        if st["last_assistant_len"] == 0:
            n_empty_last += 1
        if st["last_assistant_only_eos"]:
            n_only_eos += 1
        if st["any_empty_assistant"]:
            n_any_empty_assistant += 1

        if idx < args.verbose:
            print("=" * 80)
            print(f"Row {idx} task_name={task_name!r} loss_multiplier={lm}")
            print(f"  assistant_turns={st['n_assistant_turns']} last_assistant_len={st['last_assistant_len']} "
                  f"last_non_eos_pad_tokens={st['last_assistant_non_eos_len']} only_eos={st['last_assistant_only_eos']}")
            print(f"  flat supervised token count (sum token_loss_mask)={flat_mask_sum} seq_len={int(lengths[0])}")
            assistants = [m for m in message_log if m.get("role") == "assistant"]
            if assistants:
                last_ids = assistants[-1]["token_ids"]
                print(f"  last assistant decode preview:\n    {decode_preview(tokenizer, last_ids)!r}")
            print()

    n = len(rows)
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"rows_scanned: {n}")
    print(f"rows_with_loss_multiplier_0_truncated: {n_truncated} ({100.0 * n_truncated / n:.1f}%)")
    print(f"rows_no_messages_or_empty: {n_no_assistant} ({100.0 * n_no_assistant / n:.1f}%)")
    print(f"rows_last_assistant_len_0: {n_empty_last} ({100.0 * n_empty_last / n:.1f}%)")
    print(f"rows_last_assistant_only_eos: {n_only_eos} ({100.0 * n_only_eos / n:.1f}%)")
    print(f"rows_any_empty_assistant_turn: {n_any_empty_assistant} ({100.0 * n_any_empty_assistant / n:.1f}%)")

    if hist_last_len:
        s = sorted(hist_last_len)
        def pct(p: float) -> int:
            return s[int(p * (len(s) - 1))]

        print()
        print("last_assistant token count (per row): min / p50 / p90 / max = "
              f"{s[0]} / {pct(0.5)} / {pct(0.9)} / {s[-1]}")
    if hist_mask_sum:
        s2 = sorted(hist_mask_sum)
        print("flat supervised tokens (sum mask): min / p50 / p90 / max = "
              f"{s2[0]} / {s2[int(0.5 * (len(s2) - 1))]} / {s2[int(0.9 * (len(s2) - 1))]} / {s2[-1]}")

    print()
    if n_only_eos > 0 or n_empty_last > 0:
        print(
            "WARNING: Some rows have empty or EOS-only assistant spans — SFT may teach immediate EOS.",
            file=sys.stderr,
        )
    if n_truncated * 2 > n:
        print(
            "WARNING: Many rows hit max_seq_length truncation (loss_multiplier=0) — check max_total_sequence_length.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
