#!/usr/bin/env python3
"""Smoke-test a consolidated HF Nemotron SFT checkpoint with the same tokenizer/chat template as training.

Reads ``policy.tokenizer`` from the SFT recipe (default: ``sft-nanov3-30BA3B-1n8g.yaml``),
loads weights from a local ``hf_consolidated`` directory, formats messages with
``apply_chat_template`` using the **same** ``add_generation_prompt`` behavior as SFT
(``data.add_generation_prompt`` from the recipe when present, otherwise ``false`` as in
``examples/configs/sft.yaml``), then runs ``generate``.

**Important:** NeMo SFT defaults to ``data.add_generation_prompt: false``. Using
``add_generation_prompt=True`` here when training used ``false`` injects an extra
assistant / thinking prefix; the model then often repeats ``<|im_start|>assistant``
and looks broken.

Example:
  uv run scripts/test_nemotron_consolidated_chat.py \\
    --model-path /scratch/dhruv/sft/sft-nanov3-30BA3B-1n8g/step_1850/hf_consolidated \\
    --recipe examples/configs/recipes/llm/sft-nanov3-30BA3B-1n8g.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoModelForCausalLM

# Avoid cuDNN-backed SDPA when system libcudnn mismatches the PyTorch wheel (cudnnGetLibConfig errors).
# Same workaround as NeMo RL automodel / sft-nanov3 recipe comments (TORCH_CUDNN_SDPA_ENABLED=0).
os.environ.setdefault("TORCH_CUDNN_SDPA_ENABLED", "0")
torch.backends.cuda.enable_cudnn_sdp(False)

from nemo_rl.algorithms.utils import get_tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("/scratch/dhruv/sft/sft-nanov3-30BA3B-1n8g/step_1850/hf_consolidated"),
        help="Directory with consolidated HF weights (config + model + tokenizer files).",
    )
    p.add_argument(
        "--recipe",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "examples/configs/recipes/llm/sft-nanov3-30BA3B-1n8g.yaml",
        help="SFT YAML containing policy.tokenizer (same as training).",
    )
    p.add_argument(
        "--user-message",
        default="Give a one-sentence explanation of what a MoE transformer is.",
        help="Single user turn for the chat template.",
    )
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--device-map",
        default="auto",
        help='Transformers device_map (e.g. "auto", "cuda:0", or "cpu").',
    )
    p.add_argument(
        "--add-generation-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Passed to apply_chat_template. Default: recipe's data.add_generation_prompt, "
            "else false (NeMo sft.yaml). Must match training or the prompt prefix is wrong."
        ),
    )
    return p.parse_args()


def add_generation_prompt_from_recipe(recipe_path: Path) -> bool | None:
    """Return data.add_generation_prompt if set in this YAML file; None if omitted.

    Note: Hydra ``defaults:`` merges are not applied by OmegaConf.load() on the recipe
    alone; if the key is missing, callers should fall back to ``examples/configs/sft.yaml``
    default (false).
    """
    cfg = OmegaConf.load(recipe_path)
    if not OmegaConf.select(cfg, "data"):
        return None
    if "add_generation_prompt" not in cfg.data:
        return None
    return bool(cfg.data.add_generation_prompt)


def tokenizer_config_from_recipe(recipe_path: Path) -> dict:
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Recipe not found: {recipe_path}")
    cfg = OmegaConf.load(recipe_path)
    if not OmegaConf.select(cfg, "policy.tokenizer"):
        raise ValueError(f"No policy.tokenizer in {recipe_path}")
    raw = OmegaConf.to_container(cfg.policy.tokenizer, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("policy.tokenizer must be a mapping")
    return raw


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        print(f"error: model directory not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    tok_cfg = tokenizer_config_from_recipe(args.recipe.resolve())
    tokenizer = get_tokenizer(tok_cfg)

    if not getattr(tokenizer, "chat_template", None):
        print(
            "error: tokenizer.chat_template is empty after get_tokenizer; "
            "fix policy.tokenizer in the recipe.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.add_generation_prompt is not None:
        add_gen = args.add_generation_prompt
    else:
        from_recipe = add_generation_prompt_from_recipe(args.recipe.resolve())
        add_gen = from_recipe if from_recipe is not None else False

    messages = [{"role": "user", "content": args.user_message}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_gen,
    )
    print(f"apply_chat_template(add_generation_prompt={add_gen})  # SFT default is false (see data.add_generation_prompt in examples/configs/sft.yaml)\n")
    print("--- Prompt (chat-templated, first 800 chars) ---")
    print(prompt[:800] + ("…" if len(prompt) > 800 else ""))
    print("--- End prompt ---\n")

    dtype = torch.bfloat16 if args.device_map != "cpu" else torch.float32
    print(f"Loading model from {model_path} (dtype={dtype}, device_map={args.device_map!r}) …")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")
    inputs = {k: v.to(dev) for k, v in inputs.items()}

    in_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=max(args.temperature, 1e-6),
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0, in_len:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    print("--- Generated (decoded new tokens) ---")
    print(text)
    print("--- Done ---")


if __name__ == "__main__":
    main()
