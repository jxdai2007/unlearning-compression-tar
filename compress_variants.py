#!/usr/bin/env python3
"""Magnitude pruning CLI for compression variants.

Usage:
    python compress_variants.py --sparsity 0.10 --model MODEL_ID --out OUTPUT_DIR

Loads a model, applies GLOBAL unstructured magnitude pruning (zero smallest |w|
across all Linear weights, embeddings and lm_head excluded), saves fp16.
"""

import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def magnitude_prune(model, sparsity: float):
    """Apply global unstructured magnitude pruning to Linear/Conv1D layers only.

    Args:
        model: HF model to prune
        sparsity: Fraction of weights to zero (0.10 = 10% sparse)

    Returns:
        model with pruned weights (modified in-place)
    """
    # Collect all Linear/Conv1D layer weights (exclude embeddings/lm_head)
    all_weights = []
    target_modules = []

    for name, module in model.named_modules():
        # Handle both Linear and Conv1D (GPT-2 uses transformers.pytorch_utils.Conv1D)
        is_linear = isinstance(module, nn.Linear)
        # Check for transformers Conv1D by class name
        is_conv1d = type(module).__name__ == "Conv1D"

        if not (is_linear or is_conv1d):
            continue

        # Skip embeddings and lm_head
        if "embed" in name or "lm_head" in name:
            continue

        weight = module.weight.data.flatten()
        all_weights.append(weight)
        target_modules.append(module)

    if not all_weights:
        raise ValueError("No Linear/Conv1D modules found (excluding embeddings/lm_head)")

    # Global threshold via deterministic strided sampling. A full
    # cat+sort materializes ~13.5GB of values plus ~54GB of int64 sort
    # indices on an 8B model (OOM-killed job 21460513); a strided sample
    # of ~30M values estimates the quantile well inside the 0.01
    # achieved-sparsity tolerance and needs ~120MB.
    total = sum(w.numel() for w in all_weights)
    budget = 30_000_000
    stride = max(1, total // budget)
    samples = [w[::stride].abs().float() for w in all_weights]
    sample = torch.cat(samples)
    # torch.quantile rejects inputs over ~16M elements; kthvalue scales
    k = max(1, int(sparsity * sample.numel()))
    threshold = sample.kthvalue(k).values.item()

    # fp16 weights tie heavily at the threshold value; zeroing only
    # strictly-below undershoots the target. Zero strictly-below, then
    # zero just enough tie-valued weights to reach the exact count.
    k_total = int(round(sparsity * total))
    n_lt = sum(int((m.weight.data.abs() < threshold).sum()) for m in target_modules)
    deficit = max(0, k_total - n_lt)

    for module in target_modules:
        w = module.weight.data
        if deficit > 0:
            ties = (w.abs() == threshold).view(-1).nonzero().squeeze(-1)
            if ties.numel():
                take = min(deficit, int(ties.numel()))
                w.view(-1)[ties[:take]] = 0.0
                deficit -= take
        w.masked_fill_(w.abs() < threshold, 0.0)

    zeroed = sum(int((m.weight.data == 0.0).sum()) for m in target_modules)
    achieved = zeroed / total
    print(f"achieved_sparsity={achieved:.4f} target={sparsity:.4f} threshold={threshold:.3e}")
    if abs(achieved - sparsity) > 0.01:
        raise RuntimeError(
            f"achieved sparsity {achieved:.4f} outside 0.01 of target {sparsity:.4f}"
        )

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Apply global magnitude pruning to a model"
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        required=True,
        help="Fraction of weights to zero (e.g., 0.10 for 10%% sparse)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model ID or local path",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory for pruned model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HF tokenizer ID/model (if different from --model)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16
    )

    tokenizer_id = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        # Fallback for models without tokenizer (like TAR-Bio)
        print(f"Tokenizer not found at {tokenizer_id}, using default")
        tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

    print(f"Applying magnitude pruning at {args.sparsity:.2%} sparsity")
    model = magnitude_prune(model, args.sparsity)

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving pruned model to {out_path}")
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)

    print("Done")


if __name__ == "__main__":
    main()
