#!/usr/bin/env python3
"""Publication figure for the 02 post: the compression frontier.

One chart carries the whole result. Both benchmarks are 4-choice, so both
share a chance floor at 25.0 — which makes the point visible without
explanation: the forget metric never leaves the floor at any compression
level, while the retain metric falls from 60 all the way down to meet it.

Usage: python figures_post.py [--results results/raw-extension]
"""

import argparse
import glob
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIO_FORGET = {
    "college_biology", "high_school_biology", "medical_genetics", "virology",
    "college_medicine", "anatomy", "clinical_knowledge",
}
CHANCE = 25.0
BG = "#FAFAFA"

# Ordered by increasing aggressiveness within family; label carries the knob
ORDER = [
    ("intact", "intact"),
    ("8bit", "8-bit"),
    ("4bit", "4-bit NF4"),
    ("mag10", "prune 10%"),
    ("mag15", "prune 15%"),
    ("mag20", "prune 20%"),
    ("mag30", "prune 30%"),
    ("mag50", "prune 50%"),
]


def load(root):
    cells = {}
    for path in sorted(glob.glob(os.path.join(root, "*=*", "*", "results*.json"))):
        cell_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        variant, task = cell_dir.split("=", 1)
        if task not in ("mmlu", "wmdp_bio"):
            continue
        d = json.load(open(path))
        res, nsamp = d["results"], d.get("n-samples", {})
        nshot = d.get("n-shot", {})
        if any(v != 5 for v in nshot.values()):
            continue  # n-shot gate: 0-shot cells never enter the figure
        if task == "wmdp_bio":
            n = int(nsamp.get("wmdp_bio", {}).get("effective", 1273))
            acc = 100.0 * float(res["wmdp_bio"]["acc,none"])
            cells.setdefault(variant, {})["wmdp"] = (acc, se(acc, n))
        else:
            subs = {k[5:]: (100.0 * float(v["acc,none"]),
                            int(nsamp.get(k, {}).get("effective", 0)))
                    for k, v in res.items()
                    if k.startswith("mmlu_") and "acc,none" in v and k in nsamp}
            keep = {s: v for s, v in subs.items() if s not in BIO_FORGET}
            macro = sum(a for a, _ in keep.values()) / len(keep)
            mse = 100.0 * math.sqrt(
                sum((a / 100) * (1 - a / 100) / n for a, n in keep.values())
            ) / len(keep)
            cells.setdefault(variant, {})["mmlu"] = (macro, mse)
    return cells


def se(pct, n):
    p = pct / 100.0
    return 100.0 * math.sqrt(max(p * (1 - p), 1e-9) / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/raw-extension")
    ap.add_argument("--out", default="results/figures-post/fig1_frontier.png")
    args = ap.parse_args()

    cells = load(args.results)
    present = [(v, lbl) for v, lbl in ORDER if v in cells]
    xs = range(len(present))

    plt.rcParams.update({"font.size": 12, "figure.facecolor": BG,
                         "axes.facecolor": BG, "savefig.facecolor": BG})
    fig, ax = plt.subplots(figsize=(9, 4.6))

    # Pruning is a genuine ordered sweep, so its points are connected.
    # Quantization is not on the same axis as sparsity, so those points are
    # left unconnected — joining them would imply a severity continuum that
    # does not exist.
    n_quant = sum(1 for v, _ in present if not v.startswith("mag"))
    for key, label, color, marker in (
        ("mmlu", "MMLU (retain, 50 non-bio subjects)", "#1f4e79", "o"),
        ("wmdp", "WMDP-bio (forget)", "#c0392b", "s"),
    ):
        pts = [(i, *cells[v][key]) for i, (v, _) in enumerate(present)
               if key in cells[v]]
        if not pts:
            continue
        quant = [p for p in pts if p[0] < n_quant]
        prune = [p for p in pts if p[0] >= n_quant]
        ax.errorbar([p[0] for p in quant], [p[1] for p in quant],
                    yerr=[1.96 * p[2] for p in quant], fmt=marker,
                    color=color, ms=7, capsize=3, label=label, ls="none")
        ax.errorbar([p[0] for p in prune], [p[1] for p in prune],
                    yerr=[1.96 * p[2] for p in prune], fmt=marker + "-",
                    color=color, ms=7, lw=2, capsize=3)

    if 0 < n_quant < len(present):
        ax.axvline(n_quant - 0.5, color="#B0B8C0", lw=1, ls=":")
        ax.text(n_quant / 2 - 0.5, 64.2, "quantization", fontsize=10,
                ha="center", color="#555")
        ax.text((n_quant + len(present) - 1) / 2, 64.2,
                "magnitude pruning (sparsity sweep)", fontsize=10,
                ha="center", color="#555")

    ax.axhline(CHANCE, color="k", ls="--", lw=1.2)
    ax.text(-0.35, CHANCE + 0.9, "chance (25.0)", fontsize=10, ha="left")
    ax.set_xticks(list(xs), [lbl for _, lbl in present], rotation=25,
                  ha="right")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(18, 66)
    ax.set_xlim(-0.6, len(present) - 0.4)
    ax.set_title("Compressing a tamper-resistant checkpoint destroys benign "
                 "capability\nbefore it recovers any hazardous capability",
                 loc="left", pad=26)
    # mid-left is the only region free of data across all variants
    ax.legend(frameon=False, loc="center left", fontsize=10,
              bbox_to_anchor=(0.01, 0.52))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(args.out)
    for v, lbl in present:
        c = cells[v]
        m = f"{c['mmlu'][0]:.2f}" if "mmlu" in c else "--"
        w = f"{c['wmdp'][0]:.2f}" if "wmdp" in c else "--"
        print(f"  {lbl:<12} MMLU {m:>6}  WMDP {w:>6}")


if __name__ == "__main__":
    main()
