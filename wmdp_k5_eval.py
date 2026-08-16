#!/usr/bin/env python3
"""WMDP-bio at genuine k-shot, bypassing lm-eval's silent num_fewshot override.

The wmdp_bio task YAML pins num_fewshot=0; both the CLI --num_fewshot flag and
simple_evaluate(num_fewshot=...) are discarded with only a logger.info
(evaluator.py:327 — documented in results/2026-08-14-gate.md, fix stack item 1).
The working fix, verified during the gate (few-shot context grew 365 -> 2280
chars): set_config on the task object, assert it stuck, pass the task dict
through, and assert the recorded n-shot afterward.

Usage:
    python wmdp_k5_eval.py --model-args pretrained=...,dtype=float16 \
        --out /scratch/.../<variant>=wmdp_bio/k5 [--k 5]
"""

import argparse
import json
import os

from lm_eval import simple_evaluate
from lm_eval.tasks import get_task_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-args", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    task_dict = get_task_dict(["wmdp_bio"])
    task = task_dict["wmdp_bio"]
    task.set_config(key="num_fewshot", value=args.k)
    assert task.config.num_fewshot == args.k, (
        f"set_config did not stick: {task.config.num_fewshot}"
    )

    res = simple_evaluate(
        model="hf",
        model_args=args.model_args,
        # this lm-eval's task manager passes Task OBJECTS through untouched but
        # chokes on a {name: task} dict — pass the configured object in a list
        tasks=[task],
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
    )

    n_shot = res.get("n-shot", {}).get("wmdp_bio")
    assert n_shot == args.k, f"recorded n-shot {n_shot} != {args.k}"

    os.makedirs(args.out, exist_ok=True)
    slim = {
        k: v
        for k, v in res.items()
        if k in ("results", "n-shot", "configs", "n-samples", "versions",
                 "higher_is_better")
    }
    out_path = os.path.join(args.out, f"results_k{args.k}.json")
    with open(out_path, "w") as f:
        json.dump(slim, f, indent=2, default=str)

    acc = res["results"]["wmdp_bio"]["acc,none"]
    print(f"wmdp_bio acc={100 * acc:.2f} n-shot={n_shot} -> {out_path}")


if __name__ == "__main__":
    main()
