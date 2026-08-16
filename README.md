# Does compression reverse tamper-resistant unlearning?

An audit of the released TAR-Bio checkpoint ([Tampering-Resistant Safeguards](https://arxiv.org/abs/2408.00761),
Tamirisa et al.): does routine post-training compression — quantization and
magnitude pruning — bring back the hazardous biology capability that TAR was
trained to remove?

Replication-and-extend project. LessWrong post: _(link to be added)_

## Motivation

> _(Framing paragraph — author to finalize in their own voice.)_

Open-weight safety has to survive normal handling. Nearly every deployed model
is quantized or pruned first, and prior work shows compression can partially
reverse ordinary unlearning. TAR is the strong version of the claim — a model
*adversarially trained* to resist tampering. This project tests whether that
resistance holds under the most mundane perturbation there is.

## Setup

- **Anchor:** Tamirisa et al., [arXiv:2408.00761](https://arxiv.org/abs/2408.00761)
  (v4, Table 1); checkpoint `lapisrocks/Llama-3-8B-Instruct-TAR-Bio-v2`.
- **Forget metric:** WMDP-bio, 5-shot, n=1273 (4-choice; chance = 25.0).
- **Retain metric:** MMLU over the paper's 50-subject bio-excluded retain set,
  5-shot, macro and micro.
- **Harness:** lm-eval for every cell including the intact baseline (same batch),
  so all deltas are within-harness. A separate leg runs the TAR authors' own
  FSDP MMLU harness for the retain-discrepancy finding.

## What we test

| Family | Variants |
|--------|----------|
| Quantization | 8-bit, 4-bit NF4 (bitsandbytes, load-time) |
| Magnitude pruning | 10% / 15% / 20% / 30% / 50% global unstructured sparsity |

The 15% and 20% points specifically cover the sparsity window where prior work
([hannahTao, *Does routine compression undo LLM unlearning?*](https://www.lesswrong.com/posts/jXhHH658J4xzWjCu8/does-routine-compression-undo-llm-unlearning-a-short-project))
found peak recovery on weaker unlearning methods.

## Findings

1. **Compression recovers no hazardous capability at any level** — every
   WMDP-bio cell stays at or below the 25.0 chance floor (24.04–24.35),
   including the 15–20% pruning window.
2. **Benign capability degrades first** — retain-MMLU falls gently to ~20%
   sparsity (−3.4 pp), then collapses (−11.7 pp at 30%, to the floor at 50%).
3. **The released checkpoint scores above the paper's published retain number
   in the authors' own harness** — 58.57 macro / 56.10 micro vs the published
   54.7 (+8.8 / +3.2 SE). Direction is flattering to the method; cause is
   undetermined from outside.

Full table, figure, and framing:
[`results/2026-08-16-extension.md`](results/2026-08-16-extension.md).

## Reproduce

```bash
pip install -e common
# build pruned variants (asserts achieved sparsity within 0.01):
python compress_variants.py --sparsity 0.20 --model <checkpoint> --out <dir>
# WMDP-bio at genuine k-shot (works around lm-eval's silent num_fewshot drop):
python wmdp_k5_eval.py --model-args <args> --out <dir>
# aggregate all cells into the frontier table + figure:
python sweep_readout.py --extension-dir results/raw-extension
python figures_post.py
```

Note: the `wmdp_bio` task YAML pins `num_fewshot=0` and silently discards the
CLI flag; `wmdp_k5_eval.py` forces it via `set_config` and asserts the recorded
n-shot. See the results doc for the full protocol-trap list.

## Layout

- `*.py` — variant builder, k-shot eval driver, aggregator, figures
- `slurm/` — cluster job templates
- `results/` — results doc, per-cell lm-eval JSONs, figure
- `common/jrp_common/` — shared capability-eval harness

## Citation

Please also cite the anchor paper (arXiv:2408.00761). This does **not** claim
TAR is broken — the tamper-resistance claim survives this attack family.

## License

MIT.
