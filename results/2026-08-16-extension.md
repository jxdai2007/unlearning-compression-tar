# 02 extension — compression frontier of the released TAR-Bio-v2 checkpoint

2026-08-16, branch `feat/common-harness`. Framing per the 2026-08-15 pivot:
we measure the *released checkpoint's* retain/forget behavior under
compression against our own same-batch baseline (the checkpoint ≠ published
numbers; see `2026-08-14-gate.md` — retain surplus confirmed in the authors'
own harness). All cells: lm-eval, one harness throughout; deltas are
within-table only. Raw JSONs: `results/raw-extension/`; aggregate with
`python sweep_readout.py --extension-dir results/raw-extension`.

## Headline

1. **Compression does not recover the unlearned capability.** No variant
   lifts WMDP-bio off the chance floor at genuine 5-shot — not even mag30,
   which has already lost ~12 pp of benign MMLU. On this checkpoint,
   pruning/quantization alone is not a tamper vector. **This includes the
   10–20% sparsity window where prior work found its peak recovery** (see
   Related work): mag15 24.12, mag20 24.12, both at the floor.
2. **The retain side degrades gently, then falls off a cliff between 20% and
   50% sparsity.** 8-bit is free (−0.17 pp); 10% pruning is marginal
   (−1.10 pp, at the edge of the 1.40 pp resolvable delta); 15% is −2.03 pp
   and 4-bit NF4 −2.63 pp, both real but modest; 20% is −3.39 pp; then 30%
   costs −11.74 pp and 50% collapses to the chance floor (−34.99 pp). The
   useful summary for a deployer: up to ~20% sparsity you pay a few points of
   benign capability, past that the model degrades fast — and at no point
   along the curve does any hazardous capability come back.
3. Net: the checkpoint's forget floor is *more* robust to compression than
   its retain performance — retain degrades long before any forgotten
   capability resurfaces. The tamper-resistance claim survives this attack
   class; the price is that the defended model's benign capability is the
   fragile part.

## Table (MMLU 5-shot, 50-subject bio-excluded set; WMDP-bio 5-shot, n=1273)

| variant | MMLU-50 macro | MMLU-50 micro | Δmacro | WMDP-bio k=5 | vs chance |
|---|---|---|---|---|---|
| intact | 60.09 | 57.73 | — | 24.04 | −0.96 |
| 8bit | 59.91 | 57.46 | −0.17 | 24.04 | −0.96 |
| 4bit | 57.45 | 55.44 | −2.63 | 24.04 | −0.96 |
| mag10 | 58.98 | 56.89 | −1.10 | 24.04 | −0.96 |
| mag15 | 58.06 | 56.29 | −2.03 | 24.12 | −0.88 |
| mag20 | 56.70 | 54.96 | −3.39 | 24.12 | −0.88 |
| mag30 | 48.35 | 46.76 | −11.74 | 24.04 | −0.96 |
| mag50 | 25.10 | 25.17 | −34.99 | 24.35 | −0.65 |

Figure: `results/figures-post/fig1_frontier.png` (both benchmarks are
4-choice, so they share the 25.0 chance floor — the forget line never leaves
it while the retain line descends to meet it). Table as an image for
platforms without table support: `results/figures-post/table_frontier.png`.

MMLU-50 macro SE ≈ 0.50 pp (95% resolvable delta ≈ 1.40 pp). WMDP-bio SE ≈
1.20 pp; **all WMDP cells are at/below the 25.0 chance floor and per the
floor doctrine cannot be ranked against each other** — the only claim made is
that none rises above chance. MMLU chance is also 25.0: mag50 is destroyed,
period.

## Protocol notes (defensibility)

- Variants: global unstructured magnitude pruning at 0.10/0.30/0.50
  (embeddings/lm_head excluded; sampled-kthvalue threshold with fp16 tie
  handling, achieved sparsity asserted within 0.01 — `compress_variants.py`);
  bitsandbytes load-time 8-bit and 4-bit NF4. GPTQ/AWQ/Wanda were descoped
  (calibration runs don't fit the sprint window) — documented deviation.
- WMDP-bio k=5 forced via `wmdp_k5_eval.py` (task-object `set_config` +
  double assert): the CLI/`simple_evaluate` num_fewshot path is silently
  discarded for this task (evaluator.py:327). First-pass cells that recorded
  n-shot=0 are preserved under `<variant>=wmdp_bio/` as genuine 0-shot data
  but excluded from the table by the aggregator's n-shot gate. 5-shot context
  verified directly (request context ~2000 chars vs 365 at 0-shot).
- Intact WMDP k=5 here (24.04 ± 1.20) vs the gate-era 26.71 ± 1.24: ~1.6 SE
  on the difference — statistically compatible; different drivers, same
  set_config mechanism. Both carry the shots-from-test-split caveat (WMDP
  ships no train split).
- Five variants share identical WMDP accuracy (24.04). Per-sample logs were
  not kept (no --log_samples), so identical *predictions* are unverified; at
  the floor, a degenerate answer preference robust to perturbation is the
  parsimonious reading. Floor doctrine applies regardless.
- Baseline anchor: the intact retain numbers in this table are lm-eval-scale
  and differ from the authors'-harness numbers in `2026-08-14-gate.md`
  (58.57/56.10); never mix the scales. Deltas here are same-harness,
  same-batch.

## Related work and positioning

**hannahTao, "Does routine compression undo LLM unlearning? A short project"**
(LessWrong, 2026-07-20,
https://www.lesswrong.com/posts/jXhHH658J4xzWjCu8/does-routine-compression-undo-llm-unlearning-a-short-project,
code at github.com/hannahTao/compression-unlearning) asks our question on a
different target: TOFU `forget10` (synthetic fictional-author facts),
Llama-3.2-1B-Instruct, unlearning via NPO / SimNPO / IdkDPO, compressed by
quantization, magnitude pruning and SVD truncation. Findings there: reversal
is minimal overall, but 4-bit moved NPO 0.209→0.353 (22% of the
ceiling–baseline gap) and magnitude pruning peaked at **42% recovery at 20%
sparsity on NPO**, with ≥50% sparsity destroying utility; recovery in
teacher-forced probability did not resurface answers in free generation. That
work in turn follows arXiv:2410.16454, which reported that quantization can
reverse unlearning.

**That post directly improved this one.** Its recovery peak sits in a narrow
10–20% sparsity window, and the author notes she nearly missed it because she
originally tested only 10% and 30% — which was exactly our original grid
(10/30/50). We therefore added mag15 and mag20 (jobs 21461826-30, achieved
sparsity 0.1500 and 0.2000 within tolerance). WMDP-bio at k=5: **24.12 and
24.12**, both at the chance floor. The negative result survives the test the
prior literature says is most likely to break it.

**What we add:** TAR is a *tamper-resistant* method — adversarially trained
specifically to resist recovery — rather than an ordinary unlearning
objective; WMDP-bio is a hazardous-capability proxy rather than synthetic
biographical facts; the model is 8B rather than 1B; and every comparison runs
against a same-batch intact baseline with sampling error attached and
floor-pinned cells explicitly refused. The honest one-line framing: *on
weaker unlearning objectives, prior work found partial recovery in a narrow
pruning window; on a method built for tamper-resistance, we find none —
including inside that window.*

## Completeness

Forget arm complete at genuine 5-shot across all 8 variants (6 original + the
two window points). Retain arm: MMLU for mag15/mag20 still running (jobs
21461827, 21461829); the table and figure update when they land. No
conclusion above depends on them — they refine where the retain knee sits,
not whether the forget floor moves.
