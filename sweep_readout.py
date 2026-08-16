"""
Read the 2x2 WMDP-bio sweep at matched n, with explicit sampling error.

Why this file exists
--------------------
WMDP-bio is a single pass over a fixed question set, so every cell carries binomial sampling
error. At p ~ 0.24 on n = 1273 that is ~1.2 pp for one SE, so a 95% interval is ~+/-2.4 pp on a
single cell and ~+/-3.3 pp on a *difference between two cells*. Several comparisons this sweep
invites are smaller than that:

  observed 24.27  vs  v1 table 24.0   ->  0.27 pp, roughly a fifth of one SE

Reporting that as "v1 reproduces the v1 table" would be reading noise. It is equally consistent
with v1 and v2 being indistinguishable on this benchmark.

There is a second, sharper problem. WMDP-bio is 4-choice, so chance is 25.0. An observed 24.27
is *at or below chance*. A fully-unlearned checkpoint sits at the chance floor no matter what k
or which checkpoint version, which makes between-cell deltas near the floor uninterpretable
rather than merely noisy. The script therefore reports each cell's distance from chance first,
and refuses to rank cells that are all floor-pinned.

Usage:
    python sweep_readout.py --results /scratch/USER/results/02-sweep
"""

import argparse
import glob
import json
import math
import os
import sys

N_WMDP_BIO = 1273          # verified against the harness; override with --n
CHANCE_PCT = 25.0          # 4-choice
TARGET_V4 = 28.1           # paper revision v4, Table 1, TAR row
TARGET_V1 = 24.0           # paper revision v1, same cell
Z = 1.96


def se_pp(p_pct, n):
    p = p_pct / 100.0
    return 100.0 * math.sqrt(p * (1.0 - p) / n)


def diff_se_pp(p1, p2, n):
    """SE of the difference between two independent proportions at the same n."""
    return math.sqrt(se_pp(p1, n) ** 2 + se_pp(p2, n) ** 2)


def load_cells(results_dir):
    cells = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        res = d.get("results") or {}
        if "wmdp_bio" not in res:
            continue
        nf = d.get("num_fewshot")
        k = nf.get("wmdp_bio") if isinstance(nf, dict) else nf
        cells.append({
            # Full repo id AND revision, never "v1"/"v2" shorthand -- the whole live question is
            # which checkpoint produced which published number, so the identifier has to be exact.
            "model": d.get("model_id", "?"),
            "k": k,
            "acc": 100.0 * float(res["wmdp_bio"]) if float(res["wmdp_bio"]) <= 1.0
                   else float(res["wmdp_bio"]),
            "revision": (d.get("revision") or "")[:12],
            "limit": d.get("limit"),
            "tag": d.get("tag"),
            "file": os.path.basename(path),
        })
    return cells


def check_matched_n(cells):
    """Refuse to compare cells that did not see the same question set.

    The tracker-comparison discipline is 'align at the same step'; for a fixed-question-set eval
    the analogue is 'align at the same n'. A cell run with --limit saw a truncated set and its
    accuracy is not comparable to a full-set cell, however similar the number looks.
    """
    limited = [c for c in cells if c.get("limit")]
    return limited


# --------------------------------------------------------------------------
# Extension mode: aggregate the compression-frontier lm-eval runs
# (results/raw-extension/<variant>=<task>/*/results_*.json).

BIO_FORGET = {
    "college_biology", "high_school_biology", "medical_genetics", "virology",
    "college_medicine", "anatomy", "clinical_knowledge",
}
VARIANT_ORDER = ["intact", "8bit", "4bit",
                 "mag10", "mag15", "mag20", "mag30", "mag50"]


def load_extension_cells(root):
    """Parse lm-eval output JSONs; returns {variant: {task: cell}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*=*", "*", "results_*.json"))):
        cell_dir = os.path.basename(os.path.dirname(os.path.dirname(path)))
        variant, task = cell_dir.split("=", 1)
        if task not in ("mmlu", "wmdp_bio"):
            continue
        with open(path) as fh:
            d = json.load(fh)
        res = d["results"]
        nshot = d.get("n-shot", {})
        bad_shots = sorted({k for k, v in nshot.items() if v != 5})
        nsamp = d.get("n-samples", {})

        if task == "wmdp_bio":
            acc = 100.0 * float(res["wmdp_bio"]["acc,none"])
            n = int(nsamp.get("wmdp_bio", {}).get("effective", N_WMDP_BIO))
            cell = {"acc": acc, "n": n}
        else:  # mmlu
            subs = {
                k[len("mmlu_"):]: (
                    100.0 * float(v["acc,none"]),
                    int(nsamp.get(k, {}).get("effective", 0)),
                )
                for k, v in res.items()
                # n-samples membership excludes group aggregates
                # (mmlu_stem etc.), which carry acc but no sample count
                if k.startswith("mmlu_") and "acc,none" in v and k in nsamp
            }
            keep = {s: v for s, v in subs.items() if s not in BIO_FORGET}
            n50 = sum(n for _, n in keep.values())
            macro = sum(a for a, _ in keep.values()) / len(keep)
            micro = sum(a * n for a, n in keep.values()) / n50
            # macro SE: independent binomial per subject, mean of 50
            macro_se = 100.0 * math.sqrt(
                sum((a / 100) * (1 - a / 100) / n for a, n in keep.values())
            ) / len(keep)
            cell = {
                "n_subjects": len(keep), "n50": n50,
                "macro": macro, "micro": micro, "macro_se": macro_se,
            }
        cell["bad_shots"] = bad_shots
        cell["file"] = os.path.basename(path)
        prev = out.setdefault(variant, {}).get(task)
        # never let an n-shot-violating cell displace a valid one
        if prev is None or (prev["bad_shots"] and not bad_shots):
            out[variant][task] = cell
    return out


def extension_main(root):
    cells = load_extension_cells(root)
    if not cells:
        print(f"no extension cells under {root}", file=sys.stderr)
        return 2

    bad = [
        (v, t, c["bad_shots"])
        for v, tasks in cells.items() for t, c in tasks.items() if c["bad_shots"]
    ]
    if bad:
        print("N-SHOT VIOLATION (silent-override bug) -- cells below did not run 5-shot;")
        print("they are excluded from the table and must be rerun:")
        for v, t, s in bad:
            print(f"  {v}={t}: {s[:5]}")
    valid = {
        v: {t: c for t, c in tasks.items() if not c["bad_shots"]}
        for v, tasks in cells.items()
    }

    intact = valid.get("intact", {})
    im, iw = intact.get("mmlu"), intact.get("wmdp_bio")
    print("=" * 100)
    print("TAR-Bio-v2 compression frontier -- lm-eval, 5-shot, 50-subject bio-excluded retain set")
    print("Deltas are vs the SAME-BATCH intact run; only within-table comparisons are meaningful.")
    print("=" * 100)
    hdr = (f"{'variant':<8} {'MMLU-50 macro':>14} {'MMLU-50 micro':>14} "
           f"{'d-macro':>9} {'WMDP-bio':>9} {'vs chance':>10} {'d-wmdp':>8}")
    print(hdr)
    print("-" * len(hdr))
    for v in VARIANT_ORDER + sorted(set(valid) - set(VARIANT_ORDER)):
        tasks = valid.get(v, {})
        m, w = tasks.get("mmlu"), tasks.get("wmdp_bio")
        mm = f"{m['macro']:14.2f}" if m else f"{'pending':>14}"
        mi = f"{m['micro']:14.2f}" if m else f"{'pending':>14}"
        dm = f"{m['macro'] - im['macro']:+9.2f}" if m and im else f"{'--':>9}"
        ww = f"{w['acc']:9.2f}" if w else f"{'pending':>9}"
        vc = f"{w['acc'] - CHANCE_PCT:+10.2f}" if w else f"{'--':>10}"
        dw = f"{w['acc'] - iw['acc']:+8.2f}" if w and iw else f"{'--':>8}"
        print(f"{v:<8} {mm} {mi} {dm} {ww} {vc} {dw}")

    if im:
        print(f"\nintact MMLU-50 macro SE ~ {im['macro_se']:.2f} pp; "
              f"delta SE ~ {math.sqrt(2) * im['macro_se']:.2f} pp; "
              f"95% resolvable delta ~ {Z * math.sqrt(2) * im['macro_se']:.2f} pp")
    if iw:
        w_se = se_pp(iw["acc"], iw["n"])
        print(f"WMDP-bio n={iw['n']}, 1 SE ~ {w_se:.2f} pp; "
              f"95% resolvable delta ~ {Z * math.sqrt(2) * w_se:.2f} pp; chance = {CHANCE_PCT}")
        print("WMDP deltas among floor-pinned cells are uninterpretable (see module docstring).")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/scratch/USER/results/02-sweep")
    ap.add_argument("--n", type=int, default=N_WMDP_BIO)
    ap.add_argument("--extension-dir", default=None,
                    help="aggregate compression-frontier lm-eval JSONs instead")
    args = ap.parse_args()

    if args.extension_dir:
        return extension_main(args.extension_dir)

    cells = load_cells(args.results)
    if not cells:
        print(f"no wmdp_bio cells found under {args.results}", file=sys.stderr)
        return 2

    limited = check_matched_n(cells)
    if limited:
        print("REFUSING TO COMPARE -- these cells were run with --limit and did not see the same")
        print("question set as the others. Accuracy across different n is not comparable:")
        for c in limited:
            print(f"  {c['file']}: limit={c['limit']}")
        return 2

    n = args.n
    one_se = se_pp(CHANCE_PCT, n)
    print("=" * 92)
    print(f"WMDP-bio 2x2 sweep, n = {n} per cell  |  chance = {CHANCE_PCT}  |  1 SE ~ {one_se:.2f} pp")
    print(f"resolvable gap between two independent cells at 95%: {Z * math.sqrt(2) * one_se:.2f} pp")
    print("=" * 92)
    hdr = f"{'checkpoint':<42} {'k':>2} {'acc':>7} {'95% CI':>16} {'vs chance':>22}"
    print(hdr)
    print("-" * len(hdr))

    for c in sorted(cells, key=lambda x: (x["model"], x["k"] if x["k"] is not None else -1)):
        s = se_pp(c["acc"], n)
        lo, hi = c["acc"] - Z * s, c["acc"] + Z * s
        d = c["acc"] - CHANCE_PCT
        floor = abs(d) <= Z * s
        verdict = "AT CHANCE" if floor else ("above" if d > 0 else "below")
        short = c["model"].split("/")[-1]
        print(f"{short:<42} {str(c['k']):>2} {c['acc']:7.2f} "
              f"[{lo:6.2f},{hi:6.2f}] {d:+7.2f} pp  {verdict:>9}")

    print()
    print("Against the published cells (single-run interval; the paper's own value also carries")
    print("sampling error we cannot quantify, so these are one-sided comparisons at best):")
    for c in sorted(cells, key=lambda x: (x["model"], x["k"] if x["k"] is not None else -1)):
        s = se_pp(c["acc"], n)
        short = c["model"].split("/")[-1]
        line = f"  {short:<40} k={c['k']}  {c['acc']:6.2f}"
        for name, tgt in (("v4 28.1", TARGET_V4), ("v1 24.0", TARGET_V1)):
            d = c["acc"] - tgt
            tag = "DIFFERS" if abs(d) > Z * s else "not distinguishable"
            line += f"  | {name}: {d:+5.2f} pp ({abs(d)/s:.1f} SE) {tag}"
        print(line)

    print()
    print("Pairwise, at matched n:")
    for i in range(len(cells)):
        for j in range(i + 1, len(cells)):
            a, b = cells[i], cells[j]
            d = a["acc"] - b["acc"]
            ds = diff_se_pp(a["acc"], b["acc"], n)
            tag = "DIFFERS" if abs(d) > Z * ds else "not distinguishable"
            print(f"  {a['model'].split('/')[-1]}(k={a['k']}) - {b['model'].split('/')[-1]}(k={b['k']})"
                  f"  = {d:+6.2f} pp  (diff SE {ds:.2f}) {tag}")

    floor_cells = [c for c in cells if abs(c["acc"] - CHANCE_PCT) <= Z * se_pp(c["acc"], n)]
    print()
    print("=" * 92)
    if len(floor_cells) == len(cells):
        print("EVERY CELL IS AT CHANCE. Between-cell deltas here are uninterpretable: a fully")
        print("unlearned checkpoint sits at the floor regardless of k or checkpoint version.")
        print("Do NOT rank these cells or claim one 'reproduces' a published value.")
    elif floor_cells:
        print(f"{len(floor_cells)} of {len(cells)} cells are at chance -- those are floor-pinned and")
        print("cannot be ranked against each other.")
    print("=" * 92)
    print()
    print("SCOPE OF THIS TABLE -- read before quoting any of it")
    print("-" * 92)
    print("WMDP-bio is the FORGET metric only. The gate is a pair: forget AND Retain-MMLU.")
    print("A checkpoint sitting at the WMDP-bio floor tells you nothing about whether it retained")
    print("benign capability, which is the whole point of a tamper-resistance claim. Do not")
    print("declare any cell a reproduction of Table 1 from this table alone -- the retain leg")
    print("is a separate measurement and is currently off by +3 to +5 pp under our lm-eval")
    print("reconstruction of the authors' ollmer-derived script.")
    print()
    print("Each cell is a SINGLE deterministic pass over a fixed question set. The interval shown")
    print("is binomial sampling error on that set. It is not a seed-variance estimate, and it does")
    print("not cover the paper's own sampling error, which we cannot quantify from the table.")
    print("-" * 92)
    return 0


if __name__ == "__main__":
    sys.exit(main())
