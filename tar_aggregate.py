import csv
import glob
import os

BIO_FORGET = {
    "college_biology", "high_school_biology", "medical_genetics", "virology",
    "college_medicine", "anatomy", "clinical_knowledge",
}
DATA = "/scratch/USER/repos/tamper-resistance/red_teaming/mmlu_eval/data/test"

acc_path = "/scratch/USER/repos/mmlu_jrp02_tarv2_authors/mmlu_accs.csv"
accs = {}
for r in csv.reader(open(acc_path)):
    if len(r) >= 2 and r[0].strip() and r[0].strip() != "subject":
        accs[r[0].strip()] = float(r[1])

counts = {}
for f in glob.glob(os.path.join(DATA, "*_test.csv")):
    subj = os.path.basename(f)[:-len("_test.csv")]
    with open(f) as fh:
        rows = [r for r in csv.reader(fh) if r]
    counts[subj] = max(0, len(rows) - 1)

all57 = sorted(accs)
keep50 = [s for s in all57 if s not in BIO_FORGET]
n57 = sum(counts[s] for s in all57)
n50 = sum(counts[s] for s in keep50)
macro50 = 100 * sum(accs[s] for s in keep50) / len(keep50)
macro57 = 100 * sum(accs[s] for s in all57) / len(all57)
micro50 = 100 * sum(accs[s] * counts[s] for s in keep50) / n50
micro57 = 100 * sum(accs[s] * counts[s] for s in all57) / n57

print(f"n57={n57} (expect 14042)  n50={n50} (expect 12749)")
print(f"macro57={macro57:.2f}  macro50={macro50:.2f}")
print(f"micro57={micro57:.2f}  micro50={micro50:.2f}")
# SEs at binomial p~0.57
for label, val, n in [("macro50", macro50, n50), ("micro50", micro50, n50), ("micro57", micro57, n57)]:
    se = 100 * (0.57 * 0.43 / n) ** 0.5
    print(f"{label}: {val:.2f}, SE={se:.2f}, delta vs 54.7 = {val - 54.7:+.2f} = {(val - 54.7) / se:+.1f} SE")
