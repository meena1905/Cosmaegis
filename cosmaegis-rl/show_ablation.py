import os
import sys
import csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = ["baseline", "risk_only", "uncertainty_only", "full"]
MODES = ["plain", "const_0.75", "mc_mean", "mc_caution"]
CONDS = ["normal", "fast_debris", "late_warning", "two_threats"]
METRICS = [("collision_rate", "collisions %", 1), ("near_miss_rate", "near-miss %", 1),
           ("avg_fuel", "fuel", 2), ("avg_steps", "steps", 2)]

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def tcrit(df):
    df = int(round(df))
    if df < 1:
        return float("nan")
    return T975.get(df, 1.96 if df > 10 else T975[1])


def mean_ci(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) < 2:
        return float(x.mean()), float("nan")
    return float(x.mean()), tcrit(len(x) - 1) * float(x.std(ddof=1) / np.sqrt(len(x)))


def diff_ci(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    if se == 0.0:
        return float(a.mean() - b.mean()), 0.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(a.mean() - b.mean()), tcrit(df) * se


def fmt(m, ci, nd=1):
    if not np.isfinite(m):
        return "n/a"
    if not np.isfinite(ci):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} +- {ci:.{nd}f}"


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in ("collision_rate", "near_miss_rate", "avg_fuel", "avg_steps"):
                r[k] = float(r[k])
            r["seed"] = int(r["seed"])
            rows.append(r)
    return rows


def vals(rows, variant, cond, mode, key):
    return [r[key] for r in rows
            if r["variant"] == variant and r["condition"] == cond and r["mode"] == mode]


def show(rows):
    for cond in CONDS:
        print("\n" + "=" * 86)
        print(f"CONDITION: {cond}   (mean +- 95% CI across seeds)")
        print("=" * 86)
        print(f"{'variant / test-time mode':30s}" + "".join(f"{n:>14s}" for _, n, _ in METRICS))
        for v in VARIANTS:
            for m in MODES:
                cells = [fmt(*mean_ci(vals(rows, v, cond, m, k)), nd) for k, _, nd in METRICS]
                print(f"{v + ' / ' + m:30s}" + "".join(f"{c:>14s}" for c in cells))

    for mode in ("plain", "mc_mean"):
        print("\n" + "=" * 86)
        print(f"TRAINING EFFECT, both evaluated with test-time mode '{mode}':  variant - baseline")
        print("(unpaired across seeds, Welch 95% CI; * = CI excludes 0)")
        print("=" * 86)
        for v in VARIANTS[1:]:
            for cond in CONDS:
                parts = []
                for k, name, nd in METRICS[:3]:
                    d, ci = diff_ci(vals(rows, v, cond, mode, k), vals(rows, "baseline", cond, mode, k))
                    star = "*" if (np.isfinite(d) and np.isfinite(ci) and abs(d) > ci) else " "
                    parts.append(f"{name} {d:+.{nd}f}+-{ci:.{nd}f}{star}")
                print(f"{v:17s} {cond:13s} " + " | ".join(parts))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "ablation_testtime.csv")
    if not os.path.exists(path):
        print(f"File not found: {path}\nRun ablate_testtime.py first.")
        return
    show(load(path))


if __name__ == "__main__":
    main()