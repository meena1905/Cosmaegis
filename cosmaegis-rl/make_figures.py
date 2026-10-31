import os
import re
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "figures")

VARIANTS = ["baseline", "risk_only", "uncertainty_only", "full", "mc_nocaution"]
LABEL = {"baseline": "Baseline PPO",
         "risk_only": "CVaR-weighted",
         "uncertainty_only": "MC-Dropout",
         "full": "MC-Dropout + CVaR",
         "mc_nocaution": "MC-Dropout (no caution in training)"}
COLOR = {"baseline": "#444444", "risk_only": "#999999", "uncertainty_only": "#1f77b4",
         "full": "#d62728", "mc_nocaution": "#2ca02c"}
CONDS = ["normal", "fast_debris", "late_warning", "two_threats"]
COND_LABEL = {"normal": "Normal", "fast_debris": "Fast debris",
              "late_warning": "Late warning", "two_threats": "Two threats"}

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
        return float(x.mean()), 0.0
    return float(x.mean()), tcrit(len(x) - 1) * float(x.std(ddof=1) / np.sqrt(len(x)))


def read_csv(path, int_cols=(), float_cols=()):
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            for k in int_cols:
                r[k] = int(r[k])
            for k in float_cols:
                r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
            rows.append(r)
    return rows


def curve(rows, variant, seed, cond):
    pts = {r["scale"]: (r["avg_fuel"], r["collision_rate"]) for r in rows
           if r["variant"] == variant and r["seed"] == seed and r["condition"] == cond}
    out, prev = [], float("inf")
    for s in sorted(pts, reverse=True):
        f, c = pts[s]
        if f > prev + 1e-9:
            break
        out.append((f, c))
        prev = f
    return out


def coll_at_fuel(points, target, tol=0.15):
    if len(points) < 2:
        return float("nan")
    f = np.array([p[0] for p in points]); c = np.array([p[1] for p in points])
    order = np.argsort(f); f, c = f[order], c[order]
    if target < f[0] - tol or target > f[-1] + tol:
        return float("nan")
    return float(np.interp(target, f, c))


def seeds_of(rows, variant):
    return sorted({r["seed"] for r in rows if r["variant"] == variant})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def load_frontier():
    a = read_csv(os.path.join(RES, "frontier_testtime.csv"), ("seed",),
                 ("scale", "collision_rate", "avg_fuel", "avg_steps"))
    b = read_csv(os.path.join(RES, "frontier_mc_nocaution.csv"), ("seed",),
                 ("scale", "collision_rate", "avg_fuel", "avg_steps"))
    if a is None:
        return None
    return a + (b or [])


def fig_tradeoff():
    rows = load_frontier()
    if rows is None:
        print("[skip] trade-off figure: results/frontier_testtime.csv not found")
        return
    abl = read_csv(os.path.join(RES, "ablation_testtime.csv"), ("seed",),
                   ("collision_rate", "near_miss_rate", "avg_fuel", "avg_steps"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for ax, cond in zip(axes.ravel(), CONDS):
        for v in VARIANTS:
            sd = seeds_of(rows, v)
            if not sd:
                continue
            scales = sorted({r["scale"] for r in rows if r["variant"] == v}, reverse=True)
            xs, ys, es = [], [], []
            for sc in scales:
                sel = [r for r in rows if r["variant"] == v and r["condition"] == cond and r["scale"] == sc]
                if not sel:
                    continue
                xs.append(np.mean([r["avg_fuel"] for r in sel]))
                m, ci = mean_ci([r["collision_rate"] for r in sel])
                ys.append(m); es.append(ci)
            ax.errorbar(xs, ys, yerr=es, color=COLOR[v], marker="o", ms=3.5, lw=1.4,
                        capsize=2, label=LABEL[v], alpha=0.9)
        if abl:
            for v in ("baseline", "uncertainty_only", "full"):
                sel = [r for r in abl if r["variant"] == v and r["condition"] == cond and r["mode"] == "mc_caution"]
                if sel:
                    ax.scatter([np.mean([r["avg_fuel"] for r in sel])],
                               [np.mean([r["collision_rate"] for r in sel])],
                               marker="*", s=130, color=COLOR[v], edgecolor="black", zorder=5,
                               label="uncertainty rule at test time" if v == "baseline" else None)
        ax.set_title(COND_LABEL[cond])
        ax.set_xlabel("average fuel per episode")
        ax.set_ylabel("collisions (%)")
        ax.grid(alpha=0.3)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Safety-fuel trade-off (lines: thrust scaled from 1.0 to 0.4; error bars: 95% CI over seeds;\n"
                 "stars: test-time uncertainty rule). Lower-left is better.", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    save(fig, "fig1_tradeoff.png")


def fig_equal_fuel():
    rows = load_frontier()
    if rows is None:
        print("[skip] equal-fuel figure: results/frontier_testtime.csv not found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, mult, title in zip(axes, (1.0, 0.85), ("same fuel as plain baseline", "15% less fuel than plain baseline")):
        vs = [v for v in VARIANTS if seeds_of(rows, v)]
        width = 0.8 / len(vs)
        for i, v in enumerate(vs):
            means, cis = [], []
            for cond in CONDS:
                base_f = [curve(rows, "baseline", s, cond)[0][0]
                          for s in seeds_of(rows, "baseline") if curve(rows, "baseline", s, cond)]
                target = mult * float(np.mean(base_f))
                vals = [coll_at_fuel(curve(rows, v, s, cond), target) for s in seeds_of(rows, v)]
                m, ci = mean_ci(vals)
                means.append(m); cis.append(ci)
            ax.bar(np.arange(len(CONDS)) + i * width - 0.4 + width / 2, means, width,
                   yerr=cis, capsize=2, color=COLOR[v], label=LABEL[v])
        ax.set_xticks(np.arange(len(CONDS)))
        ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("collisions (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Collisions at equal fuel (mean and 95% CI over seeds)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig2_equal_fuel.png")


LOG_RE = re.compile(r"\[(\w+) s(\d+)\] rollout (\d+)/\d+ \| steps (\d+) \| train collisions \(last 300 ep\):\s+([\d.]+)%")


def fig_training():
    data = {}
    for path in glob.glob(os.path.join(RES, "logs", "*.txt")):
        for line in open(path, errors="ignore"):
            m = LOG_RE.search(line)
            if m:
                v, seed, _, steps, coll = m.groups()
                data.setdefault(v, {}).setdefault(int(seed), []).append((int(steps), float(coll)))
    if not data:
        print("[skip] training figure: no logs found in results/logs")
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for v in VARIANTS:
        if v not in data:
            continue
        seeds = data[v]
        steps = sorted({s for pts in seeds.values() for s, _ in pts})
        mean, se = [], []
        for st in steps:
            vals = [dict(pts)[st] for pts in seeds.values() if st in dict(pts)]
            mean.append(np.mean(vals))
            se.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        mean, se = np.array(mean), np.array(se)
        ax.plot(steps, mean, color=COLOR[v], label=f"{LABEL[v]} (n={len(seeds)})")
        ax.fill_between(steps, mean - se, mean + se, color=COLOR[v], alpha=0.15)
    ax.set_xlabel("training steps")
    ax.set_ylabel("collisions in training episodes (%)")
    ax.set_title("Training curves (mean +- s.e. over seeds; stochastic training policy)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig3_training.png")


def fig_rule_vs_shrink():
    rows = load_frontier()
    abl = read_csv(os.path.join(RES, "ablation_testtime.csv"), ("seed",),
                   ("collision_rate", "near_miss_rate", "avg_fuel", "avg_steps"))
    if rows is None or abl is None:
        print("[skip] rule-vs-shrink figure: needs frontier_testtime.csv and ablation_testtime.csv")
        return
    vs = ["baseline", "risk_only", "uncertainty_only", "full"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    width = 0.8 / len(vs)
    for i, v in enumerate(vs):
        means, cis = [], []
        for cond in CONDS:
            diffs = []
            for s in seeds_of(rows, v):
                pt = [r for r in abl if r["variant"] == v and r["seed"] == s
                      and r["condition"] == cond and r["mode"] == "mc_caution"]
                if not pt:
                    continue
                c_s = coll_at_fuel(curve(rows, v, s, cond), pt[0]["avg_fuel"])
                if np.isfinite(c_s):
                    diffs.append(pt[0]["collision_rate"] - c_s)
            m, ci = mean_ci(diffs)
            means.append(m); cis.append(ci)
        ax.bar(np.arange(len(CONDS)) + i * width - 0.4 + width / 2, means, width,
               yerr=cis, capsize=2, color=COLOR[v], label=LABEL[v])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(CONDS)))
    ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
    ax.set_ylabel("extra collisions (percentage points)")
    ax.set_title("Uncertainty rule minus plain thrust scaling, at equal fuel\n"
                 "(positive = the rule is worse; 95% CI over seeds)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, "fig4_rule_vs_shrink.png")


def fig_uncertainty():
    rows = read_csv(os.path.join(RES, "eval_per_model.csv"), ("seed",),
                    ("collision_rate", "near_miss_rate", "avg_fuel", "avg_steps",
                     "mean_unc0", "spearman_unc_miss", "auc_unc_collision"))
    if rows is None:
        print("[skip] uncertainty figure: results/eval_per_model.csv not found")
        return
    vs = ["uncertainty_only", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title, ref in ((axes[0], "mean_unc0", "mean step-0 uncertainty", None),
                                (axes[1], "auc_unc_collision", "AUC: uncertainty predicting a later collision", 0.5)):
        width = 0.8 / len(vs)
        for i, v in enumerate(vs):
            means, cis = [], []
            for cond in CONDS:
                vals = [r[key] for r in rows if r["variant"] == v and r["condition"] == cond]
                m, ci = mean_ci(vals)
                means.append(m); cis.append(ci)
            ax.bar(np.arange(len(CONDS)) + i * width - 0.4 + width / 2, means, width,
                   yerr=cis, capsize=2, color=COLOR[v], label=LABEL[v])
        if ref is not None:
            ax.axhline(ref, color="black", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(CONDS)))
        ax.set_xticklabels([COND_LABEL[c] for c in CONDS])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Uncertainty diagnostics (mean and 95% CI over seeds; dashed line = no signal)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig5_uncertainty.png")


def table_main():
    rows = read_csv(os.path.join(RES, "eval_per_model.csv"), ("seed",),
                    ("collision_rate", "near_miss_rate", "avg_fuel", "avg_steps",
                     "mean_unc0", "spearman_unc_miss", "auc_unc_collision"))
    if rows is None:
        print("[skip] table: results/eval_per_model.csv not found")
        return
    lines = []
    for cond in CONDS:
        lines.append(f"\n**{COND_LABEL[cond]}** (mean +- 95% CI over seeds)\n")
        lines.append("| Agent | Collisions (%) | Near-misses (%) | Fuel | Steps |")
        lines.append("|---|---|---|---|---|")
        for v in ("baseline", "risk_only", "uncertainty_only", "full"):
            sel = [r for r in rows if r["variant"] == v and r["condition"] == cond]
            if not sel:
                continue
            cells = []
            for k, nd in (("collision_rate", 1), ("near_miss_rate", 1), ("avg_fuel", 2), ("avg_steps", 2)):
                m, ci = mean_ci([r[k] for r in sel])
                cells.append(f"{m:.{nd}f} +- {ci:.{nd}f}")
            lines.append(f"| {LABEL[v]} (n={len(sel)}) | " + " | ".join(cells) + " |")
    text = "\n".join(lines)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "table_main.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"saved {path}")


def main():
    for fn in (fig_tradeoff, fig_equal_fuel, fig_training, fig_rule_vs_shrink, fig_uncertainty, table_main):
        try:
            fn()
        except Exception as e:
            print(f"[error] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()