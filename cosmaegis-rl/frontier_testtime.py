import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

SCALES = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
CONDS = ["normal", "fast_debris", "late_warning", "two_threats"]
VARIANTS = ["baseline", "risk_only", "uncertainty_only", "full"]
SEEDS = [0, 1, 2, 3, 4]

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
        return float("nan"), float("nan"), 0
    if len(x) < 2:
        return float(x.mean()), float("nan"), len(x)
    return float(x.mean()), tcrit(len(x) - 1) * float(x.std(ddof=1) / np.sqrt(len(x))), len(x)


def diff_ci(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    if se == 0.0:
        return float(a.mean() - b.mean()), 0.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(a.mean() - b.mean()), tcrit(df) * se


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
    f = np.array([p[0] for p in points])
    c = np.array([p[1] for p in points])
    order = np.argsort(f)
    f, c = f[order], c[order]
    if target < f[0] - tol or target > f[-1] + tol:
        return float("nan")
    return float(np.interp(target, f, c))


def fmt(m, ci, n, nd=1):
    if not np.isfinite(m):
        return "n/a"
    ci_s = f" +- {ci:.{nd}f}" if np.isfinite(ci) else ""
    return f"{m:.{nd}f}{ci_s} (n={n})"


def analyse(rows, mc_points):
    for cond in ("normal",):
        c0 = [r for r in rows if r["variant"] == "baseline" and r["condition"] == cond and r["scale"] == 1.0]
        if c0:
            print(f"\nsanity: baseline at scale 1.0, '{cond}': collisions "
                  f"{np.mean([r['collision_rate'] for r in c0]):.1f}%, fuel "
                  f"{np.mean([r['avg_fuel'] for r in c0]):.2f}   (earlier table: 2.0%, 2.84)")

    for mult in (1.0, 0.85):
        print("\n" + "=" * 92)
        label = "the SAME fuel as the plain baseline" if mult == 1.0 else "15% LESS fuel than the plain baseline"
        print(f"SECTION 1: collisions % at {label}")
        print("(each model's own scale curve, interpolated; mean +- 95% CI across seeds)")
        print("=" * 92)
        print(f"{'condition':14s}{'target fuel':>12s}" + "".join(f"{v:>26s}" for v in VARIANTS))
        for cond in CONDS:
            base_fuel = np.mean([curve(rows, "baseline", s, cond)[0][0]
                                 for s in SEEDS if curve(rows, "baseline", s, cond)])
            target = mult * base_fuel
            per_var = {}
            for v in VARIANTS:
                per_var[v] = [coll_at_fuel(curve(rows, v, s, cond), target) for s in SEEDS]
            cells = [fmt(*mean_ci(per_var[v])) for v in VARIANTS]
            print(f"{cond:14s}{target:12.2f}" + "".join(f"{c:>26s}" for c in cells))
            diffs = []
            for v in VARIANTS[1:]:
                d, ci = diff_ci(per_var[v], per_var["baseline"])
                star = "*" if (np.isfinite(d) and np.isfinite(ci) and abs(d) > ci) else " "
                diffs.append(f"{v} {d:+.1f}+-{ci:.1f}{star}")
            print(f"{'':26s}vs baseline: " + " | ".join(diffs))

    print("\n" + "=" * 92)
    print("SECTION 2: the uncertainty (mc_caution) rule vs shrinking the SAME model's thrust, at equal fuel")
    print("difference in collisions % = (rule) - (plain thrust-scaling at the rule's fuel);")
    print("POSITIVE = the rule is worse than simple shrinking.  mean +- 95% CI across seeds; * = CI excludes 0")
    print("=" * 92)
    print(f"{'variant / condition':34s}{'rule coll %':>13s}{'shrink coll %':>15s}{'difference':>20s}")
    for v in VARIANTS:
        for cond in CONDS:
            rule, shrink, diff = [], [], []
            for s in SEEDS:
                pt = mc_points.get((v, s, cond))
                if pt is None:
                    continue
                f_m, c_m = pt
                c_s = coll_at_fuel(curve(rows, v, s, cond), f_m)
                if np.isfinite(c_s):
                    rule.append(c_m); shrink.append(c_s); diff.append(c_m - c_s)
            m, ci, n = mean_ci(diff)
            star = "*" if (np.isfinite(m) and np.isfinite(ci) and abs(m) > ci) else " "
            print(f"{v + ' / ' + cond:34s}{np.mean(rule) if rule else float('nan'):13.1f}"
                  f"{np.mean(shrink) if shrink else float('nan'):15.1f}"
                  f"{('%+.1f+-%.1f%s (n=%d)' % (m, ci, star, n)) if np.isfinite(m) else 'n/a':>20s}")


# ---- torch part ----
import torch
import evaluate_all as E
import ppo_train as P


def make_scaled_fn(actor, s):
    @torch.no_grad()
    def act(obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        m1, m2 = actor.ones_masks(1)
        mean, _ = actor.mean_and_unc(x, m1, m2)
        return (mean * s)[0].numpy(), 0.0
    return act


def load_mc_points(path):
    pts = {}
    if not os.path.exists(path):
        return pts
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r["mode"] == "mc_caution":
                pts[(r["variant"], int(r["seed"]), r["condition"])] = (
                    float(r["avg_fuel"]), float(r["collision_rate"]))
    return pts


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    results_dir = os.path.join(HERE, "results")
    t0 = time.time()
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            path = os.path.join(results_dir, f"ppo_{variant}_s{seed}.pt")
            if not os.path.exists(path):
                print(f"[skip] missing {path}")
                continue
            actor, meta = P.load_model(path)
            for scale in SCALES:
                act_fn = make_scaled_fn(actor, scale)
                for cname, kw in E.CONDITIONS.items():
                    ep = E.run_episodes(E.make_env(kw), act_fn, n_episodes, E.SEED0)
                    s = E.summarize_episodes(ep)
                    rows.append(dict(variant=variant, seed=seed, condition=cname, scale=scale,
                                     collision_rate=s["collision_rate"], avg_fuel=s["avg_fuel"],
                                     avg_steps=s["avg_steps"]))
            print(f"done {variant} s{seed}  ({time.time() - t0:.0f}s)", flush=True)

    out_csv = os.path.join(results_dir, "frontier_testtime.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["variant", "seed", "condition", "scale",
                                           "collision_rate", "avg_fuel", "avg_steps"])
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\nSaved {out_csv}")
    analyse(rows, load_mc_points(os.path.join(results_dir, "ablation_testtime.csv")))


if __name__ == "__main__":
    main()