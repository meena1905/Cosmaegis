import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

NEW = "mc_nocaution"
COMPARE = ["baseline", "uncertainty_only", "full", NEW]


def load_old_rows(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            r["seed"] = int(r["seed"])
            r["scale"] = float(r["scale"])
            for k in ("collision_rate", "avg_fuel", "avg_steps"):
                r[k] = float(r[k])
            rows.append(r)
    return rows


def analyse_extra(rows, F):
    for mult in (1.0, 0.85):
        label = "the SAME fuel as the plain baseline" if mult == 1.0 else "15% LESS fuel than the plain baseline"
        print("\n" + "=" * 100)
        print(f"collisions % at {label}")
        print("(each model's own thrust-scale curve, interpolated; mean +- 95% CI across seeds)")
        print("=" * 100)
        print(f"{'condition':14s}{'target fuel':>12s}" + "".join(f"{v:>22s}" for v in COMPARE))
        for cond in F.CONDS:
            base_fuel = np.mean([F.curve(rows, "baseline", s, cond)[0][0]
                                 for s in F.SEEDS if F.curve(rows, "baseline", s, cond)])
            target = mult * base_fuel
            per = {v: [F.coll_at_fuel(F.curve(rows, v, s, cond), target) for s in F.SEEDS]
                   for v in COMPARE}
            cells = [F.fmt(*F.mean_ci(per[v])) for v in COMPARE]
            print(f"{cond:14s}{target:12.2f}" + "".join(f"{c:>22s}" for c in cells))
            diffs = []
            for ref in ("baseline", "uncertainty_only", "full"):
                d, ci = F.diff_ci(per[NEW], per[ref])
                star = "*" if (np.isfinite(d) and np.isfinite(ci) and abs(d) > ci) else " "
                diffs.append(f"{NEW} - {ref}: {d:+.1f}+-{ci:.1f}{star}")
            print(f"{'':26s}" + " | ".join(diffs))
    print("\n* = 95% CI of the difference excludes 0 (negative = mc_nocaution has FEWER collisions)")


# ---- torch part ----
import frontier_testtime as F


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    results_dir = os.path.join(HERE, "results")
    old_path = os.path.join(results_dir, "frontier_testtime.csv")
    if not os.path.exists(old_path):
        print(f"Missing {old_path}. Run frontier_testtime.py first.")
        return

    t0 = time.time()
    rows_new = []
    for seed in F.SEEDS:
        path = os.path.join(results_dir, f"ppo_{NEW}_s{seed}.pt")
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        actor, meta = F.P.load_model(path)
        for scale in F.SCALES:
            act_fn = F.make_scaled_fn(actor, scale)
            for cname, kw in F.E.CONDITIONS.items():
                ep = F.E.run_episodes(F.E.make_env(kw), act_fn, n_episodes, F.E.SEED0)
                s = F.E.summarize_episodes(ep)
                rows_new.append(dict(variant=NEW, seed=seed, condition=cname, scale=scale,
                                     collision_rate=s["collision_rate"], avg_fuel=s["avg_fuel"],
                                     avg_steps=s["avg_steps"]))
        print(f"done {NEW} s{seed}  ({time.time() - t0:.0f}s)", flush=True)

    out_csv = os.path.join(results_dir, "frontier_mc_nocaution.csv")
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["variant", "seed", "condition", "scale",
                                           "collision_rate", "avg_fuel", "avg_steps"])
        wr.writeheader()
        for r in rows_new:
            wr.writerow(r)
    print(f"\nSaved {out_csv}")

    rows = load_old_rows(old_path) + rows_new
    c0 = [r for r in rows_new if r["condition"] == "normal" and r["scale"] == 1.0]
    if c0:
        print(f"\nmc_nocaution at scale 1.0, 'normal': collisions "
              f"{np.mean([r['collision_rate'] for r in c0]):.1f}%, fuel "
              f"{np.mean([r['avg_fuel'] for r in c0]):.2f}")
    analyse_extra(rows, F)


if __name__ == "__main__":
    main()