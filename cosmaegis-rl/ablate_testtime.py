import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

MODES = ["plain", "const_0.75", "mc_mean", "mc_caution"]
CONST = 0.75
KEYS = [("collision_rate", "collisions %", 1), ("near_miss_rate", "near-miss %", 1),
        ("avg_fuel", "fuel", 2)]


def _by_seed(rows, variant, cond, mode, key):
    return {r["seed"]: r[key] for r in rows
            if r["variant"] == variant and r["condition"] == cond and r["mode"] == mode}


def paired(rows, variant, cond, mode_a, mode_b, key):
    da = _by_seed(rows, variant, cond, mode_a, key)
    db = _by_seed(rows, variant, cond, mode_b, key)
    return [da[s] - db[s] for s in sorted(set(da) & set(db))]


def summarize_ablation(rows, E, n_episodes):
    print("\n" + "=" * 86)
    print(f"ABSOLUTE NUMBERS, condition 'normal' ({n_episodes} episodes; mean +- 95% CI over seeds)")
    print("(check: baseline/plain and uncertainty_only/mc_caution must equal the earlier table)")
    print("=" * 86)
    print(f"{'variant / test-time mode':34s}" + "".join(f"{n:>15s}" for _, n, _ in KEYS))
    for v in E.VARIANTS:
        for m in MODES:
            cells = []
            for k, _, nd in KEYS:
                vals = list(_by_seed(rows, v, "normal", m, k).values())
                cells.append(E.fmt(*E.mean_ci(vals), nd))
            print(f"{v + ' / ' + m:34s}" + "".join(f"{c:>15s}" for c in cells))

    comparisons = [
        ("A. only shrink thrust by a constant 0.75   (const_0.75 - plain)", "const_0.75", "plain"),
        ("B. only average 15 dropout passes          (mc_mean - plain)", "mc_mean", "plain"),
        ("C. the full uncertainty rule               (mc_caution - plain)", "mc_caution", "plain"),
        ("D. uncertainty rule vs constant shrink     (mc_caution - const_0.75)", "mc_caution", "const_0.75"),
    ]
    for title, a, b in comparisons:
        print("\n" + "=" * 86)
        print(title)
        print("paired over seeds (same model, same episodes); * = 95% CI excludes 0")
        print("=" * 86)
        print(f"{'variant / condition':34s}" + "".join(f"{n:>17s}" for _, n, _ in KEYS))
        for v in E.VARIANTS:
            for cond in E.CONDITIONS:
                cells = []
                for k, _, nd in KEYS:
                    d = paired(rows, v, cond, a, b, k)
                    m, ci = E.mean_ci(d)
                    star = "*" if (np.isfinite(m) and np.isfinite(ci) and abs(m) > ci) else " "
                    cells.append(f"{m:+.{nd}f}+-{ci:.{nd}f}{star}")
                print(f"{v + ' / ' + cond:34s}" + "".join(f"{c:>17s}" for c in cells))


# ---- torch part ----
import torch
import evaluate_all as E
import ppo_train as P


def make_mode_fn(actor, mode, n_mc=15):
    @torch.no_grad()
    def act(obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if mode in ("plain", "const_0.75"):
            m1, m2 = actor.ones_masks(1)
        else:
            m1, m2 = actor.sample_masks(1, n_mc)
        mean, unc = actor.mean_and_unc(x, m1, m2)
        if mode == "const_0.75":
            mean = mean * CONST
        elif mode == "mc_caution":
            mean = mean * (1.0 - unc).clamp(min=0.3).unsqueeze(-1)
        return mean[0].numpy(), float(unc[0])
    return act


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    results_dir = os.path.join(HERE, "results")
    t0 = time.time()
    rows = []
    for variant in E.VARIANTS:
        for seed in E.SEEDS:
            path = os.path.join(results_dir, f"ppo_{variant}_s{seed}.pt")
            if not os.path.exists(path):
                print(f"[skip] missing {path}")
                continue
            actor, meta = P.load_model(path)
            for mode in MODES:
                act_fn = make_mode_fn(actor, mode)
                for cname, kw in E.CONDITIONS.items():
                    ep = E.run_episodes(E.make_env(kw), act_fn, n_episodes, E.SEED0)
                    rows.append(dict(variant=variant, seed=seed, condition=cname, mode=mode,
                                     **E.summarize_episodes(ep)))
            print(f"done {variant} s{seed}  ({time.time() - t0:.0f}s)", flush=True)

    out_csv = os.path.join(results_dir, "ablation_testtime.csv")
    fields = ["variant", "seed", "condition", "mode", "collision_rate", "near_miss_rate",
              "avg_fuel", "avg_steps"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\nSaved {out_csv}")
    summarize_ablation(rows, E, n_episodes)


if __name__ == "__main__":
    main()