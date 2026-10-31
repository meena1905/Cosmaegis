import os
import sys
import csv
import time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "env"))
sys.path.append(os.path.join(HERE, "agents"))

from orbital_env import OrbitalEnv
from satellite import Satellite

VARIANTS = ["baseline", "risk_only", "uncertainty_only", "full"]
SEEDS = [0, 1, 2, 3, 4]
SEED0 = 20_000
MC_VARIANTS = ("uncertainty_only", "full")


class ShiftedEnv(OrbitalEnv):
    def __init__(self, threat_speed=(0.5, 1.5), threat_tca=(4.0, 10.0), **kw):
        super().__init__(**kw)
        self.threat_speed = threat_speed
        self.threat_tca = threat_tca

    def _make_threat(self, sat, max_tries=100):
        for _ in range(max_tries):
            T = np.random.uniform(*self.threat_tca)
            speed = np.random.uniform(*self.threat_speed)
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            v_d = speed * np.array([np.cos(ang), np.sin(ang)])
            w = v_d - sat.velocity
            wn = float(np.linalg.norm(w))
            if wn < 0.3:
                continue
            perp = np.array([-w[1], w[0]]) / wn
            b = np.random.uniform(0.0, self.threat_miss_max) * np.random.choice([-1.0, 1.0])
            target = sat.position + sat.velocity * T + perp * b
            pos = target - v_d * T
            if np.linalg.norm(pos - sat.position) >= self.min_spawn_dist:
                return Satellite(position=pos, velocity=v_d)
        return Satellite(position=sat.position + np.array([8.0, 0.0]),
                         velocity=np.array([-1.0, 0.0]))


CONDITIONS = {
    "normal":       dict(threat_speed=(0.5, 1.5), threat_tca=(4.0, 10.0)),
    "fast_debris":  dict(threat_speed=(1.5, 2.5), threat_tca=(4.0, 10.0)),
    "late_warning": dict(threat_speed=(1.5, 2.5), threat_tca=(2.5, 4.0), min_spawn_dist=3.0),
    "two_threats":  dict(threat_speed=(0.5, 1.5), threat_tca=(4.0, 10.0), num_threats=2),
}


def make_env(cond_kwargs):
    return ShiftedEnv(num_satellites=1, num_debris=3, **cond_kwargs)


def noop_policy(obs, env):
    return np.zeros(2)


def rule_policy(obs, env, safe_dist=2.5, horizon=8.0):
    sat_vel = obs[2:4]
    best_miss, best_dir = None, None
    for k in range(env.num_debris):
        base = 4 + 4 * k
        r = obs[base:base + 2]
        w = obs[base + 2:base + 4] - sat_vel
        denom = float(np.dot(w, w))
        t = 0.0 if denom < 1e-9 else float(np.clip(-np.dot(r, w) / denom, 0.0, horizon))
        m = r + w * t
        miss = float(np.linalg.norm(m))
        if miss < safe_dist and (best_miss is None or miss < best_miss):
            best_miss = miss
            best_dir = np.array([-w[1], w[0]]) if miss < 1e-6 else -m
    if best_dir is not None:
        n = np.linalg.norm(best_dir)
        if n > 1e-9:
            return best_dir / n
    return np.clip(-sat_vel, -1.0, 1.0)


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def tcrit(df):
    df = int(round(df))
    if df < 1:
        return float("nan")
    return T975.get(df, 1.96 if df > 10 else T975[1])


def finite(x):
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def mean_ci(x):
    x = finite(x)
    if len(x) == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    if len(x) < 2:
        return m, float("nan")
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return m, tcrit(len(x) - 1) * se


def diff_ci(a, b):
    a, b = finite(a), finite(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = float(np.sqrt(va + vb))
    if se == 0.0:
        return float(a.mean() - b.mean()), 0.0
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(a.mean() - b.mean()), tcrit(df) * se


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, label):
    label = np.asarray(label, dtype=bool)
    npos = int(label.sum())
    nneg = len(label) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def summarize_episodes(ep):
    return dict(collision_rate=100.0 * float(np.mean(ep["collided"])),
                near_miss_rate=100.0 * float(np.mean(ep["near"])),
                avg_fuel=float(np.mean(ep["fuel"])),
                avg_steps=float(np.mean(ep["steps"])))


def fmt(m, ci, nd=1):
    if not np.isfinite(m):
        return "n/a"
    if not np.isfinite(ci):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} +- {ci:.{nd}f}"


def summarize(rows, refs, n_episodes):
    metrics = [("collision_rate", "collisions %", 1), ("near_miss_rate", "near-miss %", 1),
               ("avg_fuel", "fuel", 2), ("avg_steps", "steps", 2)]

    def col(cond, variant, key):
        return [r[key] for r in rows if r["condition"] == cond and r["variant"] == variant]

    for cond in CONDITIONS:
        print("\n" + "=" * 78)
        print(f"CONDITION: {cond}   ({n_episodes} test episodes per model; "
              f"mean +- 95% CI across seeds)")
        print("=" * 78)
        print(f"{'agent':24s}" + "".join(f"{name:>14s}" for _, name, _ in metrics))
        for rname in ("do-nothing", "rule-based"):
            r = refs.get((cond, rname))
            if r:
                print(f"{rname:24s}" + "".join(f"{r[k]:14.{nd}f}" for k, _, nd in metrics))
        for v in VARIANTS:
            n_seeds = len(col(cond, v, "collision_rate"))
            if n_seeds == 0:
                continue
            cells = [fmt(*mean_ci(col(cond, v, k)), nd) for k, _, nd in metrics]
            print(f"{v + f' (n={n_seeds})':24s}" + "".join(f"{c:>14s}" for c in cells))

        print("\n  difference from baseline (variant - baseline); * = 95% CI excludes 0")
        for v in VARIANTS[1:]:
            parts = []
            for k, name, nd in metrics[:3]:
                d, ci = diff_ci(col(cond, v, k), col(cond, "baseline", k))
                star = "*" if (np.isfinite(d) and np.isfinite(ci) and abs(d) > ci) else " "
                parts.append(f"{name} {d:+.{nd}f} +- {ci:.{nd}f}{star}")
            print(f"  {v:18s} " + " | ".join(parts))

    print("\n" + "=" * 78)
    print("UNCERTAINTY CHECK (MC-Dropout variants; step-0 uncertainty, mean +- 95% CI over seeds)")
    print("=" * 78)
    print(f"{'agent / condition':32s}{'mean unc':>16s}{'spearman(unc,miss)':>22s}{'AUC(unc->collision)':>22s}")
    for v in MC_VARIANTS:
        for cond in CONDITIONS:
            u = fmt(*mean_ci(col(cond, v, "mean_unc0")), 3)
            s = fmt(*mean_ci(col(cond, v, "spearman_unc_miss")), 2)
            a = fmt(*mean_ci(col(cond, v, "auc_unc_collision")), 2)
            print(f"{v + ' / ' + cond:32s}{u:>16s}{s:>22s}{a:>22s}")
    print("\nHow to read it: if uncertainty is well calibrated it should be HIGHER in")
    print("the shifted conditions than in 'normal', the Spearman value should be")
    print("NEGATIVE (more uncertain when the predicted miss distance is smaller), and the")
    print("AUC should be clearly above 0.5. Values near 0 / 0.5 mean the uncertainty")
    print("carries no useful signal.")


# ---- torch part ----
import torch
import ppo_train as P


def run_episodes(env, act_fn, n_episodes, seed0):
    base = 4 + 4 * env.num_debris + 4 * (env.num_satellites - 1)
    ep = dict(collided=[], near=[], fuel=[], steps=[], unc0=[], miss0=[])
    for s in range(n_episodes):
        np.random.seed(seed0 + s)
        torch.manual_seed(seed0 + s)
        obs = env.reset()[0]
        ep["miss0"].append(min(float(obs[base + 4 * k + 3]) for k in range(env.num_threats)))
        done, hit, fuel, steps, first = False, False, 0.0, 0, True
        while not done:
            a, u = act_fn(obs)
            if first:
                ep["unc0"].append(u)
                first = False
            a = np.clip(a, -1.0, 1.0)
            fuel += float(np.linalg.norm(a))
            obs_list, _, done, info = env.step([a])
            obs = obs_list[0]
            steps += 1
            if info["collided"][0]:
                hit = True
                break
        ep["collided"].append(hit)
        ep["near"].append((not hit) and env.min_dist_seen < env.collision_threshold * 2.0)
        ep["fuel"].append(fuel)
        ep["steps"].append(steps)
    return ep


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    results_dir = os.path.join(HERE, "results")
    t0 = time.time()

    refs = {}
    for cname, kw in CONDITIONS.items():
        env = make_env(kw)
        for rname, fn in (("do-nothing", noop_policy), ("rule-based", rule_policy)):
            ep = run_episodes(env, lambda o, f=fn, e=env: (f(o, e), 0.0), n_episodes, SEED0)
            refs[(cname, rname)] = summarize_episodes(ep)
    print(f"reference policies done ({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for variant in VARIANTS:
        use_unc = variant in MC_VARIANTS
        for seed in SEEDS:
            path = os.path.join(results_dir, f"ppo_{variant}_s{seed}.pt")
            if not os.path.exists(path):
                print(f"[skip] missing {path}")
                continue
            actor, meta = P.load_model(path)
            act_fn = P.make_act_fn(actor, use_unc)
            for cname, kw in CONDITIONS.items():
                ep = run_episodes(make_env(kw), act_fn, n_episodes, SEED0)
                row = dict(variant=variant, seed=seed, condition=cname, **summarize_episodes(ep))
                if use_unc:
                    row["mean_unc0"] = float(np.mean(ep["unc0"]))
                    row["spearman_unc_miss"] = spearman(ep["unc0"], ep["miss0"])
                    row["auc_unc_collision"] = auc(ep["unc0"], ep["collided"])
                else:
                    row["mean_unc0"] = row["spearman_unc_miss"] = row["auc_unc_collision"] = float("nan")
                rows.append(row)
            print(f"evaluated {variant} s{seed}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "eval_per_model.csv")
    fields = ["variant", "seed", "condition", "collision_rate", "near_miss_rate", "avg_fuel",
              "avg_steps", "mean_unc0", "spearman_unc_miss", "auc_unc_collision"]
    with open(out_csv, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    print(f"\nSaved per-model results to {out_csv}")

    summarize(rows, refs, n_episodes)


if __name__ == "__main__":
    main()