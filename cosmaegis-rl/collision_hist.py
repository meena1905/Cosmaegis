"""
collision_hist.py

For each trained agent, runs held-out test episodes and records the
STEP at which collisions happen. If most collisions occur in the first
few steps, they are mostly unavoidable spawn situations rather than
policy mistakes, and collision rate is a weak measure of policy quality.

Run from the project root (same folder as evaluate.py):
    python collision_hist.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import load_model, run_eval_episode   # also adds env/ and agents/ to sys.path
from orbital_env import OrbitalEnv

N_EPISODES = 500
EARLY_STEPS = 5

AGENTS = {
    "baseline":         dict(path="baseline_model.pt",         use_uncertainty=False),
    "risk_only":        dict(path="risk_only_model.pt",        use_uncertainty=False),
    "uncertainty_only": dict(path="uncertainty_only_model.pt", use_uncertainty=True),
    "full":             dict(path="full_model.pt",             use_uncertainty=True),
}


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    env = OrbitalEnv(num_satellites=1, num_debris=3)

    all_steps = {}
    for name, cfg in AGENTS.items():
        path = os.path.join(results_dir, cfg["path"])
        if not os.path.exists(path):
            print(f"[skip] {name}: {path} not found")
            continue
        net = load_model(path, env.obs_dim())
        steps = []
        for s in range(N_EPISODES):
            r = run_eval_episode(env, net, cfg["use_uncertainty"], seed=s)
            if r["collided"]:
                steps.append(r["steps"])
        all_steps[name] = steps

        n = len(steps)
        if n == 0:
            print(f"{name:17s}: 0 collisions in {N_EPISODES} episodes")
            continue
        early = np.mean(np.array(steps) <= EARLY_STEPS)
        print(f"{name:17s}: {n:3d} collisions ({100 * n / N_EPISODES:.1f}%) | "
              f"median step {np.median(steps):.0f} | "
              f"{100 * early:.0f}% within first {EARLY_STEPS} steps")

    if not all_steps:
        return

    fig, axes = plt.subplots(1, len(all_steps), figsize=(4 * len(all_steps), 3.5),
                             sharey=True, squeeze=False)
    for ax, (name, steps) in zip(axes[0], all_steps.items()):
        ax.hist(steps, bins=range(1, 31), color="#4c72b0", edgecolor="white")
        ax.set_title(name)
        ax.set_xlabel("collision step")
    axes[0][0].set_ylabel("# collisions")
    plt.tight_layout()
    out = os.path.join(results_dir, "collision_steps.png")
    plt.savefig(out, dpi=150)
    print(f"\nSaved histogram to {out}")


if __name__ == "__main__":
    main()