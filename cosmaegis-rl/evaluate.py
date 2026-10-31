"""
evaluate.py

Loads trained agent(s) and runs them through a fixed set of held-out
test episodes (same random seeds across all agents, for a fair
comparison) to produce the metrics table for the paper:

  - collision rate       (% of episodes ending in collision)
  - near-miss rate        (% of episodes with a close call but no collision)
  - avg episode length    (proxy for "survived how long")
  - avg fuel used          (sum of |action| across the episode)

Run baseline_agent.py and bayesian_agent.py (all variants) first to
produce the .pt model files this script loads.
"""

import os
import sys
import csv
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from orbital_env import OrbitalEnv
from policy import PolicyNet, mc_dropout_predict

NEAR_MISS_MULTIPLIER = 2.0  # distance < threshold*2 but no collision = near miss


def load_model(path, obs_dim):
    net = PolicyNet(obs_dim=obs_dim)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    return net


def run_eval_episode(env, net, use_uncertainty, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    obs_list = env.reset()
    done = False
    min_dist_seen = np.inf
    fuel_used = 0.0
    steps = 0

    while not done:
        actions = []
        for i in range(env.num_satellites):
            obs_t = torch.tensor(obs_list[i], dtype=torch.float32).unsqueeze(0)
            if use_uncertainty:
                mean_action, uncertainty = mc_dropout_predict(net, obs_t, n_samples=15)
                mean_action = mean_action.squeeze(0)
                caution_scale = max(0.3, 1.0 - min(uncertainty.item(), 1.0))
                action = (mean_action * caution_scale).detach().numpy()
            else:
                net.eval()
                with torch.no_grad():
                    action = net(obs_t).squeeze(0).numpy()
            action = np.clip(action, -1.0, 1.0)
            actions.append(action)
            fuel_used += np.linalg.norm(action)

        obs_list, rewards, done, info = env.step(actions)
        steps += 1

        # track closest approach this episode (for near-miss stats)
        for sat in env.satellites:
            for d in env.debris:
                dist = np.linalg.norm(sat.position - d.position)
                min_dist_seen = min(min_dist_seen, dist)

        collided = any(info["collided"])
        if collided:
            return dict(collided=True, near_miss=False, steps=steps, fuel=fuel_used)

    near_miss = min_dist_seen < env.collision_threshold * NEAR_MISS_MULTIPLIER
    return dict(collided=False, near_miss=near_miss, steps=steps, fuel=fuel_used)


def evaluate_agent(model_path, use_uncertainty, n_episodes=200, num_debris=3):
    env = OrbitalEnv(num_satellites=1, num_debris=num_debris)
    net = load_model(model_path, env.obs_dim())

    results = [run_eval_episode(env, net, use_uncertainty, seed=s) for s in range(n_episodes)]

    collision_rate = np.mean([r["collided"] for r in results])
    near_miss_rate = np.mean([r["near_miss"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_fuel = np.mean([r["fuel"] for r in results])

    return dict(
        collision_rate=collision_rate,
        near_miss_rate=near_miss_rate,
        avg_episode_length=avg_steps,
        avg_fuel_used=avg_fuel,
    )


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    agents = {
        "baseline":          dict(path="baseline_model.pt",          use_uncertainty=False),
        "risk_only":         dict(path="risk_only_model.pt",         use_uncertainty=False),
        "uncertainty_only":  dict(path="uncertainty_only_model.pt",  use_uncertainty=True),
        "full (COSMAEGIS)":  dict(path="full_model.pt",               use_uncertainty=True),
    }

    rows = []
    for name, cfg in agents.items():
        model_path = os.path.join(results_dir, cfg["path"])
        if not os.path.exists(model_path):
            print(f"[skip] {name}: model file not found ({model_path}). Train it first.")
            continue
        metrics = evaluate_agent(model_path, cfg["use_uncertainty"])
        metrics["agent"] = name
        rows.append(metrics)
        print(f"{name}: {metrics}")

    if rows:
        out_csv = os.path.join(results_dir, "comparison_table.csv")
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["agent", "collision_rate", "near_miss_rate",
                                                     "avg_episode_length", "avg_fuel_used"])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\nSaved comparison table to {out_csv}")


if __name__ == "__main__":
    main()
