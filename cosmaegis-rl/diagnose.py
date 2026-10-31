import os
import sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "env"))
sys.path.append(os.path.join(HERE, "agents"))

from orbital_env import OrbitalEnv
from policy import PolicyNet

N_EPISODES = 500
SAFE_DIST = 2.5
HORIZON = 8.0


def policy_noop(obs, env):
    return np.zeros(2)


def policy_random(obs, env):
    return np.random.uniform(-1, 1, size=2)


def policy_rule(obs, env):
    sat_vel = obs[2:4]
    best_miss = None
    best_dir = None
    for k in range(env.num_debris):
        base = 4 + 4 * k
        r = obs[base:base + 2]
        w = obs[base + 2:base + 4] - sat_vel
        denom = float(np.dot(w, w))
        t = 0.0 if denom < 1e-9 else float(np.clip(-np.dot(r, w) / denom, 0.0, HORIZON))
        m = r + w * t
        miss = float(np.linalg.norm(m))
        if miss < SAFE_DIST and (best_miss is None or miss < best_miss):
            best_miss = miss
            if miss < 1e-6:
                best_dir = np.array([-w[1], w[0]])
            else:
                best_dir = -m
    if best_dir is not None:
        n = np.linalg.norm(best_dir)
        if n > 1e-9:
            return best_dir / n
    return np.clip(-sat_vel, -1.0, 1.0)


def make_policy_net(path, obs_dim):
    net = PolicyNet(obs_dim=obs_dim)
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()

    def pol(obs, env):
        with torch.no_grad():
            return net(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
    return pol


def run(env, policy_fn, n_episodes=N_EPISODES):
    collided, steps_at_collision, fuels = [], [], []
    for s in range(n_episodes):
        np.random.seed(s)
        torch.manual_seed(s)
        obs_list = env.reset()
        done, steps, fuel, hit = False, 0, 0.0, False
        while not done:
            a = np.clip(policy_fn(obs_list[0], env), -1.0, 1.0)
            fuel += float(np.linalg.norm(a))
            obs_list, _, done, info = env.step([a])
            steps += 1
            if info["collided"][0]:
                hit = True
                break
        collided.append(hit)
        fuels.append(fuel)
        if hit:
            steps_at_collision.append(steps)
    rate = 100 * np.mean(collided)
    med = np.median(steps_at_collision) if steps_at_collision else float("nan")
    return rate, med, float(np.mean(fuels))


def main():
    env = OrbitalEnv(num_satellites=1, num_debris=3)
    policies = {
        "do-nothing": policy_noop,
        "random": policy_random,
        "rule-based": policy_rule,
    }
    model_path = os.path.join(HERE, "results", "baseline_model.pt")
    if os.path.exists(model_path):
        policies["RL baseline (current .pt)"] = make_policy_net(model_path, env.obs_dim())

    print(f"{N_EPISODES} test episodes, seeds 0..{N_EPISODES - 1}\n")
    print(f"{'policy':28s} {'collisions':>11s} {'median step':>12s} {'avg fuel':>9s}")
    for name, fn in policies.items():
        rate, med, fuel = run(env, fn)
        print(f"{name:28s} {rate:10.1f}% {med:12.1f} {fuel:9.1f}")


if __name__ == "__main__":
    main()