import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "env"))
sys.path.append(os.path.join(HERE, "agents"))

from orbital_env import OrbitalEnv
from diagnose import run, policy_noop, policy_random, policy_rule

TIMESTEPS = 300_000


class OrbitGym(gym.Env):
    def __init__(self, num_debris=3):
        super().__init__()
        self.env = OrbitalEnv(num_satellites=1, num_debris=num_debris)
        d = self.env.obs_dim()
        self.scale = np.tile(np.array([10.0, 10.0, 2.0, 2.0], dtype=np.float32), d // 4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(d,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def _obs(self, o):
        return np.asarray(o, dtype=np.float32) / self.scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._obs(self.env.reset()[0]), {}

    def step(self, action):
        obs_list, rewards, done, info = self.env.step([np.asarray(action, dtype=float)])
        collided = bool(info["collided"][0])
        timed_out = bool(info["timed_out"])
        terminated = collided or (bool(done) and not timed_out)
        truncated = timed_out and not collided
        return self._obs(obs_list[0]), float(rewards[0]) * 0.1, terminated, truncated, {}


def main():
    np.random.seed(0)
    gym_env = OrbitGym()
    model = PPO("MlpPolicy", gym_env, seed=0, verbose=1,
                policy_kwargs=dict(net_arch=[64, 64]))
    print(f"Training PPO for {TIMESTEPS} steps (watch ep_rew_mean: it should trend upward)...")
    model.learn(total_timesteps=TIMESTEPS)

    scale = gym_env.scale

    def ppo_policy(obs, env):
        a, _ = model.predict(np.asarray(obs, dtype=np.float32) / scale, deterministic=True)
        return a

    env = OrbitalEnv(num_satellites=1, num_debris=3)
    policies = {
        "do-nothing": policy_noop,
        "random": policy_random,
        "rule-based": policy_rule,
        "PPO (Stable-Baselines3)": ppo_policy,
    }
    print("\n500 test episodes, seeds 0..499\n")
    print(f"{'policy':28s} {'collisions':>11s} {'median step':>12s} {'avg fuel':>9s}")
    for name, fn in policies.items():
        rate, med, fuel = run(env, fn)
        print(f"{name:28s} {rate:10.1f}% {med:12.1f} {fuel:9.1f}")


if __name__ == "__main__":
    main()