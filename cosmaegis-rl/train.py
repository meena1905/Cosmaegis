"""
train.py  (v3)

REINFORCE-style policy gradient with a learned value baseline
(advantage = reward-to-go minus value estimate), gradient clipping and a
lower learning rate. Shared by all agents (they differ only in config).

  use_uncertainty=True : MC Dropout action selection with caution scaling
  risk_sensitive=True  : worst cvar_alpha-fraction of episodes in each
                         batch get their gradient multiplied by cvar_weight
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from orbital_env import OrbitalEnv
from policy import PolicyNet, mc_dropout_predict

ACTION_STD = 0.15   # fixed exploration noise around the chosen mean action
RTG_SCALE = 50.0   # reward-to-go is divided by this before the critic sees it


class ValueNet(nn.Module):
    """Critic: estimates the (scaled) reward-to-go from an observation."""
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        scale = torch.tensor([10.0, 10.0, 2.0, 2.0] * (obs_dim // 4))
        self.register_buffer("obs_scale", scale, persistent=False)

    def forward(self, x):
        x = x / self.obs_scale
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


def select_action(net, obs, use_uncertainty):
    """Used during TRAINING, so no torch.no_grad()."""
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    net.train()  # dropout active

    if use_uncertainty:
        samples = torch.stack([net(obs_t) for _ in range(15)])
        mean_action = samples.mean(dim=0).squeeze(0)
        uncertainty_value = samples.std(dim=0).mean().item()
        caution_scale = max(0.3, 1.0 - min(uncertainty_value, 1.0))
        mean_action = mean_action * caution_scale
    else:
        mean_action = net(obs_t).squeeze(0)
        uncertainty_value = 0.0

    dist = torch.distributions.Normal(mean_action, ACTION_STD)
    action_sample = dist.sample()
    log_prob = dist.log_prob(action_sample).sum()
    action_clamped = torch.clamp(action_sample, -1.0, 1.0)
    return action_clamped.detach().numpy(), log_prob, uncertainty_value


def collect_episode(env, net, use_uncertainty):
    obs_list = env.reset()
    done = False
    n = env.num_satellites
    log_probs = [[] for _ in range(n)]
    rewards = [[] for _ in range(n)]
    observations = [[] for _ in range(n)]
    thrusts = []

    while not done:
        actions = []
        for i in range(n):
            observations[i].append(obs_list[i])
            action, log_prob, _ = select_action(net, obs_list[i], use_uncertainty)
            actions.append(action)
            log_probs[i].append(log_prob)
            thrusts.append(float(np.linalg.norm(action)))

        obs_list, step_rewards, done, info = env.step(actions)
        for i in range(n):
            rewards[i].append(step_rewards[i])

    return log_probs, rewards, observations, thrusts


def reward_to_go(rewards, gamma=0.99):
    rtg = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        rtg[t] = running
    return rtg


def train(num_satellites=1, num_debris=3, episodes=800,
          use_uncertainty=False, risk_sensitive=False,
          cvar_alpha=0.2, cvar_weight=3.0, lr=3e-4, seed=0,
          verbose_every=50, batch_size=4):

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = OrbitalEnv(num_satellites=num_satellites, num_debris=num_debris)
    net = PolicyNet(obs_dim=env.obs_dim())
    value_net = ValueNet(env.obs_dim())
    params = list(net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=lr)

    history = []         # total reward per episode
    thrust_history = []  # average thrust magnitude per episode
    batch = []           # (log_probs, scaled rtg, observations, total reward)

    for ep in range(episodes):
        log_probs, rewards, observations, thrusts = collect_episode(env, net, use_uncertainty)

        ep_lp, ep_rtg, ep_obs = [], [], []
        for i in range(num_satellites):
            ep_lp.extend(log_probs[i])
            ep_rtg.extend(reward_to_go(rewards[i]).tolist())
            ep_obs.extend(observations[i])

        total_reward = sum(sum(r) for r in rewards)
        history.append(total_reward)
        thrust_history.append(float(np.mean(thrusts)))
        batch.append((
            torch.stack(ep_lp),
            torch.tensor(ep_rtg, dtype=torch.float32) / RTG_SCALE,
            torch.tensor(np.array(ep_obs), dtype=torch.float32),
            total_reward,
        ))

        if (ep + 1) % batch_size == 0:
            all_obs = torch.cat([b[2] for b in batch])
            all_rtg = torch.cat([b[1] for b in batch])

            values = value_net(all_obs)
            adv = all_rtg - values.detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv_per_ep = torch.split(adv, [len(b[1]) for b in batch])

            totals = np.array([b[3] for b in batch])
            cutoff = np.quantile(totals, cvar_alpha)

            terms = []
            for (lp, _, _, tot), a in zip(batch, adv_per_ep):
                w = cvar_weight if (risk_sensitive and tot <= cutoff) else 1.0
                terms.append(-lp * a * w)
            actor_loss = torch.cat(terms).mean()
            critic_loss = F.smooth_l1_loss(values, all_rtg)
            loss = actor_loss + 0.5 * critic_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 0.5)
            optimizer.step()
            batch = []

        if verbose_every and (ep + 1) % verbose_every == 0:
            print(f"Episode {ep + 1}/{episodes} | "
                  f"avg reward: {np.mean(history[-verbose_every:]):.2f} | "
                  f"avg thrust: {np.mean(thrust_history[-verbose_every:]):.2f}")

    return net, history


if __name__ == "__main__":
    net, history = train(episodes=200, verbose_every=20)
    print("Smoke test complete. Final avg reward:", np.mean(history[-20:]))