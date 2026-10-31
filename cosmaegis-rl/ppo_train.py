import os
import sys
import csv
import time
import argparse
import numpy as np


# ======================================================================
# helpers that do not need torch
# ======================================================================

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - float(dones[t])
        next_v = last_value if t == n - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    returns = adv + np.asarray(values, dtype=np.float32)
    return adv, returns


def cvar_weights(rewards, dones, alpha=0.2, weight=3.0):
    rewards = np.asarray(rewards, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    ids = np.concatenate([[0.0], np.cumsum(dones)[:-1]]).astype(int)
    totals = np.bincount(ids, weights=rewards)
    cutoff = np.quantile(totals, alpha)
    w = np.where(totals[ids] <= cutoff, weight, 1.0)
    return w.astype(np.float32), totals, cutoff


# ---- torch part ----
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "env"))
sys.path.append(os.path.join(HERE, "agents"))

from orbital_env import OrbitalEnv

VARIANTS = ("baseline", "risk_only", "uncertainty_only", "full", "mc_nocaution")
REWARD_SCALE = 0.1


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim=2, hidden=64, dropout_p=0.2):
        super().__init__()
        self.hidden = hidden
        self.p = dropout_p
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, act_dim)
        nn.init.orthogonal_(self.out.weight, gain=0.01)
        nn.init.zeros_(self.out.bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), -0.5))
        scale = torch.tensor(np.tile([10.0, 10.0, 2.0, 2.0], obs_dim // 4),
                             dtype=torch.float32)
        self.register_buffer("obs_scale", scale, persistent=False)

    def ones_masks(self, batch):
        ones = torch.ones(batch, 1, self.hidden)
        return ones, ones.clone()

    def sample_masks(self, batch, k):
        if self.p <= 0.0:
            ones = torch.ones(batch, k, self.hidden)
            return ones, ones.clone()
        keep = 1.0 - self.p
        m1 = (torch.rand(batch, k, self.hidden) < keep).float() / keep
        m2 = (torch.rand(batch, k, self.hidden) < keep).float() / keep
        return m1, m2

    def mean_and_unc(self, obs, m1, m2):
        x = (obs / self.obs_scale).unsqueeze(1)
        h = torch.relu(self.fc1(x)) * m1
        h = torch.relu(self.fc2(h)) * m2
        out = torch.tanh(self.out(h))
        mean = out.mean(dim=1)
        if out.shape[1] > 1:
            unc = out.std(dim=1).mean(dim=-1)
        else:
            unc = torch.zeros(out.shape[0])
        return mean, unc


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, 1)
        scale = torch.tensor(np.tile([10.0, 10.0, 2.0, 2.0], obs_dim // 4),
                             dtype=torch.float32)
        self.register_buffer("obs_scale", scale, persistent=False)

    def forward(self, x):
        x = x / self.obs_scale
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


def policy_mean(actor, obs, m1, m2, use_unc):
    mean, unc = actor.mean_and_unc(obs, m1, m2)
    if use_unc:
        caution = (1.0 - unc).clamp(min=0.3).detach().unsqueeze(-1)
        mean = mean * caution
    return mean, unc


def train_ppo(variant="baseline", seed=0, total_steps=300_000, dropout_p=0.2,
              n_mc=15, cvar_alpha=0.2, cvar_weight=3.0, n_steps=2048,
              epochs=10, minibatch=128, lr=3e-4, gamma=0.99, lam=0.95,
              clip=0.2, ent_coef=0.0, env_kwargs=None, verbose=True):
    assert variant in VARIANTS, f"unknown variant {variant}"
    use_unc = variant in ("uncertainty_only", "full")
    use_risk = variant in ("risk_only", "full")
    use_mc = use_unc or variant == "mc_nocaution"   # 15-pass average, with or without caution
    k_train = n_mc if use_mc else 1
    env_kwargs = env_kwargs or dict(num_satellites=1, num_debris=3)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = OrbitalEnv(**env_kwargs)
    obs_dim = env.obs_dim()
    actor = Actor(obs_dim, dropout_p=dropout_p)
    critic = Critic(obs_dim)
    params = list(actor.parameters()) + list(critic.parameters())
    opt = torch.optim.Adam(params, lr=lr, eps=1e-5)

    obs = env.reset()[0]
    recent_collisions = []
    n_rollouts = max(1, total_steps // n_steps)
    t0 = time.time()

    for it in range(1, n_rollouts + 1):
        B_obs, B_m1, B_m2, B_act, B_logp, B_val = [], [], [], [], [], []
        B_rew, B_done = [], []

        for _ in range(n_steps):
            o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                m1, m2 = actor.sample_masks(1, k_train)
                mean, _ = policy_mean(actor, o, m1, m2, use_unc)
                dist = torch.distributions.Normal(mean, actor.log_std.exp())
                a = dist.sample()
                logp = dist.log_prob(a).sum(-1)
                v = critic(o)

            a_env = np.clip(a[0].numpy(), -1.0, 1.0)
            obs_list, rewards, done, info = env.step([a_env])

            B_obs.append(o[0])
            B_m1.append(m1[0])
            B_m2.append(m2[0])
            B_act.append(a[0])
            B_logp.append(logp[0])
            B_val.append(v[0])
            B_rew.append(float(rewards[0]) * REWARD_SCALE)
            B_done.append(float(done))

            if done:
                recent_collisions.append(bool(info["collided"][0]))
                obs = env.reset()[0]
            else:
                obs = obs_list[0]

        with torch.no_grad():
            last_v = float(critic(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0])

        rewards_np = np.array(B_rew, dtype=np.float64)
        dones_np = np.array(B_done, dtype=np.float64)
        values_np = torch.stack(B_val).numpy()
        adv, ret = compute_gae(rewards_np, values_np, dones_np, last_v, gamma, lam)

        if use_risk:
            w, _, _ = cvar_weights(rewards_np, dones_np, cvar_alpha, cvar_weight)
        else:
            w = np.ones(n_steps, dtype=np.float32)

        T_obs = torch.stack(B_obs)
        T_m1 = torch.stack(B_m1)
        T_m2 = torch.stack(B_m2)
        T_act = torch.stack(B_act)
        T_logp = torch.stack(B_logp)
        T_adv = torch.as_tensor(adv)
        T_ret = torch.as_tensor(ret)
        T_w = torch.as_tensor(w)

        for _ in range(epochs):
            perm = torch.randperm(n_steps)
            for s in range(0, n_steps, minibatch):
                b = perm[s:s + minibatch]
                if len(b) < 2:
                    continue
                mean, _ = policy_mean(actor, T_obs[b], T_m1[b], T_m2[b], use_unc)
                dist = torch.distributions.Normal(mean, actor.log_std.exp())
                new_logp = dist.log_prob(T_act[b]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = torch.exp(new_logp - T_logp[b])

                a_b = T_adv[b]
                a_b = (a_b - a_b.mean()) / (a_b.std() + 1e-8)
                s1 = ratio * a_b
                s2 = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * a_b
                pi_loss = -(torch.min(s1, s2) * T_w[b]).mean()
                v_loss = F.mse_loss(critic(T_obs[b]), T_ret[b])
                loss = pi_loss + 0.5 * v_loss - ent_coef * entropy

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 0.5)
                opt.step()

        if verbose and (it % 10 == 0 or it == n_rollouts):
            last = recent_collisions[-300:]
            rate = 100.0 * np.mean(last) if last else float("nan")
            print(f"[{variant} s{seed}] rollout {it}/{n_rollouts} | "
                  f"steps {it * n_steps} | train collisions (last 300 ep): {rate:5.1f}% | "
                  f"std {actor.log_std.exp().mean().item():.3f} | "
                  f"{time.time() - t0:.0f}s", flush=True)

    meta = dict(variant=variant, seed=seed, dropout_p=dropout_p, n_mc=n_mc,
                obs_dim=obs_dim, total_steps=total_steps)
    return actor, meta


def make_act_fn(actor, use_unc, n_mc=15):
    @torch.no_grad()
    def act(obs):
        x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if use_unc:
            m1, m2 = actor.sample_masks(1, n_mc)
        else:
            m1, m2 = actor.ones_masks(1)
        mean, unc = actor.mean_and_unc(x, m1, m2)
        if use_unc:
            mean = mean * (1.0 - unc).clamp(min=0.3).unsqueeze(-1)
        return mean[0].numpy(), float(unc[0])
    return act


def evaluate(act_fn, n_episodes=500, seed0=10_000, env_kwargs=None):
    env_kwargs = env_kwargs or dict(num_satellites=1, num_debris=3)
    env = OrbitalEnv(**env_kwargs)
    collided, near, fuels, lens, unc0 = [], [], [], [], []
    for s in range(n_episodes):
        np.random.seed(seed0 + s)
        torch.manual_seed(seed0 + s)
        obs = env.reset()[0]
        done, hit, fuel, steps, first = False, False, 0.0, 0, True
        while not done:
            a, u = act_fn(obs)
            if first:
                unc0.append(u)
                first = False
            a = np.clip(a, -1.0, 1.0)
            fuel += float(np.linalg.norm(a))
            obs_list, _, done, info = env.step([a])
            obs = obs_list[0]
            steps += 1
            if info["collided"][0]:
                hit = True
                break
        collided.append(hit)
        fuels.append(fuel)
        lens.append(steps)
        near.append((not hit) and env.min_dist_seen < env.collision_threshold * 2.0)
    return dict(collision_rate=100.0 * float(np.mean(collided)),
                near_miss_rate=100.0 * float(np.mean(near)),
                avg_fuel=float(np.mean(fuels)),
                avg_len=float(np.mean(lens)),
                collided=collided, unc0=unc0)


def save_model(actor, path, meta):
    torch.save({"state": actor.state_dict(), "meta": meta}, path)


def load_model(path):
    ck = torch.load(path, map_location="cpu")
    meta = ck["meta"]
    actor = Actor(meta["obs_dim"], dropout_p=meta["dropout_p"])
    actor.load_state_dict(ck["state"])
    return actor, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="baseline", choices=VARIANTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--no-refs", action="store_true")
    args = ap.parse_args()

    use_unc = args.variant in ("uncertainty_only", "full")
    print(f"Training PPO variant={args.variant} seed={args.seed} "
          f"steps={args.steps} dropout={args.dropout}", flush=True)
    actor, meta = train_ppo(args.variant, seed=args.seed, total_steps=args.steps,
                            dropout_p=args.dropout)

    results_dir = os.path.join(HERE, "results")
    os.makedirs(results_dir, exist_ok=True)
    tag = "" if args.dropout > 0 else "_nodrop"
    save_path = os.path.join(results_dir, f"ppo_{args.variant}_s{args.seed}{tag}.pt")
    save_model(actor, save_path, meta)
    print(f"Saved model to {save_path}")

    rows = []
    if not args.no_refs:
        from diagnose import policy_noop, policy_rule
        ref_env = OrbitalEnv(num_satellites=1, num_debris=3)
        rows.append(("do-nothing",
                     evaluate(lambda o: (policy_noop(o, ref_env), 0.0), args.episodes)))
        rows.append(("rule-based",
                     evaluate(lambda o: (policy_rule(o, ref_env), 0.0), args.episodes)))
    res = evaluate(make_act_fn(actor, use_unc), args.episodes)
    rows.append((f"PPO {args.variant} (seed {args.seed})", res))

    print(f"\n{args.episodes} test episodes\n")
    print(f"{'policy':30s} {'collisions':>11s} {'near-miss':>10s} {'avg fuel':>9s} {'avg steps':>10s}")
    for name, r in rows:
        print(f"{name:30s} {r['collision_rate']:10.1f}% {r['near_miss_rate']:9.1f}% "
              f"{r['avg_fuel']:9.2f} {r['avg_len']:10.1f}")

    csv_path = os.path.join(results_dir, "ppo_results.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        wr = csv.writer(f)
        if new_file:
            wr.writerow(["variant", "seed", "steps", "dropout", "episodes",
                         "collision_rate", "near_miss_rate", "avg_fuel", "avg_len"])
        wr.writerow([args.variant, args.seed, args.steps, args.dropout, args.episodes,
                     f"{res['collision_rate']:.2f}", f"{res['near_miss_rate']:.2f}",
                     f"{res['avg_fuel']:.3f}", f"{res['avg_len']:.2f}"])
    print(f"\nAppended result to {csv_path}")


if __name__ == "__main__":
    main()