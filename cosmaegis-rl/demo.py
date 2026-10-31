"""
demo.py

Produces the visual demo: an animated episode showing the satellite
(blue), debris (red), and -- for the uncertainty-aware agent -- a
live readout of the agent's current uncertainty level, so you can
SEE it become more cautious in unfamiliar situations.

Usage:
    python demo.py --model results/full_model.pt --uncertainty
    python demo.py --model results/baseline_model.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.append(os.path.join(os.path.dirname(__file__), "env"))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from orbital_env import OrbitalEnv
from policy import PolicyNet, mc_dropout_predict


def run_demo(model_path, use_uncertainty, num_debris=3, seed=42, save_path=None):
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = OrbitalEnv(num_satellites=1, num_debris=num_debris)
    net = PolicyNet(obs_dim=env.obs_dim())
    net.load_state_dict(torch.load(model_path, map_location="cpu"))

    obs_list = env.reset()
    frames = []  # list of (sat_pos, debris_positions, uncertainty)

    done = False
    while not done:
        obs_t = torch.tensor(obs_list[0], dtype=torch.float32).unsqueeze(0)
        if use_uncertainty:
            mean_action, uncertainty = mc_dropout_predict(net, obs_t, n_samples=15)
            action = mean_action.squeeze(0).detach().numpy()
            unc_value = uncertainty.item()
        else:
            net.eval()
            with torch.no_grad():
                action = net(obs_t).squeeze(0).numpy()
            unc_value = 0.0

        action = np.clip(action, -1.0, 1.0)
        frames.append((
            env.satellites[0].position.copy(),
            [d.position.copy() for d in env.debris],
            unc_value,
        ))

        obs_list, rewards, done, info = env.step([action])
        if info["collided"][0]:
            frames.append((
                env.satellites[0].position.copy(),
                [d.position.copy() for d in env.debris],
                unc_value,
            ))
            break

    fig, ax = plt.subplots(figsize=(6, 6))
    lim = env.world_size + 2

    def update(frame_idx):
        ax.clear()
        sat_pos, debris_positions, unc = frames[frame_idx]
        ax.scatter(*sat_pos, color="blue", s=140, label="Satellite", zorder=3)
        for dp in debris_positions:
            ax.scatter(*dp, color="red", s=50, zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        title = f"Step {frame_idx}"
        if use_uncertainty:
            level = "HIGH" if unc > 0.15 else "low"
            title += f" | uncertainty: {unc:.3f} ({level})"
        ax.set_title(title)
        ax.grid(True)
        ax.legend(loc="upper right")

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=100, repeat=False)

    if save_path:
        ani.save(save_path, writer="pillow", fps=8)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to a saved .pt model")
    parser.add_argument("--uncertainty", action="store_true",
                        help="use MC-dropout uncertainty-aware action selection")
    parser.add_argument("--save", default=None, help="optional path to save .gif instead of showing")
    args = parser.parse_args()

    run_demo(args.model, use_uncertainty=args.uncertainty, save_path=args.save)
