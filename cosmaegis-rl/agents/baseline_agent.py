"""
baseline_agent.py

The "control group" agent: standard REINFORCE, no uncertainty
awareness, no risk-sensitivity. This is what COSMAEGIS is compared
against in the paper's results section.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import train


def main():
    net, history = train(
        num_satellites=1,
        num_debris=3,
        episodes=3000,
        use_uncertainty=False,
        risk_sensitive=False,
        seed=0,
    )
    out_path = os.path.join(os.path.dirname(__file__), "..", "results", "baseline_model.pt")
    torch.save(net.state_dict(), out_path)
    print(f"Saved baseline model to {out_path}")
    return net, history


if __name__ == "__main__":
    main()
