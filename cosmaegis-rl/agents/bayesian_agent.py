"""
bayesian_agent.py

The full COSMAEGIS agent: uncertainty-aware (MC Dropout) AND
risk-sensitive (CVaR-style reward weighting). This is the paper's
core contribution.

Also exposes two ablation configs (uncertainty only, risk-sensitive
only) so results.py / the paper can report a clean ablation table
showing each piece's individual contribution.
"""

import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import train


def main(variant="full"):
    """
    variant:
      "full"              -> uncertainty + risk-sensitive (COSMAEGIS)
      "uncertainty_only"  -> ablation
      "risk_only"         -> ablation
    """
    configs = {
        "full": dict(use_uncertainty=True, risk_sensitive=True),
        "uncertainty_only": dict(use_uncertainty=True, risk_sensitive=False),
        "risk_only": dict(use_uncertainty=False, risk_sensitive=True),
    }
    cfg = configs[variant]

    net, history = train(
        num_satellites=1,
        num_debris=3,
        episodes=1500,
        seed=0,
        **cfg,
    )

    out_path = os.path.join(os.path.dirname(__file__), "..", "results", f"{variant}_model.pt")
    torch.save(net.state_dict(), out_path)
    print(f"Saved {variant} model to {out_path}")
    return net, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="full",
                        choices=["full", "uncertainty_only", "risk_only"])
    args = parser.parse_args()
    main(args.variant)
