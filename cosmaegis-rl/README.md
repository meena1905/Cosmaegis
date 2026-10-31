# Uncertainty-Aware Training and Caution for PPO-Based Satellite Collision Avoidance


The study asks whether Monte Carlo (MC) Dropout uncertainty and a CVaR-style weighting improve a PPO-based collision-avoidance agent in a simplified two-dimensional encounter simulator in which every episode contains a genuine collision threat.

## Main findings

Five PPO variants (baseline, CVaR-only, MC-Dropout, MC-Dropout + CVaR, and an MC-Dropout control without caution scaling) were trained with five seeds each and evaluated on the training condition and three shifted conditions (fast debris, late warning, two threats). Agents are compared at equal fuel.

- Policies trained with a 15-pass dropout-averaged action collided 0.8-1.7 percentage points less often than the baseline at equal fuel in all four conditions (for example 0.8% versus 2.0% in the training condition).
- Using the estimated uncertainty to reduce thrust at test time saved 16-26% fuel but increased collisions in every comparison, and at equal fuel it was worse than constant thrust scaling in the two harder conditions.
- CVaR-style weighting had no measurable effect, and the uncertainty estimate was only weakly informative.

## Setup

```bash
pip install -r requirements.txt
```

## Repository layout

| Path | Contents |
|---|---|
| `env/` | Simulator code. The encounter-based simulator used in the paper is `orbital_env_v5_encounter.py`. |
| `agents/` | Agent and policy code. |
| `ppo_train.py` | PPO training (baseline, CVaR weighting, MC Dropout, caution rule). |
| `run_seeds.py` | Launches training over seeds. |
| `evaluate_all.py` | Evaluates all trained models in all conditions. |
| `ablate_testtime.py` | Test-time mode comparison (plain, constant scale, 15-pass average, caution rule). |
| `frontier_testtime.py`, `frontier_extra.py` | Safety-fuel trade-off curves (thrust scaling) and equal-fuel analysis. |
| `make_figures.py` | Generates the figures used in the paper. |
| `demo.py` | Demo script. |
| `results/` | Evaluation CSV files, training logs and figures. |

## Reproducing the results

Run the scripts in this order from the repository folder. Check each script's header or `--help` for its exact arguments.

1. `python run_seeds.py` (train all variants over seeds)
2. `python evaluate_all.py` (evaluate every model in every condition)
3. `python ablate_testtime.py` (test-time modes)
4. `python frontier_testtime.py` and `python frontier_extra.py` (trade-off curves and equal-fuel comparison)
5. `python make_figures.py` (figures)

Evaluation uses 1,000 test episodes per model and condition, with test seeds starting at 20,000 (never used in training).

## Results files

- `results/eval_per_model.csv`: per-model, per-condition results.
- `results/figures/`: the five figures from the paper.
- `results/logs/`: training logs. Training logs for seed 0 of the four original variants are not included.

