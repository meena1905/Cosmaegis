# COSMAEGIS: Uncertainty-Aware Risk-Sensitive Collision Avoidance

A reinforcement learning project for autonomous satellite collision avoidance.
An RL agent controls a satellite's thrust to avoid debris, using:

1. **Bayesian uncertainty (MC Dropout)** — the agent knows when it's unsure,
   and becomes more cautious in unfamiliar situations.
2. **Risk-sensitivity (CVaR-style)** — training prioritizes avoiding
   worst-case outcomes (collisions), not just average performance.
3. **Multi-agent support** — the environment supports multiple
   independently-controlled satellites (set `num_satellites > 1`).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Project structure

```
cosmaegis-rl/
├── env/
│   ├── satellite.py       # basic moving-object class (position + velocity)
│   └── orbital_env.py     # the simulated world: reset(), step(), render()
├── agents/
│   ├── policy.py          # shared neural network + MC Dropout uncertainty
│   ├── baseline_agent.py  # control-group agent (no uncertainty/risk features)
│   └── bayesian_agent.py  # COSMAEGIS agent (+ ablation variants)
├── train.py                # core REINFORCE training loop
├── evaluate.py              # runs all trained agents through test episodes,
│                             produces the comparison table
├── demo.py                  # animated visual demo of a trained agent
├── results/                  # trained models + comparison_table.csv land here
├── paper/
│   └── paper.md              # research paper draft
└── EXPLAINER.md              # plain-language walkthrough of every file —
                               read this before your viva/defense
```

## How to run everything, in order

### 1. Train the baseline agent (control group)
```bash
python agents/baseline_agent.py
```

### 2. Train the ablations + full COSMAEGIS agent
```bash
python agents/bayesian_agent.py --variant risk_only
python agents/bayesian_agent.py --variant uncertainty_only
python agents/bayesian_agent.py --variant full
```
Each of these trains for 1500 episodes. On a normal laptop CPU this
should take roughly 5-15 minutes per agent — no GPU required. If
you're very short on time, you can lower `episodes=1500` to something
smaller (e.g. 500) inside `agents/baseline_agent.py` and
`agents/bayesian_agent.py` — note this in your paper's limitations
section as "reduced training budget due to time constraints," which is
an honest and acceptable thing to state.

### 3. Evaluate and compare
```bash
python evaluate.py
```
This prints and saves `results/comparison_table.csv` — your paper's
main results table (collision rate, near-miss rate, episode length,
fuel used, for all four agents).

### 4. Generate the demo
```bash
python demo.py --model results/full_model.pt --uncertainty
python demo.py --model results/baseline_model.pt
```
Add `--save results/demo.gif` to save an animation file instead of
displaying it live (useful for embedding in slides).

## Important note on this project's scope

This project deliberately uses a **simplified 2D kinematic simulation**,
not real orbital mechanics (SGP4/Keplerian propagation). This is a
documented, defensible research design choice — see `paper/paper.md`
section "Limitations" for the full justification. It isolates the
research question (does uncertainty-aware RL reduce collisions?) from
the separate, well-solved problem of orbital physics simulation.

## If you're asked to explain this project

Read `EXPLAINER.md` — it walks through what every file does and why,
in plain language, specifically so you can answer questions about your
own submission.
