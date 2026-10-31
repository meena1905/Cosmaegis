# EXPLAINER — read this before you submit or present

This is a plain-language map of the whole project. If you only have
time to read one file before your viva/demo, read this one.

## The one-sentence pitch
An RL agent controls a satellite's thrusters to avoid debris. Unlike a
normal RL agent, it estimates its own uncertainty and becomes more
cautious when unsure, and it's trained to specifically avoid worst-case
outcomes (collisions), not just optimize average performance.

## File-by-file, what it does and why

### `env/satellite.py`
The smallest building block. A `Satellite` object just has a position
`(x, y)` and velocity `(vx, vy)`, and a `step()` method that moves it:
`position += velocity * dt`. Used for both actual satellites and debris
— they're physically the same kind of object in this simplified world.

### `env/orbital_env.py`
The simulated world. This is the standard `reset()` / `step(action)`
interface used throughout reinforcement learning (matches the format
used by OpenAI Gym / Gymnasium, the industry standard).

- `reset()`: creates a new episode with satellites and debris at random
  starting positions/velocities. Randomization matters — if the
  scenario were always identical, the agent could just memorize one
  answer instead of learning general collision-avoidance behavior.
- `step(actions)`: applies each satellite's thrust, moves everything
  forward one timestep, checks for collisions (distance between
  objects < threshold), and returns a reward. Collisions give -100;
  every other step gives a small negative "fuel cost" penalty. This
  ratio (-100 vs ~-0.01/step) is deliberate: across a whole episode,
  even the worst case of never risking anything (-0.01 × 200 steps =
  -2) is far better than a single collision (-100), so the agent
  strongly prefers avoiding collision even though no single step
  explicitly says "good job."
- Supports multiple satellites (`num_satellites`) for the multi-agent
  extension — each is an "independent learner" using the same policy
  network but observing itself and the others separately.

### `agents/policy.py`
The neural network — the agent's "brain." Given an observation (its
own position/velocity + the position/velocity of all debris/other
satellites relative to it), it outputs a 2D thrust direction.

The key trick, **MC Dropout**: normally, "dropout" (randomly turning
off some neurons) is used only during training to prevent overfitting.
Here, we deliberately keep dropout ON even when the agent is making
real decisions, and ask the network the same question ~15 times. Since
dropout randomly changes which neurons are active each time, we get 15
slightly different answers.
- If all 15 largely agree → low uncertainty → the network is confident
- If the 15 disagree a lot → high uncertainty → the network is unsure

This is a cheap, well-established approximation of full Bayesian
inference (see Gal & Ghahramani, "Dropout as a Bayesian Approximation," 2016).

### `train.py`
The training loop. Uses **REINFORCE** (a foundational policy-gradient
RL algorithm): the agent tries actions, sees the total reward at the
end of an episode, and adjusts its network to make actions that led to
higher reward more likely in the future, and actions that led to lower
reward less likely.

Two optional behaviors turn this into the full COSMAEGIS agent:
- `use_uncertainty=True`: actions are chosen via MC Dropout, and if
  the uncertainty is high, the chosen action's magnitude is scaled
  down — i.e., the agent literally moves more cautiously (smaller
  thrust) when it's unsure, rather than confidently committing.
- `risk_sensitive=True`: after collecting a batch of episodes, if a
  meaningful fraction of them were unusually bad (bottom 20% by total
  reward — a CVaR-style cutoff), the whole batch's training update is
  up-weighted. This pushes the agent to fix its worst-case behavior
  specifically, not just its average behavior.

### `agents/baseline_agent.py` and `agents/bayesian_agent.py`
Thin entry-point scripts that call `train.py` with different settings.
`baseline_agent.py` = the control group (neither switch on).
`bayesian_agent.py --variant full` = both switches on (your actual
contribution). The `--variant uncertainty_only` / `risk_only` options
exist so you can show an **ablation table**: does each piece help on
its own, or only together? This is exactly the kind of evidence
reviewers/evaluators look for.

### `evaluate.py`
Runs every trained agent through the same 200 test episodes (same
random seeds for all agents, so the comparison is fair — not luck)
and reports: collision rate, near-miss rate, average episode length,
average fuel used. Saves this as `results/comparison_table.csv` — this
IS your results section.

### `demo.py`
Produces the animated visual — satellite (blue) dodging debris (red),
with a live uncertainty readout shown on screen for the uncertainty-
aware agent. This is what you show live in a presentation or record as
a clip.

## Likely viva questions and how to answer them

**"Why not use real orbital mechanics (SGP4/TLE data)?"**
Because the research question is about decision-making under
uncertainty, not about physics simulation accuracy. A simplified
kinematic model isolates that question cleanly — this is standard
practice in early-stage RL research (cite: Kahn et al. 2017 used
simulated robots, not real hardware, for exactly this reason).

**"Why REINFORCE and not PPO?"**
REINFORCE is simpler to implement, understand, and explain fully —
appropriate for this project's scope and timeline. PPO would likely
be more sample-efficient; that's noted as future work.

**"How do you know the uncertainty estimate is meaningful, not just
noise?"**
Good challenge to acknowledge honestly: the project's evaluation
should ideally check whether high predicted uncertainty correlates
with actually risky situations (e.g., close approaches). If you have
time, add this as an extra evaluation metric; if not, list it as a
limitation/future work item — reviewers respect honesty about this far
more than an unsupported claim that it "works."

**"What's actually novel here?"**
Not any single technique — each piece (MC Dropout uncertainty, CVaR
risk-sensitivity, multi-agent RL) exists individually in prior work.
The novel piece is combining uncertainty-awareness + risk-sensitivity
specifically for multi-satellite collision avoidance, with an ablation
showing each piece's contribution — a combination not found in prior
published work on this exact problem, to the best of our search.

## If results don't look great
Be honest about it in the paper rather than hiding it — an honest
negative or mixed result with good analysis of *why* is more
publishable/defensible at student level than a suspiciously perfect
result. If your uncertainty-aware agent's collision rate isn't clearly
better than baseline, discuss possible reasons (training budget,
reward tuning, environment difficulty) rather than overclaiming.
