# Uncertainty-Aware Risk-Sensitive Collision Avoidance for Satellite Systems using Bayesian Deep Reinforcement Learning

**Author:** [Your name]
**Institution:** [Your institution]

## Abstract

Autonomous collision avoidance is an increasingly important problem for
satellite operations as orbital traffic grows. We present COSMAEGIS, a
reinforcement learning agent for satellite collision avoidance that
combines two ideas not previously applied together in this domain:
Bayesian uncertainty estimation via Monte Carlo Dropout, which lets the
agent recognize when it is operating in an unfamiliar situation, and a
risk-sensitive training objective that specifically penalizes worst-case
outcomes rather than only optimizing average performance. We train and
evaluate our agent in a simplified 2D simulated orbital environment
against a standard reinforcement learning baseline and two ablations
(uncertainty-only, risk-sensitivity-only), reporting collision rate,
near-miss rate, episode survival length, and fuel cost. [Insert 1-2
sentence summary of your actual results here once evaluate.py has run.]

## 1. Introduction

Orbital space is becoming increasingly congested. Each collision
between satellites or debris generates additional fragments, raising
the likelihood of further collisions — a self-reinforcing process known
as the Kessler Syndrome (Kessler & Cour-Palais, 1978). As the number of
active satellites grows into the tens of thousands, manual,
human-reviewed collision avoidance does not scale, motivating research
into autonomous decision-making systems.

A key limitation of standard reinforcement learning (RL) agents applied
to this problem is that they output a single, fully-confident action
regardless of how familiar or unfamiliar the situation is. In a
safety-critical domain, this is undesirable: an agent facing a novel
sensor pattern or an unusual conjunction geometry should ideally
recognize its own uncertainty and act more conservatively, rather than
committing confidently to a possibly-incorrect maneuver.

This paper investigates whether combining (1) epistemic uncertainty
estimation and (2) a risk-sensitive training objective improves
collision-avoidance performance, specifically on worst-case outcomes,
relative to a standard RL baseline, in a multi-agent satellite setting.

## 2. Related Work

**RL for satellite collision avoidance.** Prior work has applied
standard policy-gradient methods (e.g. PPO) to train single-satellite
collision-avoidance agents in simulated astrodynamics environments,
generally reporting substantial improvements over rule-based and
impulsive-maneuver baselines. [TODO: insert full citation and specific
figures from your literature search — see conversation history for the
paper found on PPO-trained GEO collision avoidance.]

**Multi-agent RL for satellite constellations.** More recent work has
extended RL-based collision avoidance to multi-agent, communication-
constrained settings, focusing on coordination between agents rather
than on decision-making under uncertainty. [TODO: insert full
citation.]

**Bayesian deep learning for conjunction assessment.** Separately,
Bayesian recurrent neural network approaches have been applied to
*predicting* the evolution of conjunction events (i.e., forecasting
collision risk) with associated uncertainty estimates, rather than
using that uncertainty to drive an acting agent's behavior. [TODO:
insert full citation — the ESA/Kessler-related paper found during
literature search.]

**Uncertainty-aware RL in robotics.** The core technique this project
builds on — using MC Dropout to estimate epistemic uncertainty and
scale agent caution accordingly — was introduced in the robotics
literature by Kahn et al. (2017), validated on simulated ground/aerial
robots navigating around obstacles, not in the orbital domain.

**Risk-sensitive (CVaR) collision avoidance.** Conditional Value-at-Risk
objectives have been explored for multi-robot collision avoidance,
with prior work explicitly noting that principled CVaR-based approaches
accounting for prediction uncertainty in other agents' positions remain
under-explored, even in the ground-robotics domain. [TODO: insert full
citation.]

**Gap addressed by this work.** To the best of our knowledge, no prior
work combines epistemic-uncertainty-driven caution with a CVaR-style
risk-sensitive training objective in a multi-agent satellite collision-
avoidance setting. Each individual technique above has precedent in an
adjacent domain or sub-problem; this paper investigates their
combination in the orbital multi-agent setting, with an ablation study
isolating each component's contribution.

## 3. Method

### 3.1 Environment

We model collision avoidance as a simplified 2D kinematic environment.
Satellites and debris are point objects with position and velocity;
position updates via Euler integration (`position += velocity * dt`).
We deliberately do not model full orbital mechanics (e.g. SGP4/SDP4
propagation, Keplerian elements) — see Section 6 for justification.
Each episode randomizes the starting positions and velocities of all
satellites and debris, so the agent must learn general
collision-avoidance behavior rather than memorizing a fixed scenario.
A collision is registered when the distance between any two objects
falls below a fixed threshold. The environment supports an arbitrary
number of independently-controlled satellites and debris objects.

### 3.2 Reward function

At each timestep, a satellite receives a small negative reward
proportional to its thrust magnitude (a fuel-cost penalty), and a large
fixed negative reward (-100) upon collision. This reward asymmetry is
intentional: it ensures that even a full episode of conservative
fuel-spending is preferable to any risk of collision, while still
discouraging unnecessary thrust use.

### 3.3 Policy network and uncertainty estimation

The policy is a small feed-forward network (two hidden layers, 64
units each, with dropout) mapping an observation (own state plus
relative position/velocity of all debris and other satellites) to a 2D
thrust vector. Uncertainty is estimated via Monte Carlo Dropout (Gal &
Ghahramani, 2016): dropout is kept active at decision time, and the
network is queried N=15 times per decision; the standard deviation
across these samples serves as an epistemic uncertainty estimate. When
uncertainty is high, the chosen action's magnitude is scaled down,
producing more conservative behavior in unfamiliar situations.

### 3.4 Risk-sensitive training

We train via REINFORCE (Monte Carlo policy gradient) with
reward-to-go. In the risk-sensitive variant, after each training
batch, we identify episodes falling in the bottom [alpha]-quantile of
total reward (a CVaR-style cutoff) and up-weight the batch's gradient
update when a meaningful fraction of the batch was drawn from this
worst-case tail, pushing the policy to specifically improve worst-case,
not just average-case, behavior.

### 3.5 Multi-agent extension

Multiple satellites are handled as independent learners: each observes
its own state plus the relative state of all other satellites and
debris, and is trained with the same shared policy network. This is a
standard simplification in multi-agent RL research that avoids the
added complexity of centralized-training frameworks, appropriate for
this project's scope.

## 4. Experimental Setup

We compare four agents, all trained for [N] episodes with identical
environment settings (1 satellite, 3 debris) [update if you ran
multi-agent experiments] and the same random seed for network
initialization:

- **Baseline** — standard REINFORCE, no uncertainty-awareness, no
  risk-sensitivity.
- **Uncertainty-only** — MC Dropout caution scaling, standard training.
- **Risk-only** — standard action selection, CVaR-weighted training.
- **Full (COSMAEGIS)** — both uncertainty-awareness and risk-sensitivity.

All four agents are evaluated on the same 200 held-out test episodes
(fixed random seeds, identical across agents) to ensure a fair
comparison. Metrics reported: collision rate, near-miss rate (closest
approach fell below 2x the collision threshold without colliding),
average episode survival length, and average fuel used.

## 5. Results

[Insert the contents of `results/comparison_table.csv` here as a
formatted table once you've run `evaluate.py`. Report the numbers
honestly, including if the full agent does not outperform every
baseline on every metric — discuss why in Section 6.]

| Agent | Collision rate | Near-miss rate | Avg. episode length | Avg. fuel used |
|---|---|---|---|---|
| Baseline | | | | |
| Risk-only | | | | |
| Uncertainty-only | | | | |
| Full (COSMAEGIS) | | | | |

[Optional, if time permits: a plot of training reward curves for all
four agents, and/or a calibration check — does predicted uncertainty
actually correlate with proximity to debris in test episodes?]

## 6. Limitations

- **Simplified 2D kinematic environment, not real orbital mechanics.**
  This isolates the uncertainty/risk-sensitivity research question from
  the separate problem of high-fidelity orbital physics simulation,
  consistent with standard practice in early-stage RL research (e.g.
  Kahn et al. 2017 evaluated on simulated, not real, robots). Results
  should be interpreted as evidence for the *mechanism* (uncertainty-
  aware caution, risk-sensitive training) rather than as
  flight-ready collision-avoidance guidance.
- **REINFORCE rather than a more sample-efficient algorithm (e.g. PPO).**
  Chosen for implementation simplicity and full explainability within
  the project's timeline; likely leaves performance on the table.
- **Reduced training budget** relative to what a longer research
  project would use, due to time constraints. [State your actual
  episode counts here.]
- **Uncertainty calibration was not independently validated** against
  ground-truth risk in this iteration — future work should verify that
  high predicted uncertainty correlates with objectively riskier
  situations, not just network noise.
- **Multi-agent setting uses independent learners**, a simplification
  that does not model communication or fully centralized coordination.

## 7. Conclusion

We investigated combining epistemic uncertainty estimation and
risk-sensitive training for multi-agent satellite collision avoidance,
an unexplored combination in this domain based on our literature
review. [Summarize your actual findings honestly once results are in —
e.g. "The full agent achieved a lower collision rate than the baseline
at the cost of increased fuel usage" or whatever your real numbers
show.] Future work includes validating uncertainty calibration,
extending to a higher-fidelity physics model, and replacing REINFORCE
with a more sample-efficient policy-gradient method.

## References

- Kessler, D. J., & Cour-Palais, B. G. (1978). Collision frequency of
  artificial satellites: The creation of a debris belt. *Journal of
  Geophysical Research*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian
  approximation: Representing model uncertainty in deep learning.
  *ICML*.
- Kahn, G., Villaflor, A., Pong, V., Abbeel, P., & Levine, S. (2017).
  Uncertainty-aware reinforcement learning for collision avoidance.
  *arXiv preprint arXiv:1702.01182*.
- [TODO: add the remaining citations found during your literature
  search — the PPO-based single-satellite paper, the multi-agent
  constellation paper, the ESA Bayesian conjunction-prediction paper,
  and the multi-robot CVaR paper. Fill in full author/year/venue from
  the search results earlier in this project's development.]
