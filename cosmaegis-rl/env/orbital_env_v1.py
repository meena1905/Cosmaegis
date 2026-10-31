"""
orbital_env.py

Simplified 2D orbital collision-avoidance environment.

Design choice (documented for the paper's "limitations" section):
We use a simplified 2D kinematic model rather than real orbital mechanics
(SGP4/Keplerian propagation). This isolates the research question --
does uncertainty-aware, risk-sensitive RL reduce collisions? -- from the
separate, well-solved problem of orbital physics simulation. This mirrors
standard practice in early-stage RL research (e.g. Kahn et al. 2017 used
a simulated car/drone, not real hardware, to isolate the uncertainty
question).

Supports:
- N satellites (agents), each independently controlled
- M debris objects (randomly moving, not controlled)
- Collision detection: satellite-debris and satellite-satellite
- A standard reset()/step(actions) RL interface
- render() for a single-frame snapshot, and an episode recorder for
  building an animated demo (see demo.py)
"""

import numpy as np
from satellite import Satellite


class OrbitalEnv:
    def __init__(self, num_satellites=1, num_debris=3, max_steps=200,
                 collision_threshold=1.0, world_size=10.0):
        self.num_satellites = num_satellites
        self.num_debris = num_debris
        self.max_steps = max_steps
        self.collision_threshold = collision_threshold
        self.world_size = world_size

    # ------------------------------------------------------------------
    def reset(self):
        self.satellites = [
            Satellite(
                position=np.random.uniform(-2, 2, size=2),
                velocity=np.random.uniform(-0.5, 0.5, size=2),
            )
            for _ in range(self.num_satellites)
        ]
        self.debris = [
            Satellite(
                position=np.random.uniform(-self.world_size, self.world_size, size=2),
                velocity=np.random.uniform(-1, 1, size=2),
            )
            for _ in range(self.num_debris)
        ]
        self.alive = [True] * self.num_satellites
        self.current_step = 0
        return self._get_all_obs()

    # ------------------------------------------------------------------
    def _get_obs_for(self, i):
        """Observation for satellite i: own state + relative state of
        every debris object + every other satellite."""
        sat = self.satellites[i]
        obs = [sat.position, sat.velocity]

        for d in self.debris:
            obs.append(d.position - sat.position)
            obs.append(d.velocity)

        for j, other in enumerate(self.satellites):
            if j == i:
                continue
            obs.append(other.position - sat.position)
            obs.append(other.velocity)

        return np.concatenate(obs)

    def _get_all_obs(self):
        return [self._get_obs_for(i) for i in range(self.num_satellites)]

    def obs_dim(self):
        # own (4) + per debris (4) + per other satellite (4)
        return 4 + 4 * self.num_debris + 4 * (self.num_satellites - 1)

    # ------------------------------------------------------------------
    def step(self, actions):
        """
        actions: list of 2D thrust vectors, one per satellite (ignored
        for satellites already marked collided/dead).
        Returns: obs_list, reward_list, done (bool, shared episode end),
                 info dict (per-satellite collision flags, this step)
        """
        collided_this_step = [False] * self.num_satellites

        # apply thrust + move satellites
        for i, sat in enumerate(self.satellites):
            if not self.alive[i]:
                continue
            action = np.clip(np.array(actions[i], dtype=float), -1.0, 1.0)
            sat.velocity += action
            sat.step()

        # move debris
        for d in self.debris:
            d.step()

        self.current_step += 1

        # collision checks: satellite-debris
        for i, sat in enumerate(self.satellites):
            if not self.alive[i]:
                continue
            for d in self.debris:
                dist = np.linalg.norm(sat.position - d.position)
                if dist < self.collision_threshold:
                    collided_this_step[i] = True

        # collision checks: satellite-satellite
        for i in range(self.num_satellites):
            if not self.alive[i]:
                continue
            for j in range(i + 1, self.num_satellites):
                if not self.alive[j]:
                    continue
                dist = np.linalg.norm(self.satellites[i].position - self.satellites[j].position)
                if dist < self.collision_threshold:
                    collided_this_step[i] = True
                    collided_this_step[j] = True

        # rewards: fuel-cost-shaped step penalty + big collision penalty
        rewards = []
        for i in range(self.num_satellites):
            if not self.alive[i]:
                rewards.append(0.0)
                continue
            fuel_cost = 0.01 * np.linalg.norm(actions[i])
            step_penalty = -0.01 - fuel_cost
            if collided_this_step[i]:
                rewards.append(-100.0)
                self.alive[i] = False
            else:
                rewards.append(step_penalty)

        timed_out = self.current_step >= self.max_steps
        done = timed_out or not any(self.alive)

        obs = self._get_all_obs()
        info = {"collided": collided_this_step, "timed_out": timed_out}
        return obs, rewards, done, info

    # ------------------------------------------------------------------
    def render(self, ax=None):
        import matplotlib.pyplot as plt
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            created_fig = True

        ax.clear()
        for i, sat in enumerate(self.satellites):
            color = "blue" if self.alive[i] else "gray"
            ax.scatter(*sat.position, color=color, s=100,
                       label="Satellite" if i == 0 else None)
        for d in self.debris:
            ax.scatter(*d.position, color="red", s=40,
                       label="Debris" if d is self.debris[0] else None)

        lim = self.world_size + 2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_title(f"Step {self.current_step}")
        ax.legend(loc="upper right")
        ax.grid(True)

        if created_fig:
            plt.show()


if __name__ == "__main__":
    env = OrbitalEnv(num_satellites=1, num_debris=3)
    obs = env.reset()
    print("obs_dim:", env.obs_dim())
    print("Initial obs[0]:", obs[0])

    for step in range(5):
        actions = [np.zeros(2) for _ in range(env.num_satellites)]
        obs, rewards, done, info = env.step(actions)
        print(f"Step {step}: rewards={rewards}, done={done}, info={info}")
        if done:
            break
