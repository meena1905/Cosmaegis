"""
orbital_env.py  (v5, encounter-based)

Simplified 2D orbital collision-avoidance environment.

In encounter mode (default) every episode is one conjunction event:
  - debris 0 (the "threat") is aimed to pass the satellite at a random
    miss distance in [0, threat_miss_max] at a time of closest approach
    between 4 and 10 steps from the start;
  - the episode ends when no object is approaching any satellite any more
    (the encounter is over), or on collision, or at max_steps.
Other debris are spawned randomly as before.

Earlier changes kept: safe spawn distance, swept collision check, speed
cap, min_dist_seen, proximity penalty, conjunction-style observation
features (TCA, miss vector, miss distance).
"""

import numpy as np
from satellite import Satellite


def min_dist_swept(a0, a1, b0, b1):
    r0 = a0 - b0
    rel_v = (a1 - a0) - (b1 - b0)
    denom = float(np.dot(rel_v, rel_v))
    if denom < 1e-12:
        return float(np.linalg.norm(r0))
    t = float(np.clip(-np.dot(r0, rel_v) / denom, 0.0, 1.0))
    return float(np.linalg.norm(r0 + t * rel_v))


class OrbitalEnv:
    def __init__(self, num_satellites=1, num_debris=3, max_steps=200,
                 collision_threshold=1.0, world_size=10.0,
                 min_spawn_dist=5.0, max_speed=3.0,
                 encounter_mode=True, num_threats=1, threat_miss_max=2.5):
        self.num_satellites = num_satellites
        self.num_debris = num_debris
        self.max_steps = max_steps
        self.collision_threshold = collision_threshold
        self.world_size = world_size
        self.min_spawn_dist = min_spawn_dist
        self.max_speed = max_speed
        self.encounter_mode = encounter_mode
        self.num_threats = num_threats
        self.threat_miss_max = threat_miss_max
        self.min_dist_seen = np.inf

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_far(low, high, avoid, min_dist, max_tries=200):
        p = np.random.uniform(low, high, size=2)
        for _ in range(max_tries):
            if all(np.linalg.norm(p - a) >= min_dist for a in avoid):
                return p
            p = np.random.uniform(low, high, size=2)
        return p

    def _make_threat(self, sat, max_tries=100):
        """Debris on a course that passes `sat` at a chosen miss distance
        at a chosen time of closest approach T (if nobody acts)."""
        for _ in range(max_tries):
            T = np.random.uniform(4.0, 10.0)
            speed = np.random.uniform(0.5, 1.5)
            ang = np.random.uniform(0.0, 2.0 * np.pi)
            v_d = speed * np.array([np.cos(ang), np.sin(ang)])
            w = v_d - sat.velocity
            wn = float(np.linalg.norm(w))
            if wn < 0.3:
                continue
            perp = np.array([-w[1], w[0]]) / wn
            b = np.random.uniform(0.0, self.threat_miss_max) * np.random.choice([-1.0, 1.0])
            target = sat.position + sat.velocity * T + perp * b   # debris position at time T
            pos = target - v_d * T
            if np.linalg.norm(pos - sat.position) >= self.min_spawn_dist:
                return Satellite(position=pos, velocity=v_d)
        return Satellite(position=sat.position + np.array([8.0, 0.0]),
                         velocity=np.array([-1.0, 0.0]))

    def reset(self):
        sat_positions = []
        for _ in range(self.num_satellites):
            sat_positions.append(
                self._sample_far(-2, 2, sat_positions, self.min_spawn_dist / 2)
            )
        self.satellites = [
            Satellite(position=p, velocity=np.random.uniform(-0.5, 0.5, size=2))
            for p in sat_positions
        ]

        self.debris = []
        for k in range(self.num_debris):
            if self.encounter_mode and k < self.num_threats:
                target_sat = self.satellites[k % self.num_satellites]
                self.debris.append(self._make_threat(target_sat))
            else:
                self.debris.append(Satellite(
                    position=self._sample_far(-self.world_size, self.world_size,
                                              sat_positions, self.min_spawn_dist),
                    velocity=np.random.uniform(-1, 1, size=2),
                ))

        self.alive = [True] * self.num_satellites
        self.current_step = 0
        self.min_dist_seen = np.inf
        return self._get_all_obs()

    # ------------------------------------------------------------------
    def _get_obs_for(self, i):
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

        # conjunction-style features, one group of 4 per debris:
        #   [predicted relative position at closest approach (2),
        #    time of closest approach in steps (capped at 8),
        #    predicted miss distance (capped at 10)]
        for d in self.debris:
            r = d.position - sat.position
            w = d.velocity - sat.velocity
            denom = float(np.dot(w, w))
            t = 0.0 if denom < 1e-9 else float(np.clip(-np.dot(r, w) / denom, 0.0, 8.0))
            m = r + w * t
            obs.append(m)
            obs.append(np.array([t, min(float(np.linalg.norm(m)), 10.0)]))

        return np.concatenate(obs)

    def _get_all_obs(self):
        return [self._get_obs_for(i) for i in range(self.num_satellites)]

    def obs_dim(self):
        return 4 + 4 * self.num_debris + 4 * (self.num_satellites - 1) + 4 * self.num_debris

    # ------------------------------------------------------------------
    def _all_receding(self):
        """True when no alive satellite is approaching any debris or any
        other alive satellite (the encounter is over)."""
        for i, sat in enumerate(self.satellites):
            if not self.alive[i]:
                continue
            for d in self.debris:
                if np.dot(d.position - sat.position, d.velocity - sat.velocity) < 0:
                    return False
            for j in range(i + 1, self.num_satellites):
                if self.alive[j]:
                    other = self.satellites[j]
                    if np.dot(other.position - sat.position,
                              other.velocity - sat.velocity) < 0:
                        return False
        return True

    def step(self, actions):
        collided_this_step = [False] * self.num_satellites

        sat_old = [s.position.copy() for s in self.satellites]
        deb_old = [d.position.copy() for d in self.debris]

        for i, sat in enumerate(self.satellites):
            if not self.alive[i]:
                continue
            action = np.clip(np.array(actions[i], dtype=float), -1.0, 1.0)
            sat.velocity += action
            speed = np.linalg.norm(sat.velocity)
            if speed > self.max_speed:
                sat.velocity *= self.max_speed / speed
            sat.step()

        for d in self.debris:
            d.step()

        self.current_step += 1

        # collision checks: satellite-debris (swept)
        sat_min_dist = [np.inf] * self.num_satellites
        for i, sat in enumerate(self.satellites):
            if not self.alive[i]:
                continue
            for k, d in enumerate(self.debris):
                dist = min_dist_swept(sat_old[i], sat.position, deb_old[k], d.position)
                sat_min_dist[i] = min(sat_min_dist[i], dist)
                self.min_dist_seen = min(self.min_dist_seen, dist)
                if dist < self.collision_threshold:
                    collided_this_step[i] = True

        # collision checks: satellite-satellite (swept)
        for i in range(self.num_satellites):
            if not self.alive[i]:
                continue
            for j in range(i + 1, self.num_satellites):
                if not self.alive[j]:
                    continue
                dist = min_dist_swept(sat_old[i], self.satellites[i].position,
                                      sat_old[j], self.satellites[j].position)
                if dist < self.collision_threshold:
                    collided_this_step[i] = True
                    collided_this_step[j] = True

        # rewards: fuel cost + proximity warning + big collision penalty
        rewards = []
        for i in range(self.num_satellites):
            if not self.alive[i]:
                rewards.append(0.0)
                continue
            fuel_cost = 0.1 * np.linalg.norm(actions[i])
            prox = 0.0
            if sat_min_dist[i] < 4.0:
                prox = 1.0 * (1.0 - sat_min_dist[i] / 4.0)
            step_penalty = -0.01 - fuel_cost - prox
            if collided_this_step[i]:
                rewards.append(-100.0)
                self.alive[i] = False
            else:
                rewards.append(step_penalty)

        timed_out = self.current_step >= self.max_steps
        done = timed_out or not any(self.alive)
        encounter_over = False
        if self.encounter_mode and not done and self._all_receding():
            done = True
            encounter_over = True

        obs = self._get_all_obs()
        info = {"collided": collided_this_step, "timed_out": timed_out,
                "encounter_over": encounter_over}
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

    for step in range(15):
        actions = [np.zeros(2) for _ in range(env.num_satellites)]
        obs, rewards, done, info = env.step(actions)
        print(f"Step {step}: reward={rewards[0]:.2f}, done={done}, info={info}")
        if done:
            break