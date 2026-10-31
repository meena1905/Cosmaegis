"""
satellite.py
Basic moving-object class used for both satellites and debris.
Each object has a 2D position and velocity, and moves via simple
Euler integration: new_position = old_position + velocity * dt
"""

import numpy as np


class Satellite:
    def __init__(self, position, velocity):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)

    def step(self, dt=1.0):
        """Advance position by one timestep given current velocity."""
        self.position += self.velocity * dt


if __name__ == "__main__":
    # quick manual sanity check
    sat = Satellite(position=[0, 0], velocity=[1, 0.5])
    print("Start position:", sat.position)
    for i in range(5):
        sat.step()
        print(f"After step {i + 1}:", sat.position)
