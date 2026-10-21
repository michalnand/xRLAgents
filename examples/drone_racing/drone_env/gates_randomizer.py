"""Random racing courses: gates in flight order along a random path with sharp turns allowed.

A stacked pair (two touching gates, one exactly above the other, random which one first) works
as a direction reverser: the drone flies through the first, turns back through the second, and
the course continues back the way it came. A course can contain several reversers.
"""

from dataclasses import dataclass

import numpy as np

from .drone_constants import GATE_SIZE


@dataclass
class GatesRandomizer:
    num_gates: int = 8                  # gate positions along the path (each reverser adds one more gate)
    step_range: tuple = (2.0, 3.0)      # m, horizontal distance between consecutive gate positions
    max_turn_deg: float = 110.0         # largest heading change at a single gate
    reverser_turn_deg: float = 45.0     # largest heading change right after a reverser (clean out-and-back)
    height_range: tuple = (1.0, 2.5)    # m, gate centre height
    arena_radius: float = 5.0           # m, the course stays inside this circle around (0, 0)
    min_separation: float = 2.0         # m, minimum horizontal distance between different gate positions
    stacked_prob: float = 0.0           # per gate position: a reverser (stacked pair) instead of one gate

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Gate centres in flight order, shape (N, 3)."""
        low, high = self.height_range
        pos, heading = np.zeros(2), rng.uniform(-np.pi, np.pi)   # the drone starts at (0, 0)
        placed, gates, max_turn = [], [], self.max_turn_deg

        for _ in range(self.num_gates):
            pos, heading = self._next_position(rng, pos, heading, placed, max_turn)
            placed.append(pos)
            if rng.random() < self.stacked_prob:
                z = rng.uniform(low, max(low, high - GATE_SIZE))
                pair = [[*pos, z], [*pos, z + GATE_SIZE]]           # frames touch
                gates += pair if rng.random() < 0.5 else pair[::-1]  # lower or upper first
                heading += np.pi                                     # reverser: come back
                max_turn = self.reverser_turn_deg
            else:
                gates.append([*pos, rng.uniform(low, high)])
                max_turn = self.max_turn_deg
        return np.array(gates)

    def _next_position(self, rng, pos, heading, placed, max_turn_deg, tries=30):
        """Random turn + step that stays in the arena and away from earlier gates (best effort)."""
        max_turn = np.radians(max_turn_deg)
        best, best_score = None, -np.inf
        for _ in range(tries):
            h = heading + rng.uniform(-max_turn, max_turn)
            p = pos + rng.uniform(*self.step_range) * np.array([np.cos(h), np.sin(h)])
            clearance = min((np.linalg.norm(p - q) for q in placed), default=np.inf)
            score = min(clearance - self.min_separation, self.arena_radius - np.linalg.norm(p))
            if score >= 0:
                return p, h
            if score > best_score:
                best, best_score = (p, h), score
        return best