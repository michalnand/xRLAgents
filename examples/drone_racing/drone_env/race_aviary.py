"""Single-drone gate racing on top of MuJoCo-drones-gym's BaseAviary.

Action (4,) in [-1, 1]: motor speeds (-1 = off, 0 = hover, 1 = max).
Observation (35 + 12 * num_future_gates,), drone-centric:
    [0:4]    sin(roll), cos(roll), sin(pitch), cos(pitch)
    [4:7]    angular rates, body frame
    [7:9]    velocity over the ground: forward, left (heading frame)
    [9:11]   altitude z, vertical velocity vz
    [11:23]  target gate: 4 hole corners relative to the drone, body frame
    next     num_future_gates following gates, 12 values each
    last 12  nearest gate (obstacle), same format
See README.md for the full description.
"""

import gymnasium
import numpy as np
from gymnasium import spaces

from multi_drone_mujoco.envs.base_aviary import BaseAviary
from multi_drone_mujoco.utils.enums import DroneModel, Physics, ActionType, ObservationType

from .drone_constants import *
from .drone_render import DroneRenderer, CAMERAS
from .drone_utils import body_rotation, gate_yaws, gate_corners, to_gate_frame

__all__ = ["RaceAviary"]


class RaceAviary(BaseAviary):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, drone_model=DroneModel.CF2X, physics=Physics.MJC, sim_freq=250, ctrl_freq=50,
                 gates=None, gates_randomizer=None, num_future_gates=2,
                 render_mode=None, camera="third_person", trace=False):
        self.randomizer = gates_randomizer
        self.num_future_gates = num_future_gates
        self.camera = camera
        self.trace = trace
        self.viewer = None
        if gates_randomizer:
            gates = gates_randomizer.sample(np.random.default_rng())
        self._set_gates(DEFAULT_GATES if gates is None else gates)

        super().__init__(drone_model=drone_model, num_drones=1, physics=physics,
                         sim_freq=sim_freq, ctrl_freq=ctrl_freq, initial_xyzs=np.array([INITIAL_XYZ]),
                         obs_type=ObservationType.KIN, act_type=ActionType.RPM, render_mode=render_mode)

        self.observation_shape = self.observation_space.shape
        self.action_shape = self.action_space.shape

        # stats of the last finished episode, returned by get_log()
        self.metrics = {"num_episodes": 0, "num_steps": 0, "reward_episode": 0.0, "gates_passed_ratio": 0.0,
                    "success_lap": 0.0,
                    "speed_mean": 0.0, "speed_max": 0.0, "g_force_mean": 0.0, "g_force_max": 0.0,
                    "episode_done_time": 0, "episode_done_crash": 0}
        self._reset_state()

    # --- gym API ---------------------------------------------------------

    def reset(self, seed=None, options=None):
        gymnasium.Env.reset(self, seed=seed)  # seed self.np_random -> reproducible courses
        if self.randomizer:
            self._set_gates(self.randomizer.sample(self.np_random))
        if self.viewer:
            self.viewer.reset()
        self.gates_passed = 0                 # before super().reset(), which builds the first obs
        obs, info = super().reset(options=options)
        self._reset_state()
        return obs, info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        self._update_motion()

        self.ep_steps += 1
        self.ep_reward += reward
        self.ep_speed_sum += self.speed
        self.ep_speed_max = max(self.ep_speed_max, self.speed)
        self.ep_g_force_sum += self.g_force
        self.ep_g_force_max = max(self.ep_g_force_max, self.g_force)
        if terminated or truncated:
            self._write_log(crashed=terminated)

        # BaseAviary builds obs before the reward advances gates_passed; rebuild it.
        return self._computeObs(), reward, terminated, truncated, info

    def render(self, camera=None):
        """rgb_array: return the frame. human: OpenCV window, keys c = switch camera, q / Esc = quit."""
        if self.render_mode is None:
            return None
        if self.viewer is None:
            self.viewer = DroneRenderer(self.model, trace=self.trace)
        frame = self.viewer.render(self.data, self.pos[0], self.speed, self.gates, self.gate_yaw,
                                   self._upcoming(), camera or self.camera)
        if self.render_mode == "rgb_array":
            return frame

        # readout holds peaks and decays 1% per frame, so short spikes stay readable
        self.hud_speed = max(0.99 * self.hud_speed, self.speed)
        self.hud_g_force = max(0.99 * self.hud_g_force, self.g_force)
        key = self.viewer.show(frame, self.step_counter / self.SIM_FREQ, [
            (f"{self.hud_speed:4.1f} m/s   {self.hud_g_force:3.1f} g", 0.8),
            (f"gates {self.gates_passed}   {self.camera}", 0.55),
            ("[c] camera  [q] quit", 0.45)])

        if key == "c":
            self.camera = CAMERAS[(CAMERAS.index(self.camera) + 1) % len(CAMERAS)]
        elif key in ("q", "\x1b"):            # stop rendering, simulation keeps running
            self.viewer.close()
            self.viewer = None
            self.render_mode = None
            return -1
        return None

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        super().close()

    def get_metrics(self):
        """Stats of the last finished episode (see README.md)."""
        return dict(self.metrics)

    # --- episode state ---------------------------------------------------

    def _set_gates(self, gates):
        self.gates = np.array(gates, dtype=float)
        self.gate_yaw = gate_yaws(self.gates)
        self.gate_corners = gate_corners(self.gates, self.gate_yaw, GATE_HOLE / 2)   # (N, 4, 3) hole corners

    def _reset_state(self):
        self.gates_passed = 0
        self.hit_gate = False
        self.prev_pos = self.pos[0].copy()
        self.prev_vel = self.vel[0].copy()
        self.prev_potential = self._potential()
        self.speed = 0.0
        self.g_force = 1.0
        self.hud_speed = 0.0
        self.hud_g_force = 0.0

        self.ep_steps = 0
        self.ep_reward = 0.0
        self.ep_speed_sum = 0.0
        self.ep_speed_max = 0.0
        self.ep_g_force_sum = 0.0
        self.ep_g_force_max = 0.0

    def _update_motion(self):
        """Speed and g-force (specific force: acceleration minus gravity, 1 g when hovering)."""
        vel = self.vel[0].copy()
        acc = (vel - self.prev_vel) / self.CTRL_TIMESTEP
        self.prev_vel = vel
        self.speed = float(np.linalg.norm(vel))
        self.g_force = float(np.linalg.norm(acc + [0.0, 0.0, self.G]) / self.G)

    def _write_log(self, crashed):
        n = max(self.ep_steps, 1)
        self.metrics["num_episodes"] += 1
        self.metrics["num_steps"]           = self.ep_steps
        self.metrics["reward_episode"]      = self.ep_reward
        self.metrics["gates_passed_ratio"]  = self.gates_passed / len(self.gates)

        self.metrics["success_lap"] = int ( (self.gates_passed / len(self.gates)) > 0.999 )

        self.metrics["speed_mean"] = self.ep_speed_sum / n
        self.metrics["speed_max"] = self.ep_speed_max
        self.metrics["g_force_mean"] = self.ep_g_force_sum / n
        self.metrics["g_force_max"] = self.ep_g_force_max
        self.metrics["episode_done_time"] = int(not crashed)
        self.metrics["episode_done_crash"] = int(crashed)

    # --- gates -----------------------------------------------------------

    def _upcoming(self):
        """Indices of the target gate and the following num_future_gates (wrapping laps)."""
        return (self.gates_passed + np.arange(1 + self.num_future_gates)) % len(self.gates)

    def _nearest_gate(self):
        """Index of the gate whose centre is closest to the drone (any gate)."""
        return int(np.argmin(np.linalg.norm(self.gates - self.pos[0], axis=1)))

    def _update_gates(self):
        """Check all gates for frame contact (sets hit_gate); return True if the target gate was passed."""
        r = self.COLLISION_R
        a = to_gate_frame(self.prev_pos, self.gates, self.gate_yaw)
        b = to_gate_frame(self.pos[0], self.gates, self.gate_yaw)
        self.prev_pos = self.pos[0].copy()

        def inside(q, half):
            return (np.abs(q[:, 1]) < half) & (np.abs(q[:, 2]) < half)

        crossed = (a[:, 0] > 0) != (b[:, 0] > 0)                    # gate plane crossed this step
        t = a[:, 0] / np.where(crossed, a[:, 0] - b[:, 0], 1.0)
        x = a + (t * crossed)[:, None] * (b - a)                     # crossing point
        hole = GATE_HOLE / 2 - r
        outer = GATE_SIZE / 2 + r
        in_slab = np.abs(b[:, 0]) < GATE_DEPTH / 2 + r
        hit_crossing = crossed & inside(x, outer) & ~inside(x, hole)
        hit_touching = in_slab & inside(b, outer) & ~inside(b, hole)
        self.hit_gate = bool(np.any(hit_crossing | hit_touching))

        target = self._upcoming()[0]
        forward = a[target, 0] < 0 <= b[target, 0]                   # front side -> back side
        return bool(crossed[target] and inside(x, hole)[target] and (forward or not GATE_FORWARD_ONLY))

    # --- spaces / observation -------------------------------------------

    def _actionSpace(self):
        return spaces.Box(-1, 1, (4,), np.float32)

    def _observationSpace(self):
        return spaces.Box(-np.inf, np.inf, (35 + 12 * self.num_future_gates,), np.float32)

    def _preprocessAction(self, action):
        return self._normalizedActionToRPM(np.clip(action, -1, 1)).reshape(1, 4)

    def _computeObs(self):
        roll, pitch, yaw = self.rpy[0]
        vx, vy, vz = self.vel[0]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        # gate corners relative to the drone, rotated into the body frame: (K, 4, 3)
        gates = np.append(self._upcoming(), self._nearest_gate())
        corners = (self.gate_corners[gates] - self.pos[0]) @ body_rotation(self.quat[0])

        return np.concatenate([
            [np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch)],
            self.ang_v[0],                                           # body-frame rates (gyro)
            [cos_yaw * vx + sin_yaw * vy, -sin_yaw * vx + cos_yaw * vy],   # forward, left
            [self.pos[0, 2], vz],                                    # altitude, vertical velocity
            corners.ravel(),
        ]).astype(np.float32)

    # --- task ------------------------------------------------------------

    def _alignment(self):
        """0..1, 1 = drone on the target gate's axis: min/max of the distances to the 4 hole corners."""
        d = np.linalg.norm(self.gate_corners[self._upcoming()[0]] - self.pos[0], axis=1)
        return d.min() / d.max()

    def _potential(self):
        """Shaping potential for the target gate: closer and better aligned = higher."""
        dist = np.linalg.norm(self.gates[self._upcoming()[0]] - self.pos[0])
        return -PROGRESS * dist + ALIGN * self._alignment()

    def _computeReward(self):
        passed = self._update_gates()
        self.gates_passed += passed

        # shaping = change of the potential; after a pass it restarts from the new target gate
        potential = self._potential()
        reward = GATE_PASSED if passed else potential - self.prev_potential
        self.prev_potential = potential

        if self._computeTerminated():
            reward -= CRASH_PENALTY
        return float(reward)

    def _computeTerminated(self):
        crashed_floor = self.pos[0, 2] < LOW_ALTITUDE
        flipped = np.any(np.abs(self.rpy[0, :2]) > MAX_TILT)
        return bool(self.hit_gate or crashed_floor or flipped)

    def _computeTruncated(self):
        laps_done = MAX_LAPS is not None and self.gates_passed >= MAX_LAPS * len(self.gates)
        time_up = self.step_counter / self.SIM_FREQ >= EPISODE_LEN_SEC
        return bool(laps_done or time_up)

    def _computeInfo(self):
        return {"gates_passed": self.gates_passed, "num_gates": len(self.gates)}