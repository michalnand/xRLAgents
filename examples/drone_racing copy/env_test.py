import numpy
import drone_env
from multi_drone_mujoco.control.pid_control import PIDControl

if __name__ == "__main__":
    # Demo: a PID pilot flies the course using only the observation.

    env = drone_env.RaceAviary(gates_randomizer=drone_env.GatesRandomizer(shuffle_prob=0.3, stacked_prob=0.3),
                     num_future_gates=2, render_mode="human")
    pid = PIDControl(env)
    obs, info = env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        pos, vel, ang_vel, to_gate = obs[0:3], obs[9:12], obs[12:15], obs[15:18]
        rpy = numpy.arctan2(obs[3:9:2], obs[4:9:2])
        dist = max(numpy.linalg.norm(to_gate), 1e-6)
        rpm, _, _ = pid.computeControl(env.CTRL_TIMESTEP, pos, drone_env.rpy_to_quat(rpy), vel, ang_vel,
                                       target_pos=pos + to_gate * min(1, 1.5 / dist),
                                       target_vel=to_gate / dist * 0.5)
        obs, reward, terminated, truncated, info = env.step(drone_env.rpm_to_action(rpm, env.HOVER_RPM, env.MAX_RPM))
        env.render()
        if env.render_mode is None:  # window closed with q
            break
    env.close()
    print(info)