import time
import numpy
import drone_env

import libs_rl

if __name__ == "__main__":
    num_envs    = 128
    envs        = libs_rl.vector_env.VectorEnv(num_envs, drone_env.RaceAviary, gates_randomizer=drone_env.GatesRandomizer(stacked_prob=0.3), num_future_gates=2)

    print("num envs     : ", num_envs)
    print("state shape  : ", envs.observation_shape)
    print("action shape : ", envs.action_shape)


    for i in range(num_envs):
        envs.reset(i)

    num_steps = 1000

    time_start = time.time()
    for step in range(num_steps):    
        actions = numpy.random.randn(num_envs, envs.action_shape[0])
        obs, reward, terminated, truncated, info = envs.step(actions)
        for i in numpy.flatnonzero(terminated | truncated):
            obs[i], _ = envs.reset(i)
    time_stop = time.time()

    print("obs shape        : ", obs.shape)
    print("reward shape     : ", reward.shape)
    print("terminated shape : ", terminated.shape)
    print("truncated shape  : ", truncated.shape)


    print("sps/env = ", round(num_envs*num_steps/(time_stop - time_start), 3))
    print("sps     = ", round(num_steps/(time_stop - time_start), 3))

    logger = libs_rl.training_logger.TrainingLogger("result/")
    print("\n\n")
    print(envs.get_logs())

    #logger.add()