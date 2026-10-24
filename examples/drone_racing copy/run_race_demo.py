import xRLAgents
from config import *

import numpy

if __name__ == "__main__":
    print("initializing agent")
    config = Config(num_envs=1, render=True)

    result_path = config.result_path

    print("loading agent from ", result_path)
    config.agent.load(result_path)

    states, _ = config.envs.reset(0)
    states = numpy.expand_dims(states, 0)

    print("starting")
    
    

    while True:
        states, rewards, terminated, truncated, info = config.agent.step_inference(states)

        dones = truncated | terminated

        if dones[0]:
            states, _ = config.envs.reset(0)
            states = numpy.expand_dims(states, 0)

        res = config.envs[0].render()

        if res == -1:
            break
        
