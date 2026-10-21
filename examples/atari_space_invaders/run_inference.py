import xRLAgents
from config import *

import numpy

if __name__ == "__main__":
    print("initializing agent")
    config = Config(num_envs=1, render=True)

    result_path = config.result_path

    print("loading agent from ", result_path)
    #config.agent.load(result_path)

    states, _ = config.envs.reset_all()
    
    print("starting")

    
    while True:
        states, rewards, terminated, truncated, info = config.agent.step_inference(states)

        dones = truncated | terminated

        res = config.envs.render()
        