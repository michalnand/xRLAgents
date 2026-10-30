import time
import xRLAgents
from WrapperRobotics import *
from agent_config import *


if __name__ == "__main__":
    # num of paralel environments
    n_envs   = 32

    # environment name
    env_name = "CartPole-v1"

    # create result path
    result_path = "result/"
    
    '''
    # agent training

    # create environments
    print("creating envs")
    envs = xRLAgents.EnvsListParallel(env_name, n_envs, Wrapper=WrapperRobotics)

    print("creating agent")
    # create agent
    agent = xRLAgents.AgentPPO(envs, xRLAgents.ModelFC, gamma = 0.99, entropy_beta = 0.001, n_steps = 128, batch_size = 256)

    # run training
    print("starting training")
    trainer = xRLAgents.RLTrainer(envs, agent, result_path)
    trainer.run(100000)
    '''
    

    
    # inference part
    envs = xRLAgents.EnvsList(env_name, 1, render_mode='human')
    states, _ = envs.reset()

    agent = xRLAgents.AgentPPO(envs, Config, xRLAgents.ModelFC)
    agent.load(result_path)

    while True:
        states, rewards, dones, infos = agent.step(states, False)

        done_idx = numpy.where(dones)[0]
        for e in done_idx:
            states[e], _ = envs[e].reset()

        time.sleep(0.01)
    