import time
import xRLAgents    
from WrapperAtari import *
from agent_config import *


if __name__ == "__main__":
    # num of paralel environments
    n_envs   = 128

    # environment name
    env_name = "MsPacmanNoFrameskip-v4"

    # create result path
    result_path = "result/"
    
    '''
    # agent training

    # create environments
    print("creating envs")
    envs = xRLAgents.EnvsListParallel(env_name, n_envs, Wrapper=WrapperAtari)

    print("creating agent")
    # create agent
    agent = xRLAgents.AgentPPO(envs, Config, xRLAgents.ModelCNN)

    # run training
    print("starting training")
    trainer = xRLAgents.RLTrainer(envs, agent, result_path)
    trainer.run(500000)
    '''


    
    # inference part
    envs = xRLAgents.EnvsList(env_name, 1, 'human', WrapperAtari)
    states, _ = envs.reset()

    agent = xRLAgents.AgentPPO(envs, Config, xRLAgents.ModelCNN)
    agent.load(result_path)

    while True:
        states, rewards, dones, infos = agent.step(states, False)

        done_idx = numpy.where(dones)[0]
        for e in done_idx:
            states[e], _ = envs[e].reset()

        time.sleep(0.01)
    