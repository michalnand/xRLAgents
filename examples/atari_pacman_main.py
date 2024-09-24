import rl_libs
from WrapperAtari import *

if __name__ == "__main__":
    # num of paralel environments
    n_envs   = 128

    # environment name
    env_name = "MsPacmanNoFrameskip-v4"

    # create result path
    result_path = "results/" + env_name + "/"
    
    
    # create environments
    print("creating envs")
    envs = rl_libs.EnvsListParallel(env_name, n_envs, Wrapper=WrapperAtari)

    print("creating agent")
    # create agent
    agent = rl_libs.AgentPPO(envs, rl_libs.ModelCNN, gamma = 0.99, entropy_beta = 0.001, n_steps = 128, batch_size = 256)

    # run training
    print("starting training")
    trainer = rl_libs.RLTrainer(envs, agent, result_path)
    trainer.run(500000)
   


    '''
    # inference part
    envs = rl_libs.EnvsList(env_name, 1, 'human')
    states, _ = envs.reset()

    agent = rl_libs.AgentPPO(envs, rl_libs.ModelFC, result_path=result_path)
    agent.load()

    while True:
        states, rewards, dones, infos = agent.step(states, False)

        done_idx = numpy.where(dones)[0]
        for e in done_idx:
            states[e], _ = envs[e].reset()

        time.sleep(0.01)
    '''