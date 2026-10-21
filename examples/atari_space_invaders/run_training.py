import xRLAgents
from config import *


if __name__ == "__main__":
    config = Config()

    rl_trainer = xRLAgents.RLTraining(config)

    rl_trainer.run()


   