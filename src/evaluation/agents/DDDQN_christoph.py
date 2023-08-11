import random
import numpy as np

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../christoph/"))
from christoph.DuelingDQNAgent import DuelingDQNAgent

from laserhockey.hockey_env import BasicOpponent

import torch



from agent import Agent

class DDDQN_Christoph(Agent):

    def __init__(self):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DuelingDQNAgent(observation_dim=18,
                                action_dim=7,
                                device=device,
                                epsilon_max=0.0,
                                epsilon_min=0.0,
                                epsilon_decay=0.0,
                                discount=0.99,
                                lr=1e-3,
                                double=True)

        self.agent.load_network("final_network")


    def act(self, obs : np.ndarray,) -> np.ndarray:
        action, _ = self.agent.select_action(obs)
        return action
