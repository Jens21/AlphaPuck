import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../christoph/"))
import torch
from agents.agent import Agent

from src.christoph.DuelingDQNAgent import DuelingDQNAgent

CHECKPOINT = "agents/dddqn_christoph.pt"


class DDDQN_Christoph(Agent):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.agent = DuelingDQNAgent(
            observation_dim=18,
            action_dim=7,
            device=device,
            epsilon_max=0.0,
            epsilon_min=0.0,
            epsilon_decay=0.0,
            discount=0.99,
            lr=1e-3,
            double=True,
        )

        self.agent.load_network(CHECKPOINT)

    def act(
        self,
        obs: np.ndarray,
    ) -> np.ndarray:
        action, _ = self.agent.select_action(obs)
        return action
