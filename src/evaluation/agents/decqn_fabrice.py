import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import numpy as np
import torch
from agents.agent import Agent

from src.fabrice.src.rl.agent import Agent as DecQNAgent
from src.fabrice.src.rl.critic import Critic

# DO NOT CHANGE:
STATE_DIM = 18
ACTION_DIM = 4
MAX_ABS_FORCE = 1.0
MAX_ABS_TORQUE = 1.0
DEVICE = torch.device("cpu")

HIDDEN_DIM = 256
DISCRETIZATION_DIM = 3
CHECKPOINT = "agents/decqn_fabrice.pt"


class DecQN_Fabrice(Agent):
    def __init__(self) -> None:
        q_model = Critic(STATE_DIM, HIDDEN_DIM, ACTION_DIM, DISCRETIZATION_DIM)
        q_model.load(CHECKPOINT)
        self.agent = DecQNAgent(q_model, DISCRETIZATION_DIM, MAX_ABS_FORCE, MAX_ABS_TORQUE, DEVICE)

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Acts in environment, yielding an action, given a state.
        S: State dimension.
        A: Action dimension.

        Args:
            state (np.ndarray): State [S].

        Raises:
            NotImplementedError: Needs to be implemented for a concrete agent.

        Returns:
            np.ndarray: Action (continuous) [A].
        """

        return self.agent.act(state, eval_=True)[0]
