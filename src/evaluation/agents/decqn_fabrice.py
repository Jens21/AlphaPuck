import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../fabrice/src/"))
import numpy as np
import torch
from agents.agent import Agent

from src.fabrice.src.cli.utils import get_env_parameters
from src.fabrice.src.rl.agent import Agent as DecQNAgent
from src.fabrice.src.rl.critic import Critic

## Model parameters
HIDDEN_DIM = 256
DISCRETIZATION_DIM = 3
MAX_ABS_FORCE = 1.0
MAX_ABS_TORQUE = 1.0
NO_STATE_NORM = False
CHECKPOINT = "agents/decqn_fabrice.pt"


class DecQN_Fabrice(Agent):
    def __init__(self) -> None:
        state_dim, action_dim, w, h, vel, ang, ang_vel, vel_puck, t = get_env_parameters()
        device = torch.device("cpu")
        q_model = Critic(
            state_dim,
            HIDDEN_DIM,
            action_dim,
            DISCRETIZATION_DIM,
            NO_STATE_NORM,
            w,
            h,
            vel,
            ang,
            ang_vel,
            vel_puck,
            t,
            device,
        )
        q_model.load(CHECKPOINT)
        self.agent = DecQNAgent(q_model, DISCRETIZATION_DIM, MAX_ABS_FORCE, MAX_ABS_TORQUE, device)

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Acts in environment, yielding an action, given a state.
        S: State dimension.
        A: Action dimension.

        Args:
            state (np.ndarray): State [S].

        Returns:
            np.ndarray: Action (continuous) [A].
        """

        return self.agent.act(state, eval_=True)[0]
