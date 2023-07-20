import numpy as np
import torch as th
from agents.agent import Agent

ACTIONS = [
    np.array([1, 0, 0, 1]),
    np.array([-1, 0, 0, 1]),
    np.array([0, 1, 0, 1]),
    np.array([0, -1, 0, 1]),
    np.array([1, 0, 1, 1]),
    np.array([-1, 0, 1, 1]),
    np.array([0, 1, 1, 1]),
    np.array([0, -1, 1, 1]),
    np.array([1, 0, -1, 1]),
    np.array([-1, 0, -1, 1]),
    np.array([0, 1, -1, 1]),
    np.array([0, -1, -1, 1]),
    np.array([0, 0, 0, 1]),
]


class Network(th.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = th.nn.Sequential(
            th.nn.Linear(18, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, len(ACTIONS) + 1),
        )

        self.mean = th.FloatTensor([[-2.07, 0, 0, 0, 0, 0, 2.07, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.std = th.FloatTensor(
            [[1.57, 2.91, 1.04, 4, 4, 6, 1.57, 2.91, 1.04, 4, 4, 6, 3.7, 3, 12, 12, 15, 15]]
        )

    def forward(self, x):
        x = (x - self.mean) / self.std

        net_out = self.model(x)
        state_value = net_out[:, 0]
        advantages = net_out[:, 1:]

        q_values = state_value[:, None] + advantages - advantages.mean(dim=1)[:, None]

        return q_values


class Dueling_Jens(Agent):
    def __init__(self):
        self.net = Network()
        self.net.load_state_dict(th.load('agents/dueling_jens.pth'))

    def act(self, obs):
        with th.no_grad():
            inp = th.from_numpy(obs).float()[None]
            q_values = self.net(inp)[0]
            action_idx = q_values.argmax().item()

            return ACTIONS[action_idx]
