import torch as th

from utils import ACTIONS

class Network(th.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = th.nn.Sequential(
            th.nn.Linear(18, 128),
            th.nn.ReLU(),
            th.nn.Linear(128, 64),
            th.nn.ReLU(),
            th.nn.Linear(64, len(ACTIONS)+1),
            # th.nn.Tanh()
        )

    def forward(self, x):
        net_out = self.model(x)
        state_value = net_out[:, 0]
        advantages = net_out[:, 1:]

        q_values = state_value[:, None] + advantages - advantages.mean(dim=1)[:, None]

        return q_values