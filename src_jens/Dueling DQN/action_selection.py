import numpy as np
import torch as th

from utils import ACTIONS

class ActionSelection():
    def __init__(self):
        pass

    def select_action(self, obs, eps, net, evaluation=False):
        if np.random.uniform(0, 1) > eps or evaluation:
            net.eval()
            with th.no_grad():
                inp = th.from_numpy(obs).float()[None]
                q_values = net(inp)[0]
                action_idx = q_values.argmax().item()
        else:
            action_idx = np.random.randint(0, len(ACTIONS))

        return ACTIONS[action_idx], action_idx