import numpy as np
import torch as th

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

ACTIONS_T = th.from_numpy(np.array(ACTIONS))