import numpy as np
import torch as th

# 25 actions, all translational movements + diagonal + rotations, one action for shooting
ACTIONS = [
    np.array([0, 0, 0, 1]),

    np.array([0, 1, 1, 0]),
    np.array([1, 1, 1, 0]),
    np.array([1, 0, 1, 0]),
    np.array([1, -1, 1, 0]),
    np.array([0, -1, 1, 0]),
    np.array([-1, -1, 1, 0]),
    np.array([-1, 0, 1, 0]),
    np.array([-1, 1, 1, 0]),

    np.array([0, 1, -1, 0]),
    np.array([1, 1, -1, 0]),
    np.array([1, 0, -1, 0]),
    np.array([1, -1, -1, 0]),
    np.array([0, -1, -1, 0]),
    np.array([-1, -1, -1, 0]),
    np.array([-1, 0, -1, 0]),
    np.array([-1, 1, -1, 0]),

    np.array([0, 1, 0, 0]),
    np.array([1, 1, 0, 0]),
    np.array([1, 0, 0, 0]),
    np.array([1, -1, 0, 0]),
    np.array([0, -1, 0, 0]),
    np.array([-1, -1, 0, 0]),
    np.array([-1, 0, 0, 0]),
    np.array([-1, 1, 0, 0]),
]

ACTIONS_T = th.from_numpy(np.array(ACTIONS))