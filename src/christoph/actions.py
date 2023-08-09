import numpy as np

# Possible actions for the Hockey game.
# The ones not commented out were used for the final agent
ACTIONS = [
    np.array([1, 0, 0, 0]),
    np.array([-1, 0, 0, 0]),
    np.array([0, 1, 0, 0]),
    np.array([0, -1, 0, 0]),

    np.array([0, 0, 1, 0]),
    np.array([0, 0, -1, 0]),

    #np.array([1, 0, 1, 1]),
    #np.array([-1, 0, 1, 1]),
    #np.array([0, 1, 1, 1]),
    #np.array([0, -1, 1, 1]),

    #np.array([1, 0, -1, 1]),
    #np.array([-1, 0, -1, 1]),
    #np.array([0, 1, -1, 1]),
    #np.array([0, -1, -1, 1]),

    #np.array([1, 0, 1, 0]),
    #np.array([-1, 0, 1, 0]),
    #np.array([0, 1, 1, 0]),
    #np.array([0, -1, 1, 0]),

    #np.array([1, 0, -1, 0]),
    #np.array([-1, 0, -1, 0]),
    #np.array([0, 1, -1, 0]),
    #np.array([0, -1, -1, 0]),

    np.array([0, 0, 0, 1]),
    #np.array([0, 0, 0, 0]),

]