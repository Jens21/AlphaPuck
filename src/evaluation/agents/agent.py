import numpy as np


class Agent:
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

        raise NotImplementedError
