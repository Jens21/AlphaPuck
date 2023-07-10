from abc import ABC, abstractmethod

class Agent(ABC):

    @abstractmethod
    def act(self, obs):
        raise NotImplementedError('TODO')