import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'client'))
sys.path.append(os.getcwd())

import numpy as np

from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

from network import MuZero

class RemoteBasicOpponent(RemoteControllerInterface):
    def __init__(self, weak, keep_mode=True):
        # BasicOpponent.__init__(self, weak=weak, keep_mode=keep_mode)
        RemoteControllerInterface.__init__(self, identifier='MuZero')

        self.before_game_starts()

    def remote_act(self,
                   obs: np.ndarray,
                   ) -> np.ndarray:

        action = self.net.act(obs)

        return action

    def before_game_starts(self) -> None:
        self.net = MuZero('muzero.pth')

if __name__ == '__main__':
    controller = RemoteBasicOpponent(weak=False)

    # Play n (None for an infinite amount) games and quit
    client = Client(username='AlphaPuck',  # Testuser
                    password='Ooxai2aeng',
                    controller=controller,
                    output_path='logs',
                    # rollout buffer with finished games will be saved in here
                    interactive=False,
                    op='start_queuing',
                    num_games=None)

    # # Start interactive mode. Start playing by typing start_queuing. Stop playing by pressing escape and typing stop_queueing
    # client = Client(username='AlphaPuck',
    #                 password='Ooxai2aeng',
    #                 controller=controller,
    #                 output_path='logs',
    #                )