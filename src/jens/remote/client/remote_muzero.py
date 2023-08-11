# elif player == "Jens_final_fix_multi_proc_2":
# return EfficientZero_Jens_Final_Fix_Multi_Proc_2(
#     'agents/agent_final_fix_multi_proc.pth')  # Winning percentage: 6.5%  ################### Best agent in total, but bad vs strong
# elif player == "test_v1":
# return EfficientZero_Jens_Final_Single_Proc_2(
#     'agents/agent_1062500.pth')  # Winning percentage: # Winning percentage: 2.0%

import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'client'))
sys.path.append(os.getcwd())

import numpy as np

from laserhockey.hockey_env import BasicOpponent
from client.remoteControllerInterface import RemoteControllerInterface
from client.backend.client import Client

from network_fabrice import EfficientZero_Jens_Final_Single_Proc_2
from network_best import EfficientZero_Jens_Final_Fix_Multi_Proc_2

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
        if os.path.exists('client/network_version.txt'):
            with open('client/network_version.txt', 'r') as f:
                content = f.read()

            content = "network_v10"

            if content == 'network_fabrice':
                self.net = EfficientZero_Jens_Final_Single_Proc_2('client/agent_1062500.pth')
                print('loaded network_fabrice')
            elif content == 'network_v10': # network_best best v10
                self.net = EfficientZero_Jens_Final_Fix_Multi_Proc_2('client/agent_1025000.pth')
                print('network_best best v10')
            else: # network_best
                self.net = EfficientZero_Jens_Final_Fix_Multi_Proc_2('client/agent_final_fix_multi_proc.pth')
                print('loaded network_best')
        else:
            print("Can't load the network_version file")

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