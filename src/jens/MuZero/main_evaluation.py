from evaluator import Evaluator
from shared_memory import SharedMemory

import torch as th
import os
import numpy as np
import random

random.seed(51342)
np.random.seed(12347)
th.manual_seed(54323)

RENDER = False
GAMMA = 0.95
TREE_SEARCH_DEPTH = 5

def evaluate_single_checkpoint(checkpoint_path):
    shared_memory = SharedMemory()
    shared_memory.set_current_model(th.load(checkpoint_path))
    evaluator = Evaluator(gamma=GAMMA, shared_memory=shared_memory, tree_search_depth=TREE_SEARCH_DEPTH)

    n_won, n_draw, n_lost, n_games, net = evaluator.evaluate(RENDER)
    print('Won: {} ({} %)\nDraw: {} ({} %)\nLost: {} ({} %)'.format(n_won, 100 * n_won / n_games, n_draw,
                                                                    100 * n_draw / n_games, n_lost,
                                                                    100 * n_lost / n_games))

def evaluate_all_checkpoints(checkpoints_dir):
    files = os.listdir(checkpoints_dir)
    n_files = len(files)

    for i, file in enumerate(files):
        print('Checkpoint: {} ({:.2f} %)'.format(file, i/n_files*100))
        evaluate_single_checkpoint(os.path.join(checkpoints_dir, file))
        print()

if __name__ == '__main__':
    # evaluate_single_checkpoint('checkpoints/agent_3800000.pth')
    evaluate_all_checkpoints("checkpoints")