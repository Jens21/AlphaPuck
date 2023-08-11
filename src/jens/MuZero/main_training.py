from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from trainer import Trainer
from shared_memory import SharedMemory

import numpy as np
import torch as th
import random
import ray
import time

random.seed(51342)
np.random.seed(12347)
th.manual_seed(54323)

ray.init()

# actually the buffer size is slightly less due to circular buffer design
MAX_BUFFER_SIZE = 1_000_000
BATCH_SIZE = 768
N_WARMUP = 10_000

# MuZero used 0.997 for atari games but did not work here
GAMMA = 0.975

# K = 1 is minimum and stands for TD(0)
K = 5
# unroll_steps = 1 is minimum and unrolls the dynamic function only ones
unroll_steps = 3

# same like in MuZero Unplugged
T_max_scheduler = 1_000_000
# critical parameter since with larger value the trainer actor can't keep up
SELF_PLAY_WORKERS = 4

FRAME_SKIP = 2

if __name__ == '__main__':
    shared_memory = SharedMemory.remote()

    replay_buffer = ReplayBuffer.remote(MAX_BUFFER_SIZE, K, unroll_steps, N_WARMUP)

    trainer = Trainer.remote(replay_buffer, BATCH_SIZE, GAMMA, K, unroll_steps, shared_memory, T_max_scheduler)

    trainer.train.remote()
    time.sleep(5) # to ensure shared_memory has the current network parameters and the trainer actor started correctly

    # start all self play actors
    self_plays = [SelfPlay.remote(replay_buffer, GAMMA, K, unroll_steps, shared_memory, FRAME_SKIP) for _ in range(SELF_PLAY_WORKERS)]
    for self_play in self_plays:
        self_play.do_self_play.remote()

    # test if training is over
    while ray.get(trainer.get_training_steps.remote()) != T_max_scheduler:
        time.sleep(100)

    ray.shutdown()
