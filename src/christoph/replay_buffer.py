import random
import torch
import numpy as np

class ReplayBuffer():

    def __init__(self, max_size):
        self.max_size = max_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.idx = 0

    def push(self, state, action, reward, next_state, done):

        if type(state) != np.ndarray:
            state = np.array(state[0])
        if type(next_state) != np.ndarray:
            next_state = np.array(next_state[0])

        if len(self.states) < self.max_size:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
        else:
            self.states[self.idx] = state
            self.actions[self.idx] = action
            self.rewards[self.idx] = reward
            self.next_states[self.idx] = next_state
            self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.max_size

    def sample(self, batch_size, device):
        idx_to_sample = random.sample(range(len(self.states)), k=batch_size)

        states = torch.from_numpy(np.array(self.states)[idx_to_sample]).float().to(device)
        actions = torch.from_numpy(np.array(self.actions)[idx_to_sample]).to(device)
        rewards = torch.from_numpy(np.array(self.rewards)[idx_to_sample]).float().to(device)
        next_states = torch.from_numpy(np.array(self.next_states)[idx_to_sample]).float().to(device)
        dones = torch.from_numpy(np.array(self.dones)[idx_to_sample]).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.states)
    