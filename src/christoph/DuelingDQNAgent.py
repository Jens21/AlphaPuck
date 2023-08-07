import torch
import numpy as np
import random

from replay_buffer import ReplayBuffer
from network import DuelingDQN

from actions import ACTIONS

from time import sleep

class DuelingDQNAgent():
    def __init__(self, observation_dim, action_dim, device, epsilon_max, epsilon_min, epsilon_decay, buffer_max_size = 1_000_000, discount = 0.99, lr=1e-4, double=True):

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.discount = discount
        self.device = device
        self.double = double

        self.epsilon = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.replay_buffer = ReplayBuffer(buffer_max_size)

        # for lunar lander
        #self.policy_net = DuelingDQN(self.observation_dim.shape[0], self.action_dim.n, lr).to(self.device)
        #self.target_net = DuelingDQN(self.observation_dim.shape[0], self.action_dim.n, lr).to(self.device)

        # for hockey
        self.policy_net = DuelingDQN(self.observation_dim, self.action_dim, lr).to(self.device)
        self.target_net = DuelingDQN(self.observation_dim, self.action_dim, lr).to(self.device)

        self.target_net.eval()
        self.update_target_net()



    def select_action(self, state):
        rn = random.random()

        if rn < self.epsilon:
            # for lunar lander
            #return random.randint(0, self.action_dim.n - 1)

            # for hockey
            idx = random.randint(0, self.action_dim - 1)
            return ACTIONS[random.randint(0, self.action_dim - 1)], idx
        
        if not torch.is_tensor(state):
            if type(state) == tuple:
                state = np.array(state[0])
            else:
                state = np.array(state)           
            state = torch.from_numpy(state)
            state = state.type(torch.float).to(self.device)
        with torch.no_grad():
            action = torch.argmax(self.policy_net(state))
        # for lunar lander
        #return action.item()
        # for hockey
        return ACTIONS[action.item()], action.item()
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def learn(self, batchsize):
        if len(self.replay_buffer) < batchsize:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batchsize, self.device)

        actions = actions.reshape((-1,1))
        rewards = rewards.reshape((-1,1))
        dones = dones.reshape((-1,1))
        


         
        with torch.no_grad():
            if self.double:
                policy_target_qs = self.policy_net(next_states)
                policy_target_qs_idx = policy_target_qs.argmax(dim=1, keepdim=True)

                target_target_qs = self.target_net(next_states)
                target_target_qs = torch.gather(input=target_target_qs, dim=1, index=policy_target_qs_idx)
                target_target_qs = target_target_qs.reshape(-1,1)
                target_target_qs[dones] = 0.0

                target = rewards + (self.discount * target_target_qs)
                
            else:
                target_qs = self.target_net(next_states)
                target_qs = torch.max(target_qs, dim=1).values
                target_qs = target_qs.reshape(-1,1)
                target_qs[dones] = 0.0
                target = rewards + (self.discount * target_qs)
        

        predict_qs = self.policy_net(states)
        # for lunar lander
        #predict_qs = predict_qs.gather(1, actions.type(torch.int64))
        predict_qs = predict_qs.gather(1, actions.type(torch.int64))

        loss = torch.nn.functional.mse_loss(predict_qs, target)
        self.policy_net.optimizer.zero_grad()
        loss.backward()
        self.policy_net.optimizer.step()

    def save_network(self, filename):
        try:
            print("saving network")
            torch.save(self.policy_net.state_dict(), filename)
            return
        except:
            print("saving failed")

        sleep(5)

        try:
            print("saving network")
            torch.save(self.policy_net.state_dict(), filename)
            return
        except:
            print("second saving failed")

    def load_network(self, filename):
        print("loading network")
        self.policy_net.load_state_dict(torch.load(filename))
        self.policy_net.eval()