from utils import ACTIONS_T
from network import Network
from evaluator import Evaluator

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch as th
import os
import ray
import time

# Design of separating self play actors, replay buffer, shared memory and the train actor
# is derived from EfficientZero: https://arxiv.org/pdf/2111.00210.pdf but without the reanalyzing, context queue and batch queue
@ray.remote
class Trainer():
    def __init__(self, replay_buffer, batch_size, gamma, K, unroll_steps, shared_memory, T_max_scheduler):
        self.replay_buffer = replay_buffer
        self.writer = SummaryWriter()
        self.batch_size = batch_size
        self.gamma = gamma
        self.K = K
        self.unroll_steps = unroll_steps
        self.gammas = self.gamma ** th.arange(self.K)[None, :] # [1, K]
        self.shared_memory = shared_memory

        self.net_policy = Network()
        self.net_target = Network()
        self.net_target.load_state_dict(self.net_policy.state_dict())
        self.shared_memory.set_current_model.remote(self.net_policy.state_dict())

        self.optimizer = th.optim.Adam(self.net_policy.parameters(), lr=6e-4)
        self.scheduler = th.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max_scheduler)

        self.evaluator = Evaluator(gamma, shared_memory, 10)

        self.training_steps = 0

    # return the number of training steps that were done
    def get_training_steps(self):
        return self.training_steps

    # record losses to tensorboard once in a while to tensorboard
    def record_losses(self, total_loss, state_value_loss, dynamic_loss, reward_loss, policy_loss):
        if self.training_steps % 250 == 0:
            self.writer.add_scalar('total loss', total_loss, self.training_steps)
            self.writer.add_scalar('state value loss', state_value_loss, self.training_steps)
            self.writer.add_scalar('dynamic loss', dynamic_loss, self.training_steps)
            self.writer.add_scalar('reward loss', reward_loss, self.training_steps)
            self.writer.add_scalar('policy loss', policy_loss, self.training_steps)

    # returns the current lr
    def get_last_lr(self):
        return self.scheduler.get_last_lr()[0]

    # saves the networks performance once in a while to tensorboard
    def save_network(self):
        if self.training_steps % 12_500 == 0:
            n_won_weak, n_won_strong, n_draw_weak, n_draw_strong, n_lost_weak, n_lost_strong, _, _, net = self.evaluator.evaluate()
            self.writer.add_scalar('won weak', n_won_weak, self.training_steps)
            self.writer.add_scalar('draw weak', n_draw_weak, self.training_steps)
            self.writer.add_scalar('lost weak', n_lost_weak, self.training_steps)

            self.writer.add_scalar('won strong', n_won_strong, self.training_steps)
            self.writer.add_scalar('draw strong', n_draw_strong, self.training_steps)
            self.writer.add_scalar('lost strong', n_lost_strong, self.training_steps)

            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            th.save(net.state_dict(), 'checkpoints/agent_{}.pth'.format(self.training_steps))
        elif self.training_steps % 2_500 == 0:
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')

            th.save(ray.get(self.shared_memory.get_current_model.remote()), 'checkpoints/agent_{}.pth'.format(self.training_steps))

    # this functions is currently not used but convertes the samples from the replay buffer back to its individual components
    def convert_samples_back(self, samples):
        # [current_obs, action_indices, reward, value_prefix, value_mask, value_obs, similarity_obs]

        current_obs = samples[:, :18]
        action_indices = samples[:, 18].long()
        reward = samples[:, 19]
        value_prefix = samples[:, 20]
        value_mask = samples[:, 21]
        value_obs = samples[:, 22:40]
        similarity_obs = samples[:, 40:]

        return current_obs, action_indices, reward, value_prefix, value_mask, value_obs, similarity_obs

    # scale the gradients like in the MuZero paper
    def scale_gradient(self, tensor, scale):
        """Scales the gradient for the backward pass."""
        return tensor * scale + tensor.detach() * (1. - scale)

    # training loop
    # runs in parallel until training is done
    def train(self):
        while True:
            samples = ray.get(self.replay_buffer.sample.remote(self.batch_size))
            if samples is None: # not enough samples in the replay buffer, sample more
                time.sleep(0.1)
                continue

            observations, action_indices_1, action_indices_2, rewards, policy_distr_trg, dones, _, _, total_frames = samples
            actions_1 = ACTIONS_T[action_indices_1]
            actions_2 = ACTIONS_T[action_indices_2]
            # observations: [128, 6, 18]
            # action_indices: [128, 6]
            # actions: [128, 6, 4]
            # rewards: [128, 6]
            # policy_distr: [128, 6, 7]
            # dones: [128, 6]

            self.net_policy.train()
            self.net_target.eval()

            # target state values and latent states
            with th.no_grad():
                latent_states_trg, state_values_trg, _ = self.net_target.initial_inference(observations.flatten(end_dim=1))
                latent_states_trg, state_values_trg = latent_states_trg.reshape(observations.shape[0], observations.shape[1], -1), state_values_trg.reshape(observations.shape[:2])
                state_values_trg = (self.gamma**self.K) * state_values_trg # [128, 6]

            # Initial step, from the real observation.
            latent_states, esti_state_values, policy_logits = self.net_policy.initial_inference(observations[:, 0])
            predictions = [(1.0, esti_state_values, 0, policy_logits, None)]

            # Recurrent steps, from action and previous hidden state.
            for i in range(self.unroll_steps):
                latent_states, _, rewards_logits, esti_state_values, policy_logits = self.net_policy.recurrent_inference(latent_states, actions_1[:, i+1], actions_2[:, i+1])

                latent_states = self.scale_gradient(latent_states, 0.5)
                predictions.append((1.0 / self.unroll_steps, esti_state_values, rewards_logits, policy_logits, latent_states))

            # compute the losses like in the MuZero pseudocode
            total_loss, state_value_loss, dynamic_loss, reward_loss, policy_loss = 0, 0, 0, 0, 0
            updated_priorities = 0
            for i, prediction in enumerate(predictions):
                gradient_scale, value, reward_logits, policy_logits, latent_states = prediction

                value_prefix = (rewards[:, i + 1:i + 1 + self.K] * self.gammas).sum(dim=1)  # [128]
                value_trg = value_prefix + state_values_trg[:, i+self.K] * (1-dones[:, i+self.K]) # [128]

                l4 = F.cross_entropy(policy_logits, policy_distr_trg[:, i] * (1-dones[:, i, None]), reduction='none')
                policy_loss = policy_loss + l4
                l1 = F.smooth_l1_loss(value, value_trg.clip(-1+1e-4,1-1e-4), reduction='none')
                updated_priorities = updated_priorities + F.l1_loss(value, value_trg.clip(-1+1e-4,1-1e-4), reduction='none')
                state_value_loss = state_value_loss + l1

                if i>0:
                    l2 = F.cross_entropy(reward_logits, (rewards[:, i] + 1).long(), reduction='none')
                    reward_loss = reward_loss + l2

                    l3 = -F.cosine_similarity(latent_states, latent_states_trg[:, i], eps=1e-5)
                    dynamic_loss = dynamic_loss + l3
                else:
                    l2, l3 = 0, 0

                total_loss = total_loss + self.scale_gradient(0.25*l1 + 1*l2 + 2*l3 + 1*l4, gradient_scale)

            total_loss = total_loss.mean()

            self.optimizer.zero_grad()
            total_loss.backward()
            th.nn.utils.clip_grad_norm_(self.net_policy.parameters(), 5)
            self.optimizer.step()
            self.scheduler.step()

            self.training_steps += 1

            # share the network once in a while
            if self.training_steps % 200 == 0:
                self.net_target.load_state_dict(self.net_policy.state_dict())
                self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], self.training_steps)
                self.writer.add_scalar('total frames', total_frames, self.training_steps)

            if self.training_steps % 25 == 0:
                self.shared_memory.set_current_model.remote(self.net_policy.state_dict())

            self.save_network()

            # record the losses in tensorboard once in a while
            state_value_loss = 0.25 * state_value_loss
            dynamic_loss = 2 * dynamic_loss
            reward_loss = 1 * reward_loss
            policy_loss = 1 * policy_loss
            self.record_losses(total_loss, state_value_loss.mean().item(), dynamic_loss.mean().item(), reward_loss.mean().item(), policy_loss.mean().item())
