import os
import gymnasium
import numpy as np
import torch

from DuelingDQNAgent import DuelingDQNAgent

def train(env, agent, train_episodes, buffer_fill_episodes, batchsize, update_freq, model_filename):
    fill_buffer(env, agent, buffer_fill_episodes)
    print("Samples in buffer: ", len(agent.replay_buffer))

    step_count = 0
    reward_history = []
    prev_avg_score = -np.inf

    for ep_count in range(train_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            agent.learn(batchsize)

            if step_count % update_freq == 0:
                agent.update_target_net()

            state = next_state
            ep_reward += reward
            step_count += 1

        agent.update_epsilon()
        reward_history.append(ep_reward)

        current_avg_score = np.mean(reward_history[-100:])

        print("Ep: {}, Total Steps: {}, Ep Score: {}, Avg score: {}, Updated Epsilon: {}".format(ep_count, step_count, ep_reward, current_avg_score, agent.epsilon))

        if current_avg_score >= prev_avg_score:
            agent.save_network(model_filename)
            prev_avg_score = current_avg_score




def fill_buffer(env, agent, buffer_fill_episodes):
    for _ in range(buffer_fill_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            state=next_state


def test(env, agent, test_episodes):
    
    for ep_count in range(test_episodes):
        state = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            ep_reward += reward

        print('Ep: {}, Ep score: {}'.format(ep_count, ep_reward))


def set_seed(env, seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    #env.seed(seed_value)
    #env.action_space.np_random.seed(seed_value)

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = True

    env = gymnasium.make("LunarLander-v2")
    model_filename = "ll_online_net"

    if train_mode:
        set_seed(env, 0)

        agent = DuelingDQNAgent(observation_dim=env.observation_space,
                                action_dim=env.action_space,
                                device=device,
                                epsilon_max=1.0,
                                epsilon_min=0.01,
                                epsilon_decay=0.995,
                                buffer_max_size=10000,
                                discount=0.99,
                                lr=1e-3)
        
        train(env=env,
              agent=agent,
              train_episodes=2000,
              buffer_fill_episodes=20,
              batchsize=64,
              update_freq=1000,
              model_filename=model_filename)
        
    else:
        set_seed(env, 10)

        agent = DuelingDQNAgent(observation_dim=env.observation_space,
                                action_dim=env.action_space,
                                device=device,
                                epsilon_max=0.0,
                                epsilon_min=0.0,
                                epsilon_decay=0.0,
                                buffer_max_size=10000,
                                discount=0.99,
                                lr=1e-3)
        
        agent.load_network(model_filename)

        test(env=env, agent=agent, test_episodes=100)