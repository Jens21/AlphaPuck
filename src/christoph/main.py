import os
import gymnasium
import numpy as np
import torch
import time

import tensorboard
from torch.utils.tensorboard import SummaryWriter

import laserhockey.hockey_env as h_env

from actions import ACTIONS

from DuelingDQNAgent import DuelingDQNAgent

def train(env, agent, train_episodes, buffer_fill_episodes, batchsize, update_freq, model_filename, weak_oponent):
    fill_buffer(env, agent, buffer_fill_episodes)
    print("Samples in buffer: ", len(agent.replay_buffer))

    writer = SummaryWriter()

    step_count = 0
    reward_history = []
    prev_avg_score = -np.inf




    for ep_count in range(train_episodes):
        state = env.reset()
        bot_state = env.obs_agent_two()
        done = False
        ep_reward = 0
        frames = 0

        bot = h_env.BasicOpponent(weak=weak_oponent)

        while not done and frames < 1200:
            action, action_idx = agent.select_action(state)
            bot_action = bot.act(bot_state)
            next_state, reward, done, _, info = env.step(np.hstack([action, bot_action]))

            #calculate new reward
            reward_touch_puck = info["reward_touch_puck"]
            reward_puck_direction = info["reward_puck_direction"]
            reward_closeness_to_puck = info["reward_closeness_to_puck"]

            #winner info, could add to reward
            winner = info["winner"]
            winner_reward = 0
            if winner == -1:
                winner_reward = -10
            if winner == 1:
                winner_reward = 10



            reward = reward_touch_puck + reward_puck_direction + winner_reward + 0.25 * reward_closeness_to_puck

            bot_state = env.obs_agent_two()

            agent.replay_buffer.push(state, action_idx, reward, next_state, done)
            agent.learn(batchsize)

            if step_count % update_freq == 0:
                agent.update_target_net()

            state = next_state
            ep_reward += reward
            step_count += 1
            frames += 1

        
        if ep_count % 2 == 0:
            agent.update_epsilon()
        reward_history.append(ep_reward)

        current_avg_score = np.mean(reward_history[-100:])

        if ep_count % 10 == 0:
            writer.add_scalar("AVG_Reward", current_avg_score, ep_count)

        print("Ep: {}, Total Steps: {}, Ep Score: {}, Avg score: {}, Updated Epsilon: {}, Frames: {}".format(ep_count, step_count, ep_reward, current_avg_score, agent.epsilon, frames))

        if current_avg_score >= prev_avg_score and ep_count > 100:
            agent.save_network(model_filename)
            prev_avg_score = current_avg_score

    agent.save_network(model_filename + "_final_model")
    writer.flush()
    writer.close()


def fill_buffer(env, agent, buffer_fill_episodes):
    for _ in range(buffer_fill_episodes):
        state = env.reset()        
        bot_state = env.obs_agent_two()
        done = False

        bot = h_env.BasicOpponent(weak=True)

        while not done:
            action, action_idx = agent.select_action(state)
            bot_action = bot.act(bot_state)
            next_state, reward, done, _, info = env.step(np.hstack([action, bot_action]))
            agent.replay_buffer.push(state, action_idx, reward, next_state, done)
            bot_state = env.obs_agent_two()
            state=next_state


def test(env, agent, test_episodes):
    
    overall_reward = 0

    for ep_count in range(test_episodes):
        state = env.reset()
        bot_state = env.obs_agent_two()

        _ = env.render()

        done = False
        ep_reward = 0
        frames = 0
        
        bot = h_env.BasicOpponent(weak=True)

        while not done and frames < 1200:
            env.render()

            bot_action = bot.act(bot_state)
            action, action_idx = agent.select_action(state)
            next_state, reward, done, _, info = env.step(np.hstack([action, bot_action]))
            state = next_state
            bot_state = env.obs_agent_two()
            ep_reward += reward
            frames += 1

        
        overall_reward += ep_reward
        print('Ep: {}, Ep score: {}'.format(ep_count, ep_reward))
    print("Final average reward: {}".format((overall_reward/test_episodes)))


def set_seed(env, seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    #env.seed(seed_value)
    #env.action_space.np_random.seed(seed_value)

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = False
    continue_training = False

    # lunar lander with visualization
    #env = gymnasium.make("LunarLander-v2", render_mode="human")
    # lunar lander without visualization
    #env = gymnasium.make("LunarLander-v2")

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.TRAIN_SHOOTING)

    # lunar lander
    #observation_dim = env.observation_space,
    #action_dim=env.action_space

    # hockey
    observation_dim = 18
    action_dim = len(ACTIONS)
    difficulty_weak = True


    model_filename = "normal_training_nn_first"

    if train_mode:
        
        set_seed(env, 0)

        if continue_training:
            agent = DuelingDQNAgent(observation_dim=observation_dim,
                                    action_dim=action_dim,
                                    device=device,
                                    epsilon_max=1.0,
                                    epsilon_min=0.01,
                                    epsilon_decay=0.995,
                                    buffer_max_size=20000,
                                    discount=0.99,
                                    lr=1e-3,
                                    double=True)
            agent.load_network(model_filename)
            model_filename = "normal_training"
        else:
            agent = DuelingDQNAgent(observation_dim=observation_dim,
                                    action_dim=action_dim,
                                    device=device,
                                    epsilon_max=1.0,
                                    epsilon_min=0.01,
                                    epsilon_decay=0.995,
                                    buffer_max_size=20000,
                                    discount=0.99,
                                    lr=1e-4,
                                    double=True)
            model_filename = model_filename + "_first"
        
        train(env=env,
              agent=agent,
              train_episodes=4000,
              buffer_fill_episodes=40,
              batchsize=128,
              update_freq=1000,
              model_filename=model_filename,
              weak_oponent=difficulty_weak)
        
    else:
        set_seed(env, 10)

        agent = DuelingDQNAgent(observation_dim=observation_dim,
                                action_dim=action_dim,
                                device=device,
                                epsilon_max=0.0,
                                epsilon_min=0.0,
                                epsilon_decay=0.0,
                                buffer_max_size=10000,
                                discount=0.99,
                                lr=1e-3,
                                double=True)
        
        agent.load_network(model_filename)

        test(env=env, agent=agent, test_episodes=100)