import os
import gymnasium
import numpy as np
import torch
import time

import tensorboard
from torch.utils.tensorboard import SummaryWriter

from torchinfo import summary

import laserhockey.hockey_env as h_env

from actions import ACTIONS

from DuelingDQNAgent import DuelingDQNAgent


# Function that manages the training of an agent
def train(env, agent, train_episodes, buffer_fill_episodes, batchsize, update_freq, model_filename, weak_opponent, shuffle_opponent):
    fill_buffer(env, agent, buffer_fill_episodes)
    print("Samples in buffer: ", len(agent.replay_buffer))

    # SummaryWriter to create a visualization using tensorboard after the training
    writer = SummaryWriter()

    step_count = 0
    reward_history = []
    prev_avg_score = -np.inf




    for ep_count in range(train_episodes):
        state, _ = env.reset()
        bot_state = env.obs_agent_two()
        done = False
        ep_reward = 0
        frames = 0

        # if shuffle is true, play against the weak and the hard opponent alternating each turn
        # otherwise use the selected strength for the opponent
        if shuffle_opponent:
            if ep_count % 2 == 0:
                bot = h_env.BasicOpponent(weak=False)
            else:                 
                bot = h_env.BasicOpponent(weak=weak_opponent)
        else:
            bot = h_env.BasicOpponent(weak=weak_opponent)

        while not done and frames < 1200:
            action, action_idx = agent.select_action(state)

            # only record every 4 actions in the buffer
            for _ in range(4):
                bot_action = bot.act(bot_state)
                next_state, r, done, _, info = env.step(np.hstack([action, bot_action]))
                if done:
                    break

            #calculate new reward
            reward_touch_puck = info["reward_touch_puck"]
            reward_puck_direction = info["reward_puck_direction"]
            reward_closeness_to_puck = info["reward_closeness_to_puck"]

            #setting the winner reward
            winner = info["winner"]
            winner_reward = 0
            if winner == -1:
                winner_reward = -10
            if winner == 1:
                winner_reward = 10

            #actually calculating the new reward
            reward = reward_touch_puck + reward_puck_direction + winner_reward + 0.25 * reward_closeness_to_puck

            bot_state = env.obs_agent_two()

            agent.replay_buffer.push(state, action_idx, reward, next_state, done)
            agent.learn(batchsize)

            # if the step_count modulo the update frequency is 0, the target net is updated
            if step_count % update_freq == 0:
                agent.update_target_net()

            state = next_state
            ep_reward += reward
            step_count += 1
            frames += 1

        # after every 4 episodes, the epsilon is updated with the epsilon decay value
        if ep_count % 4 == 0:
            agent.update_epsilon()
        reward_history.append(ep_reward)

        # calculate the average reward of the last 100 episodes
        current_avg_score = np.mean(reward_history[-100:])

        # Every 10 episodes add the current avg reward to the visualization 
        if ep_count % 10 == 0:
            writer.add_scalar("AVG_Reward", current_avg_score, ep_count)

        # Prints a short summary of the current episode
        print("Ep: {}, Total Steps: {}, Ep Score: {}, Avg score: {}, Updated Epsilon: {}, Frames: {}".format(ep_count, step_count, ep_reward, current_avg_score, agent.epsilon, frames))

        # if the calculated average reward is higher than any time before, we save the network
        if current_avg_score >= prev_avg_score and ep_count > 100:
            agent.save_network(model_filename)
            prev_avg_score = current_avg_score

    # at the end of the training run the network is saved and the writer is closed
    agent.save_network(model_filename + "_final_model")
    writer.flush()
    writer.close()

# Function to fill the buffer at the start of the training with some random movements and their reward
def fill_buffer(env, agent, buffer_fill_episodes):
    for _ in range(buffer_fill_episodes):
        state, _ = env.reset()        
        bot_state = env.obs_agent_two()
        done = False

        bot = h_env.BasicOpponent(weak=True)

        while not done:
            action, action_idx = agent.select_action(state)
            bot_action = bot.act(bot_state)
            next_state, reward, done, _, info = env.step(np.hstack([action, bot_action]))

            #calculate new reward
            reward_touch_puck = info["reward_touch_puck"]
            reward_puck_direction = info["reward_puck_direction"]
            reward_closeness_to_puck = info["reward_closeness_to_puck"]

            #setting the winner reward
            winner = info["winner"]
            winner_reward = 0
            if winner == -1:
                winner_reward = -10
            if winner == 1:
                winner_reward = 10

            #calculating a new reward
            reward = reward_touch_puck + reward_puck_direction + winner_reward + 0.25 * reward_closeness_to_puck

            agent.replay_buffer.push(state, action_idx, reward, next_state, done)
            bot_state = env.obs_agent_two()
            state=next_state


# Function to test an agent, given a number of test runs and the given strength of the oponent
def test(env, agent, test_episodes, weak):

    # used to estimate the quality of an agent 
    overall_reward = 0
    wins = 0
    losses = 0
    draws = 0


    for ep_count in range(test_episodes):
        state, _ = env.reset()
        bot_state = env.obs_agent_two()

        _ = env.render()

        done = False
        ep_reward = 0
        frames = 0
        
        bot = h_env.BasicOpponent(weak=weak)

        # run the current game until either it is done or the frames reach the frame limit of 1200
        while not done and frames < 1200:
            env.render()

            bot_action = bot.act(bot_state)
            action, action_idx = agent.select_action(state)
            next_state, reward, done, _, info = env.step(np.hstack([action, bot_action]))
            state = next_state
            bot_state = env.obs_agent_two()
            ep_reward += reward
            frames += 1

            if done:
                if info["winner"] == -1:
                    losses += 1
                elif info["winner"] == 0:
                    draws += 1
                else:
                    wins += 1


        
        overall_reward += ep_reward
        print('Ep: {}, Ep score: {}'.format(ep_count, ep_reward))
    
    # print final results of the agent and calculate the winrate with the given formula
    print("Final average reward: {}".format((overall_reward/test_episodes)))
    print("Wins: {} Losses: {} Draws: {} Winrate: {}".format(wins, losses, draws, ((wins + 0.5*draws)/test_episodes)))


def set_seed(env, seed_value):
    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    #env.seed(seed_value)
    #env.action_space.np_random.seed(seed_value)

# Main method, is called to either train or test different agents
if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_mode = False
    continue_training = False

    # lunar lander with visualization
    #env = gymnasium.make("LunarLander-v2", render_mode="human")
    # lunar lander without visualization
    #env = gymnasium.make("LunarLander-v2")

    env = h_env.HockeyEnv()

    # lunar lander
    #observation_dim = env.observation_space,
    #action_dim=env.action_space

    # hockey
    observation_dim = 18
    action_dim = len(ACTIONS)
    difficulty_weak = True
    shuffle_opponent = True


    model_filename = "final_model"

    # When training is true, an agent will be trained, otherwise the given agent in model_filename will be evaluated
    if train_mode:
        
        set_seed(env, 0)

        # When continue_training is true, the agent stored in model_filename will be loaded and the training is continued on that agent.
        # Otherwise the training starts on a new agent and is stored at the position in model_filename.
        if continue_training:
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
            agent.load_network(model_filename)
            model_filename = "final_shuffle_training_nn_first"
        else:
            agent = DuelingDQNAgent(observation_dim=observation_dim,
                                    action_dim=action_dim,
                                    device=device,
                                    epsilon_max=1.0,
                                    epsilon_min=0.05,
                                    epsilon_decay=0.995,
                                    buffer_max_size=20000,
                                    discount=0.99,
                                    lr=1e-4,
                                    double=True)
            model_filename = model_filename + "_first"
        
        train(env=env,
              agent=agent,
              train_episodes=40000,
              buffer_fill_episodes=40,
              batchsize=128,
              update_freq=1000,
              model_filename=model_filename,
              weak_opponent=difficulty_weak,
              shuffle_opponent = shuffle_opponent)
        
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
        print("Playing against weak opponent:")
        #test(env=env, agent=agent, test_episodes=1000, weak=difficulty_weak)
        print("Playing against strong opponent:")
        #test(env=env, agent=agent, test_episodes=1000, weak=False)

        summary(agent.policy_net)