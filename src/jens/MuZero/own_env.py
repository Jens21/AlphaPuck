import laserhockey.hockey_env as h_env
import numpy as np

# Class solves the problem of the not working trunc flag
# The customized part is not used currently
# Rest is similar to given environment
# reward is changed, only return info['winnner'] as reward
# frameskip is implemented in self play actor
class OwnEnv():
    def __init__(self):
        self.env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)
        self.done = False

    def seed(self, seed):
        self.env.seed(seed)

    # performed worse than no customization of the observations after the episodes where done
    # might make sense if each returned observation contains data from previous time steps >1
    def customize(self, obs):
        if self.done:
            obs = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1], dtype='float')
        else:
            obs = np.concatenate([obs, [0]])

        return obs

    def reset(self, customized=False):
        self.done = False

        obs, info = self.env.reset()

        if customized:
            obs = self.customize(obs)

        return obs, info

    def obs_agent_two(self, customized=False):
        obs = self.env.obs_agent_two()

        if customized:
            obs = self.customize(obs)

        return obs

    def render(self):
        self.env.render()

    def step(self, action, customized=False):
        obs, reward, done, _, info = self.env.step(action)
        self.done = done

        reward = info['winner']
        trunc = done # is True is some player won or the time went up
        done = np.abs(info['winner'])

        if customized:
            obs = self.customize(obs)

        return obs, reward, done, trunc, info

    def close(self):
        self.env.close()