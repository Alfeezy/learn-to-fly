import gym
from gym import spaces
from gym.utils import seeding
import numpy as np



class PlaneEnv(gym.Env):

    def __init__(self):

        self.altn = 15
        self.sn = 2
        # action: servos
        self.action_space = spaces.Discrete(3)
        # states: pitch, altitude
        self.observation_space = spaces.Tuple((
            spaces.Discrete((self.sn * 2) + 1),
            spaces.Discrete((self.altn * 2) + 1)))
        self.seed()

        # Start the flight
        self.reset()



    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        self.pitch += action - 1
        self.pitch = max(min(self.pitch, self.sn), -self.sn)
        self.altitude += self.pitch
        if self.altitude < -self.altn:
            return self._get_obs(), -self.altn, True, {}

        reward = self.altn - abs(self.altitude)
        if reward < 2:
            self.count += 1

        self.altitude = max(min(self.altitude, self.altn), -self.altn)
        return self._get_obs(), reward, (self.count > self.altn), {}

    def _get_obs(self):
        return (self.pitch + self.sn, self.altitude + self.altn)

    def reset(self):
        self.pitch = np.random.randint(-self.sn, self.sn)
        self.altitude = np.random.randint(-self.altn, self.altn)
        self.count = 0
        return self._get_obs()