import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class PlaneEnv(gym.Env):

    metadata = {
        'render.modes': ['none', 'human']
    }

    def __init__(self):
        self.min_altitude = 0
        self.max_altitude = 30
        self.goal_altitude = 15
        self.max_pitch = 2

        self.low = np.array([self.min_altitude, -self.max_pitch])
        self.high = np.array([self.max_altitude, self.max_pitch])
        self.gravity = 0.0025

        # action: servos
        self.action_space = spaces.Discrete(3)

        # states: pitch, altitude
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.float32)

        self.seed()
        self.viewer = None

        # Start the flight
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)

    def _get_obs(self):
        return (self.pitch + self.sn, self.altitude + self.altn)

    def reset(self):
        self.pitch = np.random.randint(-self.sn, self.sn)
        self.altitude = np.random.randint(-self.altn, self.altn)
        self.reward_list = []
        self.count = 0
        return self._get_obs()

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_height = self.max_altitude - self.min_altitude
        scale = screen_height/world_height

        # creates rendering class
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            ys = bp.linspace(self.min_altitude, self.max_altitude, 100)
            xs = self._width(ys)
            xys = list(zip((xs-self.min_altitude)*scale, ys*scale))
            
            # background
            self.background = rendering.make_polyline(xys)
            self.background.set_linewidth(4)
            self.viewer.add_geom(self.background)

            # plane object
            plane = rendering.make_capsule(10, 15)
            plane.add_attr(rendering.Transform(translation=(0, 10)))
            self.planetrans = rendering.Transform()
            plane.add_attr(planetrans)
            self.viewer.add_geom(plane)

            # target altitude
            talt = rendering.Line(start=(0, goal_altitude*scale), end=(screen_width, goal_altitude*scale))
            talt.set_color(1, 0, 0)
            self.viewer.add_geom(talt)

        pos = self.state[0]
        self.planetrans.set_translate(self._width(pos)*scale, (pos-self.min_altitude)*scale)
        self.planetrans.set_rotation(math.cos(3*pos))