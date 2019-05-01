import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym.envs.classic_control import rendering

screen_width = 800
screen_height = 800

viewer = rendering.Viewer(screen_width, screen_height)
input("Press Enter to continue...")
plane = rendering.make_circle()
viewer.close()