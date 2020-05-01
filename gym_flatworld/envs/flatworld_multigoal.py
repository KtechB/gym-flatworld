import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .flatworld import FlatworldEnv
import time
"""
not inplimented yet
"""
class FlatworldMultiGoalEnv(FlatworldEnv):
    def __init__(self, seed=0):
        super().__init__(self, seed)
        

