from gymnasium.spaces.discrete import Discrete
import os
from stable_baselines3 import PPO

from CardDeck import CardDeck
from Enviroment import Enviroment
from stable_baselines3.common.env_util import make_vec_env

class Enemy:
    def __init__(self, file_path=None, model=None):
        self.action_space = Discrete(12) 
        if model is not None:
            self.model = model
        elif file_path and os.path.exists(file_path):
            self.model = PPO.load(file_path)
        else:
            self.model = MockModel(self.action_space)

    def turn(self, observation):
        action, _ = self.model.predict(observation=observation)
        return action
    
class MockModel:
    def __init__(self, action_space):
        self.action_space = action_space

    def predict(self, observation, deterministic=False):
        return self.action_space.sample(), None