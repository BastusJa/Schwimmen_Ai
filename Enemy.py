from gymnasium.spaces.discrete import Discrete

class Enemy:
    def __init__(self):
        self.action_space = Discrete(12) 

    def turn(self):
        return self.action_space.sample()