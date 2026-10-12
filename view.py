from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from Enviroment import Enviroment
from Enemy import Enemy
from CardDeck import CardDeck
import time
import keyboard

card_deck = CardDeck()
enemy = Enemy("ppo_model")

env = Enviroment(card_deck, enemy, 2, "")

obs, info = env.reset()

while True:
    action = env.render()
    if action != None:
        _, _, term, done, _ = env.step(action)
        if done or term:
            env.reset()