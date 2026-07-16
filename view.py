from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from Enviroment import Enviroment
from Enemy import Enemy
from CardDeck import CardDeck

model = PPO.load("ppo_model")
def make_env():
    card_deck = CardDeck()
    enemy = Enemy()
    return Enviroment(card_deck, enemy, 2, "human")

vec_env = make_vec_env(make_env)
obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")