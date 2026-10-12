from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from Enviroment import Enviroment
from CardDeck import CardDeck
from Enemy import Enemy

import os
import logging
import torch

file_path = "ppo_model.zip"
policy_kwargs = dict(net_arch=dict(pi=[32, 64, 32], vf=[32, 64, 32]))

torch.set_num_threads(1)

# Loading one opponent model per environment was a large startup and memory
# cost. All environments can use the same frozen opponent instance.
opponent = PPO.load(file_path) if os.path.exists(file_path) else None

def make_env():
    card_deck = CardDeck()
    enemy = Enemy(model=opponent) if opponent is not None else Enemy()
    return Enviroment(card_deck, enemy, 2)

env = make_vec_env(make_env, n_envs=16)

if opponent is not None:
    model = PPO.load(file_path, env=env)
    logging.warning("Old PPO Model gets loaded")
else:
    logging.warning("New PPO Model generated")
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
                n_steps=256, batch_size=1024, device="cpu")

for i in range(1000):
    model.learn(total_timesteps=200000)
    model.save(file_path)