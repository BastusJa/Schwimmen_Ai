from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from Enviroment import Enviroment
from CardDeck import CardDeck
from Enemy import Enemy

import os
import logging

file_path = "ppo_model.zip"
policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))

for i in range(1000):
    def make_env():
        card_deck = CardDeck()
        enemy = Enemy(file_path)
        return Enviroment(card_deck, enemy, 2)

    env = make_vec_env(
        make_env,
        n_envs=16,          # number of parallel environments
        vec_env_cls=None   # default: DummyVecEnv
    )

    if os.path.exists(file_path):
        model = PPO.load(file_path, env=env)
        logging.warning("Old PPO Model gets loaded")
    else: 
        logging.warning("New PPO Model generated")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=policy_kwargs
        )

    model.learn(total_timesteps=200000)
    model.save(file_path)