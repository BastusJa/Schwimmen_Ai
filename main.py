from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from Enviroment import Enviroment
from CardDeck import CardDeck
from Enemy import Enemy

def make_env():
    card_deck = CardDeck()
    enemy = Enemy()
    return Enviroment(card_deck, enemy, 2)


env = make_vec_env(
    make_env,
    n_envs=16,          # number of parallel environments
    vec_env_cls=None   # default: DummyVecEnv
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1
)

model.learn(total_timesteps=500000)
model.save("ppo_model")