from stable_baselines3 import PPO


def load_model(model_path="ppo_model.zip"):
    return PPO.load(model_path)
