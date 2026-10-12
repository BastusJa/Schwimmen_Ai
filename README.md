# Schwimmen AI

A reinforcement-learning agent for the German card game **Schwimmen** (also known as **31**). The project uses [Gymnasium](https://gymnasium.farama.org/) to model the game environment and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) to train a PPO agent.

## Features

- PPO-based game-playing agent.
- Gymnasium-compatible environment with a fixed-size observation vector.
- 12 discrete actions for passing, knocking, swapping all cards, or swapping individual cards.
- Parallel training with vectorized environments.
- Deterministic evaluation against a greedy Schwimmen baseline.
- Optional Pygame interface for interactive rendering.

## Requirements

- Python 3.12 or newer
- `numpy`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `pygame` for rendering

The repository includes a local virtual environment in `.venv` when checked out in the current development setup.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy gymnasium stable-baselines3 torch pygame
```

## Training

Start or resume PPO training with:

```bash
python main.py
```

The script:

1. Creates 16 parallel game environments.
2. Loads `ppo_model.zip` if it exists.
3. Otherwise creates a new PPO model.
4. Trains for 200,000 timesteps per training cycle.
5. Saves the model back to `ppo_model.zip`.
6. Repeats training cycles until interrupted.

Stop training with `Ctrl+C`. The model is saved after each completed cycle. Training uses CPU by default and limits PyTorch to one thread because the environment and policy network are small.

### Training configuration

The main settings are defined in [main.py](main.py):

- `n_envs=16`: number of parallel environments.
- `total_timesteps=200000`: timesteps per cycle.
- `n_steps=256`: rollout length per environment.
- `batch_size=1024`: PPO minibatch size.
- `net_arch`: policy and value-network architecture.

Delete `ppo_model.zip` before starting if a fresh model is required.

## Evaluation

Evaluate a trained model against the deterministic baseline:

```bash
python test.py --agent-factory util:load_model --agent-path ppo_model.zip --seed 42 --games 1000
```

The evaluation reports:

- Agent wins and win rate.
- Baseline wins.
- Draws.
- Average score for both players.
- Average number of turns per game.

The baseline evaluates all legal swaps and chooses the move that maximizes its immediate hand score. It swipes when no swap improves its score.

## Game environment

The environment is implemented in [Enviroment.py](Enviroment.py). The spelling of the filename is retained for compatibility with the existing imports.

### Actions

| Action | Meaning |
| ---: | --- |
| `0` | Knock/passively end the player's participation |
| `1` | Swipe; after all players swipe, replace the middle cards |
| `2` | Swap all three hand cards with all three middle cards |
| `3`–`11` | Swap one hand card with one middle card |

For actions `3`–`11`, the mapping is:

```text
action = (hand_card_index + 1) * 3 + middle_card_index
```

Both indexes are zero-based.

### Observation space

For two players, each observation contains 21 normalized `float32` values:

- 6 values for the agent's three cards: rank value and suit.
- 6 values for the three middle cards.
- Turn, knock, and swipe state.
- 6 values for known opponent cards.

Unknown opponent cards are represented by placeholder values so the observation always has a fixed shape.

## Project structure

| File | Purpose |
| --- | --- |
| [main.py](main.py) | PPO training entry point |
| [Enviroment.py](Enviroment.py) | Gymnasium game environment and rules |
| [Enemy.py](Enemy.py) | PPO-backed or random opponent |
| [Card.py](Card.py) | Card representation and rank values |
| [CardDeck.py](CardDeck.py) | Deck creation, shuffling, and drawing |
| [util.py](util.py) | Model-loading helper used by evaluation |
| [test.py](test.py) | Evaluation against the deterministic baseline |
| [view.py](view.py) | Rendering/view-related code |
| `ppo_model.zip` | Saved PPO model, generated during training |

## Rendering

The environment supports Pygame rendering through its `render_mode` option. Training and command-line evaluation use a non-rendering mode so that they remain fast and do not open a window.

## Troubleshooting

### Observation shape errors

The PPO model expects the environment's observations to match `observation_space`. For the default two-player setup, the shape must be `(21,)`. The environment keeps observations fixed-size even when opponent cards are unknown or the deck is nearly empty.

### Missing model

If `ppo_model.zip` does not exist, `main.py` creates a new model. The opponent uses a random policy until a saved PPO model is available.

### Slow training

Use the existing vectorized setup and keep rendering disabled. If CPU utilization is low, adjust `n_envs` in [main.py](main.py), but very large values may increase overhead for this small environment.

## License

This project is licensed under the [MIT License](LICENSE).
