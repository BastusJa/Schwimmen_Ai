"""
Compare a trained Gymnasium agent against a deterministic Schwimmen heuristic.

Assumptions based on the supplied Enviroment implementation:
- player 0 is the agent passed to this script
- player 1 is the deterministic baseline
- Enviroment.step(agent_action) automatically calls env.enemy.turn(state) for player 1
- Card objects expose .value, .suit, and .symbol
- CardDeck can be constructed with CardDeck()

Example:
    python test_vs_deterministic.py --agent-factory my_agent:load_model --agent-path model.zip --games 500

Your factory should return an agent/model. Supported interfaces:
- Stable-Baselines-style: model.predict(observation, deterministic=True)
- object with .act(observation)
- callable(observation)

The agent factory is optional for a smoke test; without it, a random legal agent is used.
"""

import argparse
import importlib
import random
import statistics
from typing import Any, Callable, Optional

import numpy as np

from CardDeck import CardDeck
from Enviroment import Enviroment


def hand_score(cards) -> float:
    """Mirror the scoring rules in Enviroment.calculate_score()."""
    totals = {}
    for card in cards:
        totals[card.suit] = totals.get(card.suit, 0) + card.value

    score = max(totals.values()) if totals else 0
    if len(cards) == 3 and cards[0].symbol == cards[1].symbol == cards[2].symbol:
        score = 30.5
    return float(score)


class DeterministicSchwimmen:
    """
    Greedy baseline: choose the legal swap that maximizes its current hand score.
    If no swap improves the score, swipe/pass. Action 2 swaps all three cards.
    Ties are resolved deterministically by keeping the first action encountered.
    """

    def __init__(self):
        self.env = None

    def turn(self, observation):
        # The supplied environment's observation contains player 0's hand, not
        # player 1's hand. Use the environment state to evaluate the baseline's
        # own cards. The env reference is assigned after environment creation.
        if self.env is None:
            raise RuntimeError("Baseline.env must be assigned to the environment.")

        hand = self.env.hand_cards[1]
        middle = self.env.middle_cards
        current_score = hand_score(hand)

        # If the hand is already strong and no swap improves it, knock to
        # finish the round instead of giving the agent another opportunity.
        best_action = 0 if current_score > 28 else 1
        best_score = current_score

        # Action 2: swap all three cards.
        candidate = list(middle)
        score = hand_score(candidate)
        if score > best_score:
            best_score, best_action = score, 2

        # Actions 3..11: swap one hand card with one middle card.
        for hand_index in range(3):
            for middle_index in range(3):
                action = (hand_index + 1) * 3 + middle_index
                candidate = list(hand)
                candidate[hand_index] = middle[middle_index]
                score = hand_score(candidate)
                if score > best_score:
                    best_score, best_action = score, action

        return best_action


def import_factory(spec: str) -> Callable:
    """Load a factory written as 'module_name:function_name'."""
    if ":" not in spec:
        raise ValueError("--agent-factory must have the form module:function")
    module_name, function_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def load_agent(factory_spec: Optional[str], agent_path: Optional[str]):
    if not factory_spec:
        print("No --agent-factory supplied; using a random legal agent for a smoke test.")
        return None

    factory = import_factory(factory_spec)
    if agent_path:
        try:
            return factory(agent_path)
        except TypeError:
            # Also support factories that load their own configured model.
            return factory()
    return factory()


def choose_agent_action(agent: Any, observation) -> int:
    if agent is None:
        return random.randrange(12)

    if hasattr(agent, "predict"):
        result = agent.predict(observation, deterministic=True)
        action = result[0] if isinstance(result, tuple) else result
    elif hasattr(agent, "act"):
        action = agent.act(observation)
    elif callable(agent):
        action = agent(observation)
    else:
        raise TypeError(
            "Agent must expose predict(observation), act(observation), or be callable."
        )

    if hasattr(action, "item"):
        action = action.item()
    action = int(action)
    if not 0 <= action < 12:
        raise ValueError(f"Agent returned invalid action {action}; expected 0..11.")
    return action


def play_one_game(agent, seed: int, max_steps: int):
    random.seed(seed)
    np.random.seed(seed)

    baseline = DeterministicSchwimmen()
    env = Enviroment(
        card_deck=CardDeck(),
        enemy_player=baseline,
        player_count=2,
        render_mode="",  # Do not open the Pygame window.
    )
    baseline.env = env

    observation, _ = env.reset(seed=seed)
    terminated = truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        action = choose_agent_action(agent, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        steps += 1

    # Recompute scores for reporting; some environments only update scores in is_done().
    scores = env.calculate_score()
    winner = env.winner

    # If the environment reached a limit without setting a winner, compare scores.
    if winner is None:
        if scores[0] > scores[1]:
            winner = 0
        elif scores[1] > scores[0]:
            winner = 1
        else:
            winner = -1  # draw

    result = {
        "winner": winner,
        "agent_score": float(scores[0]),
        "baseline_score": float(scores[1]),
        "steps": steps,
    }

    # Avoid leaking a Pygame window if another part of the environment initialized it.
    if hasattr(env, "screen"):
        import pygame
        pygame.display.quit()
        pygame.quit()

    return result


def main():
    parser = argparse.ArgumentParser(description="Evaluate an agent against deterministic Schwimmen.")
    parser.add_argument("--agent-factory", help="Factory as module:function that returns your agent/model.")
    parser.add_argument("--agent-path", help="Optional path passed to the factory, e.g. a model file.")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play (default: 100).")
    parser.add_argument("--seed", type=int, default=123, help="Starting random seed.")
    parser.add_argument("--max-steps", type=int, default=200, help="Safety cap per game.")
    args = parser.parse_args()

    if args.games < 1:
        parser.error("--games must be at least 1")

    agent = load_agent(args.agent_factory, args.agent_path)
    results = [
        play_one_game(agent, args.seed + game_index, args.max_steps)
        for game_index in range(args.games)
    ]

    wins = sum(r["winner"] == 0 for r in results)
    losses = sum(r["winner"] == 1 for r in results)
    draws = sum(r["winner"] == -1 for r in results)
    win_rate = wins / args.games * 100
    avg_agent_score = statistics.mean(r["agent_score"] for r in results)
    avg_baseline_score = statistics.mean(r["baseline_score"] for r in results)
    avg_steps = statistics.mean(r["steps"] for r in results)

    print("\n=== Agent vs Deterministic Schwimmen ===")
    print(f"Games:                 {args.games}")
    print(f"Agent wins:            {wins} ({win_rate:.1f}%)")
    print(f"Baseline wins:         {losses} ({losses / args.games * 100:.1f}%)")
    print(f"Draws:                 {draws} ({draws / args.games * 100:.1f}%)")
    print(f"Average agent score:   {avg_agent_score:.2f}")
    print(f"Average baseline score:{avg_baseline_score: .2f}")
    print(f"Average steps/game:    {avg_steps:.1f}")

    if agent is None:
        print("\nNOTE: This was a random-agent smoke test, not an evaluation of your trained agent.")


if __name__ == "__main__":
    main()
