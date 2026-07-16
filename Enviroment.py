import time

from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces.discrete import Discrete
import numpy as np
import logging 

from Enemy import Enemy
from CardDeck import CardDeck
from Card import Card
import copy

class Enviroment(Env):
    def __init__(self, card_deck: CardDeck, enemy_player: Enemy, player_count, render_mode=""):
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15,),
            dtype=np.float32
        )
        self.render_mode = render_mode
        self.action_space = Discrete(12) 

        self.init_card_deck = card_deck
        self.enemy = enemy_player
        self.player_count = player_count
        self.max_turns = 30
        self.win_count = [0] * player_count
        self.reset()

    def reset(self, seed=None, options=None):
        self.card_deck = copy.deepcopy(self.init_card_deck)
        self.card_deck.shuffle()
        self.hand_cards = []
        for i in range(self.player_count):
            cards = self.get_cards_from_deck(3)
            self.hand_cards.append(cards)
        
        self.middle_cards = self.get_cards_from_deck(3)
        self.turns = 0
        self.knocked = 0
        self.swiped = 0
        self.winner = None
        self.state = self.__get_vec_obs()
        info = {}

        return self.state, info

    def step(self, action):
        self.__do_act(action, 0)
        terminated = self.is_done()
        truncated = self.time_limit_reached()
        reward = self.calculate_reward()

        info = {}

        self.turns += 1
        action = self.enemy.turn()
        self.__do_act(action, 1)
        self.state = self.__get_vec_obs()

        return self.state, reward, terminated, truncated, info
    
    def __do_act(self, action, player_number):
        if action == 0:
            self.knocked += 1
        elif action == 1:
            self.swiped += 1
            if self.swiped == self.player_count:
                self.middle_cards[0] = self.card_deck.draw()
                self.middle_cards[1] = self.card_deck.draw()
                self.middle_cards[2] = self.card_deck.draw()
        elif action == 2:
            self.hand_cards[player_number][0], self.middle_cards[0] = self.middle_cards[0], self.hand_cards[player_number][0]
            self.hand_cards[player_number][1], self.middle_cards[1] = self.middle_cards[1], self.hand_cards[player_number][1]
            self.hand_cards[player_number][2], self.middle_cards[2] = self.middle_cards[2], self.hand_cards[player_number][2]
        elif action == 3:
            self.hand_cards[player_number][0], self.middle_cards[0] = self.middle_cards[0], self.hand_cards[player_number][0]
        elif action == 4:
            self.hand_cards[player_number][0], self.middle_cards[1] = self.middle_cards[1], self.hand_cards[player_number][0]
        elif action == 5:
            self.hand_cards[player_number][0], self.middle_cards[2] = self.middle_cards[2], self.hand_cards[player_number][0]
        elif action == 6:
            self.hand_cards[player_number][1], self.middle_cards[0] = self.middle_cards[0], self.hand_cards[player_number][1]
        elif action == 7:
            self.hand_cards[player_number][1], self.middle_cards[1] = self.middle_cards[1], self.hand_cards[player_number][1]
        elif action == 8:
            self.hand_cards[player_number][1], self.middle_cards[2] = self.middle_cards[2], self.hand_cards[player_number][1]
        elif action == 9:
            self.hand_cards[player_number][2], self.middle_cards[0] = self.middle_cards[0], self.hand_cards[player_number][2]
        elif action == 10:
            self.hand_cards[player_number][2], self.middle_cards[1] = self.middle_cards[1], self.hand_cards[player_number][2]
        elif action == 11:
            self.hand_cards[player_number][2], self.middle_cards[2] = self.middle_cards[2], self.hand_cards[player_number][2]

        if action > 1:
            self.swiped = 0
        if action > 1 and self.knocked > 0:
            self.knocked += 1

    def calculate_reward(self):
        if self.winner == None:
            return 0
        elif self.winner == 1:
            return 100
        elif self.winner == 2:
            return -100
        else: 
            return -200
        
    def is_done(self):
        scores = self.calculate_score()
        for i, score in enumerate(scores):
            if score == 31:
                self.winner = i
                self.win_count[self.winner] += 1
                logging.warning(f"Player {i} has 31")
                return True
            
        if self.knocked == self.player_count:
            self.winner = scores.index(max(scores))
            self.win_count[self.winner] += 1
            logging.warning(f"Player {self.winner} has {max(scores)}")
            return True
        
    def time_limit_reached(self):
        if self.turns > self.max_turns:
            return True
        return False

    def calculate_score(self):
        scores = []
        for hand in self.hand_cards:
            highest_per_suit = {}
            card_symbols = []
            for card in hand:
                card_symbols.append(card.symbol)
                if card.suit not in highest_per_suit:
                    highest_per_suit[card.suit] = 0
                highest_per_suit[card.suit] += card.value
            highest_value = max(highest_per_suit.values())
            if card_symbols[0] == card_symbols[1] == card_symbols[2]:
                highest_value = 30.5
            
            scores.append(highest_value)

        return scores

    def get_cards_from_deck(self, amount):
        cards = []
        for j in range(amount):
            cards.append(self.card_deck.draw())
        return cards
    
    def __get_vec_obs(self):
        observation = np.array([
            self.hand_cards[0][0].value / 11,
            self.hand_cards[0][1].value / 11,
            self.hand_cards[0][2].value / 11,
            self.middle_cards[0].value / 11,
            self.middle_cards[1].value / 11,
            self.middle_cards[2].value / 11,
            self.__get_suit_value(self.hand_cards[0][0].suit) / 3,
            self.__get_suit_value(self.hand_cards[0][1].suit) / 3,
            self.__get_suit_value(self.hand_cards[0][2].suit) / 3,
            self.__get_suit_value(self.middle_cards[0].suit) / 3,
            self.__get_suit_value(self.middle_cards[1].suit) / 3,
            self.__get_suit_value(self.middle_cards[2].suit) / 3,
            self.turns / self.max_turns,
            1 if self.knocked > 0 else 0,
            1 if self.swiped < 0 else 0
        ], dtype=np.float32)
        return observation
    
    def __get_suit_value(self, suit):
        if suit == "Clower":
            return 0
        elif suit == "Hearth":
            return 1
        elif suit == "Caro":
            return 2
        elif suit == "Pick":
            return 3
        else:
            return None
        
    def render(self):
        import pygame
        if self.render_mode == "human":
            time.sleep(0.5)

        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((1000, 700))
            pygame.display.set_caption("31 Card Game")
            self.font = pygame.font.Font(None, 32)
            self.small_font = pygame.font.Font(None, 24)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        self.screen.fill((30, 130, 60))

        def draw_text(text, x, y, font=None):
            if font is None:
                font = self.font
            img = font.render(str(text), True, (255, 255, 255))
            self.screen.blit(img, (x, y))


        def draw_card(card, x, y):
            # Card background
            pygame.draw.rect(
                self.screen,
                (255, 255, 255),
                (x, y, 120, 170)
            )

            # Border
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (x, y, 120, 170),
                2
            )

            # Convert numpy strings to normal strings
            symbol = str(card.symbol)
            suit = str(card.suit)
            name = str(card.name)
            value = str(card.value)

            # Create text surfaces
            symbol_text = self.font.render(
                symbol,
                True,
                (0, 0, 0)
            )

            name_text = self.small_font.render(
                name,
                True,
                (0, 0, 0)
            )

            suit_text = self.small_font.render(
                suit,
                True,
                (0, 0, 0)
            )

            value_text = self.small_font.render(
                "Value: " + value,
                True,
                (0, 0, 0)
            )

            # Draw texts on card
            self.screen.blit(
                symbol_text,
                (x + 50, y + 10)
            )

            self.screen.blit(
                name_text,
                (x + 10, y + 55)
            )

            self.screen.blit(
                suit_text,
                (x + 10, y + 85)
            )

            self.screen.blit(
                value_text,
                (x + 10, y + 125)
            )

        # ----------------------
        # Player 1 hand
        # ----------------------
        draw_text("Player 1", 40, 30)

        for i, card in enumerate(self.hand_cards[0]):
            draw_card(
                card,
                40 + i * 130,
                80
            )


        # ----------------------
        # Middle cards
        # ----------------------
        draw_text("Middle Cards", 40, 280)

        for i, card in enumerate(self.middle_cards):
            draw_card(
                card,
                40 + i * 130,
                330
            )


        # ----------------------
        # Player 2 hand
        # ----------------------
        draw_text("Enemy Player", 500, 30)

        for i, card in enumerate(self.hand_cards[1]):
            draw_card(
                card,
                500 + i * 130,
                80
            )


        # ----------------------
        # Game info
        # ----------------------
        draw_text(
            f"Turn: {self.turns}",
            650,
            330
        )

        scores = self.calculate_score()

        draw_text(
            f"Player 1 Score: {scores[0]}",
            650,
            380
        )

        draw_text(
            f"Player 2 Score: {scores[1]}",
            650,
            420
        )

        draw_text(
            f"Knocked: {self.knocked}",
            650,
            460
        )

        draw_text(
            f"Player 1 Wins: {self.win_count[0]}",
            650,
            500
        )

        draw_text(
            f"Player 2 Wins: {self.win_count[1]}",
            650,
            540
        )

        pygame.display.flip()