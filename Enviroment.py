import time

from gymnasium import Env
from gymnasium import spaces
from gymnasium.spaces.discrete import Discrete
import numpy as np
import logging 

from CardDeck import CardDeck
from Card import Card
import copy
import pygame

# lose and knocked with same score

class Enviroment(Env):
    def __init__(self, card_deck: CardDeck, enemy_player, player_count, render_mode=""):
        self.render_mode = render_mode
        self.action_space = Discrete(12) 

        self.init_card_deck = card_deck
        self.enemy = enemy_player
        self.player_count = player_count
        self.max_turns = 30
        self.win_count = [0] * player_count
        self.former_middle_cards = np.array([])
        self.ui_cards = [[None for _ in range(3)] for _ in range(self.player_count + 1)]
        self.selected_ui_cards = []
        self.selected_card_pos = []
        self.buttons_events = []
        self.player_knocked = False
        self.player_swipped = False

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(15+(self.player_count - 1) * 6,),
            dtype=np.float32
        )
        self.reset()

    def reset(self, seed=None, options=None):
        self.card_deck = copy.deepcopy(self.init_card_deck)
        self.deck_out_of_cards = False
        self.card_deck.shuffle()
        self.hand_cards = []
        self.known_enemy_cards = [[] for _ in range(self.player_count - 1)]
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
        info = {}
        self.do_act(action, 0)
        terminated = self.is_done()
        truncated = self.time_limit_reached()
        reward = self.calculate_reward()
        if not self.deck_out_of_cards:
            self.state = self.__get_vec_obs()

            self.turns += 1
            action = self.enemy.turn(self.state)
            self.do_act(action, 1)
            terminated = self.is_done()
            truncated = self.time_limit_reached()
            if not self.deck_out_of_cards:
                self.state = self.__get_vec_obs()

        return self.state, reward, terminated, truncated, info
    
    def do_act(self, action, player_number):
        self.former_middle_cards = np.append(self.former_middle_cards, copy.deepcopy(self.middle_cards))
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
        reward = 0
        reward = self.winner_to_reward(reward)
        reward = self.score_to_reward(reward)
        return reward


    def winner_to_reward(self, reward):
        if self.winner == None:
            reward += 0
        elif self.winner == 0:
            reward += 1
        elif self.winner == 1:
            reward -= 1
        elif self.turns > self.max_turns:
            reward -= 0.5
        elif self.deck_out_of_cards:
            reward -= 0.5
        else: 
            reward -= 2

        return reward

    def score_to_reward(self, reward):
        return reward + self.scores[0] / 100
        
    def is_done(self):
        self.scores = self.calculate_score()
        for i, score in enumerate(self.scores):
            if score == 31:
                self.winner = i
                self.win_count[self.winner] += 1
                logging.debug(f"Player {i} has 31")
                return True
            
        if self.knocked >= self.player_count:
            self.winner = self.scores.index(max(self.scores))
            self.win_count[self.winner] += 1
            logging.debug(f"Player {self.winner} has {max(self.scores)}")
            return True
        return False
        
    def time_limit_reached(self):
        if self.turns > self.max_turns:
            logging.debug("Timesteps of 30 turns reached")
            return True
        if self.deck_out_of_cards:
            logging.debug("No cards in deck")
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
            card = self.card_deck.draw()
            if card == None:
                self.deck_out_of_cards = True
            cards.append(card)
        return cards
    
    def __get_vec_obs(self):
        base_observation = np.array([
            self.hand_cards[0][0].value / 12,
            self.__get_suit_value(self.hand_cards[0][0].suit) / 4,
            self.hand_cards[0][1].value / 12,
            self.__get_suit_value(self.hand_cards[0][1].suit) / 4,
            self.hand_cards[0][2].value / 12,
            self.__get_suit_value(self.hand_cards[0][2].suit) / 4,
            self.middle_cards[0].value / 12,
            self.__get_suit_value(self.middle_cards[0].suit) / 4,
            self.middle_cards[1].value / 12,
            self.__get_suit_value(self.middle_cards[1].suit) / 4,
            self.middle_cards[2].value / 12,
            self.__get_suit_value(self.middle_cards[2].suit) / 4,         
            self.turns / self.max_turns,
            1 if self.knocked > 0 else 0,
            1 if self.swiped < 0 else 0
        ], dtype=np.float32)
        # Keep the observation length fixed. Unknown enemy cards are encoded
        # as ones, while known cards replace the corresponding placeholders.
        enemy_observation = []
        for enemy_cards in self.known_enemy_cards:
            for j in range(3):
                if j < len(enemy_cards):
                    enemy_observation.extend([
                        enemy_cards[j].value / 12,
                        self.__get_suit_value(enemy_cards[j].suit) / 4,
                    ])
                else:
                    enemy_observation.extend([1.0, 1.0])
        observation = np.append(base_observation, enemy_observation)
        return observation
    
    def __get_suit_value(self, suit):
        if suit == "Clubs":
            return 0
        elif suit == "Hearts":
            return 1
        elif suit == "Diamonds":
            return 2
        elif suit == "Spades":
            return 3
        else:
            return None
        
    def __get_suit_symbol(self, suit):
        if suit == "Clubs":
            return "♣"
        elif suit == "Hearts":
            return "♥"
        elif suit == "Diamonds":
            return "♦"
        elif suit == "Spades":
            return "♠"
        else:
            return None
        
    def __get_suit_color(self, suit):
        if suit == "Clubs":
            return (0, 0, 0)
        elif suit == "Hearts":
            return (200, 30, 30)
        elif suit == "Diamonds":
            return (200, 30, 30)
        elif suit == "Spades":
            return (0, 0, 0)
        else:
            return (0, 0, 0)
        
    def render(self):
        def check_do_action():
            cached_row = None
            middle_card = None
            hand_card = None
            if self.player_knocked == True:
                self.player_knocked = False
                self.selected_ui_cards = []
                self.selected_card_pos = []
                return 0
            if self.player_swipped == True:
                self.player_swipped = False
                self.selected_ui_cards = []
                self.selected_card_pos = []
                return 1

            for (row, col) in self.selected_card_pos:
                if cached_row == None:
                    cached_row = row
                if row != cached_row and len(self.selected_card_pos) > 2:
                    self.selected_ui_cards = []
                    self.selected_card_pos = []
                    return 2
                if row == 1:
                    middle_card = col
                if row != 1:
                    hand_card = col
            if middle_card is not None and hand_card is not None:
                action = (hand_card + 1) * 3 + middle_card
                self.selected_ui_cards = []
                self.selected_card_pos = []
                return action
            else: 
                return None
                    
        def event_loop():
            for event in pygame.event.get():
                # Mouse click event
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        for i, row in enumerate(self.ui_cards):
                            for j, card in enumerate(row):
                                if card and card.collidepoint(event.pos):
                                    if not card in self.selected_ui_cards:
                                        self.selected_ui_cards.append(card)
                                        self.selected_card_pos.append((i, j))
                                    else:
                                        self.selected_ui_cards.remove(card)
                                        self.selected_card_pos.remove((i, j))
                            
                        for button, call in self.buttons_events:
                            if button.collidepoint(event.pos):
                                call()

        def knock_call():
            self.player_knocked = True

        def pass_call():
            self.player_swipped = True

        if not hasattr(self, "screen"):
            pygame.init()
            self.font = pygame.font.SysFont('Tahoma', 32)
            self.screen = pygame.display.set_mode((1000, 700))
            pygame.display.set_caption("31 Card Game")
            self.small_font = pygame.font.SysFont('Tahoma', 24)

        self.screen.fill((40, 120, 40))

        self.buttons_events.clear()
        self.draw_cards()
        self.make_button(700, 130, "Knock", knock_call)
        self.make_button(700, 170, "Swipe", pass_call)
        event_loop()
        pygame.display.flip()
        return check_do_action()
        
    def draw_cards(self):
        card_height = 200
        card_width = 133
        enemy_cards = self.hand_cards[0]
        player_cards = self.hand_cards[1]

        # Enemy cards
        self.ui_cards[0][0] = self.draw_card(30, 30, enemy_cards[0].symbol, enemy_cards[0].suit, card_width, card_height)
        self.ui_cards[0][1] = self.draw_card(card_width + 30 * 2, 30, enemy_cards[1].symbol, enemy_cards[1].suit, card_width, card_height)
        self.ui_cards[0][2] = self.draw_card(card_width * 2 + 30 * 3, 30, enemy_cards[2].symbol, enemy_cards[2].suit, card_width, card_height)

        # Middle cards
        self.ui_cards[1][0] = self.draw_card(30, card_height + 30 * 2, self.middle_cards[0].symbol, self.middle_cards[0].suit, card_width, card_height)
        self.ui_cards[1][1] = self.draw_card(card_width + 30 * 2, card_height + 30 * 2, self.middle_cards[1].symbol, self.middle_cards[1].suit, card_width, card_height)
        self.ui_cards[1][2] = self.draw_card(card_width * 2 + 30 * 3, card_height + 30 * 2, self.middle_cards[2].symbol, self.middle_cards[2].suit, card_width, card_height)

        # Player cards
        # Enemy cards
        self.ui_cards[2][0] = self.draw_card(30, card_height * 2 + 30 * 3, player_cards[0].symbol, player_cards[0].suit, card_width, card_height)
        self.ui_cards[2][1] = self.draw_card(card_width + 30 * 2, card_height * 2 + 30 * 3, player_cards[1].symbol, player_cards[1].suit, card_width, card_height)
        self.ui_cards[2][2] = self.draw_card(card_width * 2 + 30 * 3, card_height * 2 + 30 * 3, player_cards[2].symbol, player_cards[2].suit, card_width, card_height)

    def make_button(self, x, y, label, callback, width=60, height=30):
        label = str(label)
        button = pygame.Rect(x, y, width, height)

        pygame.draw.rect(self.screen, (170, 170, 170), button, border_radius=12)

        text = self.small_font.render(label, True, (0, 0, 0))
        text_rect = text.get_rect(center=button.center)
        self.screen.blit(text, text_rect)

        if not any(b == button and cb == callback for b, cb in self.buttons_events):
            self.buttons_events.append((button, callback))

    def draw_card(self, x, y, rank, suit, width=200, height=300):
        color = self.__get_suit_color(suit)
        symbol = self.__get_suit_symbol(suit)

        # Shadow
        shadow = pygame.Rect(x + 4, y + 4, width, height)
        pygame.draw.rect(self.screen, (60, 60, 60), shadow, border_radius=12)

        # Card body
        card = pygame.Rect(x, y, width, height)
        if card in self.selected_ui_cards:
            pygame.draw.rect(self.screen, (173, 216, 230), card, border_radius=12)
        else:
            pygame.draw.rect(self.screen, (250, 250, 250), card, border_radius=12)
        pygame.draw.rect(self.screen, (40, 40, 40), card, 2, border_radius=12)

        # Top-left
        tl = self.small_font.render(rank, True, color)
        ts = self.small_font.render(symbol, True, color)
        self.screen.blit(tl, (x + 8, y + 6))
        self.screen.blit(ts, (x + 10, y + 28))

        # Center suit
        center = self.font.render(symbol, True, color)
        rect = center.get_rect(center=(x + width // 2, y + height // 2))
        self.screen.blit(center, rect)

        # Bottom-right (rotated)
        br_rank = pygame.transform.rotate(self.small_font.render(rank, True, color), 180)
        br_suit = pygame.transform.rotate(self.small_font.render(symbol, True, color), 180)

        self.screen.blit(br_rank, (x + width - br_rank.get_width() - 8,
                            y + height - br_rank.get_height() - 30))
        self.screen.blit(br_suit, (x + width - br_suit.get_width() - 10,
                            y + height - br_suit.get_height() - 8))
        
        return card


