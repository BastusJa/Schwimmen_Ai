from Card import Card
import numpy as np

class CardDeck:
    def __init__(self):
        self.__cards = self.__generate_all_cards()

    def __generate_all_cards(self):
        cards = []
        symbols = ["A", "K", "Q", "J", "10", "9", "8", "7"]
        suits = ["Clubs", "Hearts", "Spades", "Diamonds"]
        for suit in suits:
            for symbol in symbols:
                cards.append(Card(symbol, suit))

        return cards

    def draw(self):
        card = None
        if not self.__cards:
            return None
        return self.__cards.pop()
        
    def shuffle(self):
        np.random.shuffle(self.__cards)