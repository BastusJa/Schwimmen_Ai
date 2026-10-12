from Card import Card
import numpy as np

class CardDeck:
    def __init__(self):
        cards = self.__generate_all_cards()
        self.__cards = cards

    def __generate_all_cards(self):
        cards = []
        symbols = np.array(["A", "K", "Q", "J", "10", "9", "8", "7"])
        suits = np.array(["Clubs", "Hearts", "Spades", "Diamonds"])
        for suit in suits:
            for symbol in symbols:
                cards.append(Card(symbol, suit))

        return np.array(cards)

    def draw(self):
        card = None
        if len(self.__cards) > 1:
            card, self.__cards = self.__cards[0], self.__cards[1:]
        else:
            card = self.__cards[0]
        return card
        
    def shuffle(self):
        np.random.shuffle(self.__cards)