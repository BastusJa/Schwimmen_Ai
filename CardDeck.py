from Card import Card
import numpy as np

class CardDeck:
    def __init__(self):
        cards = self.__generate_all_cards()
        self.__cards = cards

    def __generate_all_cards(self):
        cards = []
        symbols = np.array(["A", "K", "Q", "J", "10", "9", "8", "7"])
        suits = np.array(["Clower", "Hearth", "Pick", "Caro"])
        for suit in suits:
            for symbol in symbols:
                cards.append(Card(symbol, suit))

        return np.array(cards)

    def draw(self):
        card, self.__cards = self.__cards[0], self.__cards[1:]
        # Log played card
        with open('card_history.txt', 'a') as f:
            f.write(f"{card.symbol} of {card.suit}\n")
        return card
        
    def shuffle(self):
        np.random.shuffle(self.__cards)