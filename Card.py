class Card:
    def __init__(self, symbol, suit):
        self.value = self.__get_value(symbol)
        self.suit = suit
        self.symbol = symbol
        self.name = self.__get_name(symbol)

    def __get_value(self, symbol):
        if symbol == "A":
            return 11
        elif symbol == "J" or symbol == "Q" or symbol == "10" or symbol == "K":
            return 10
        elif symbol == "9":
            return 9
        elif symbol == "8":
            return 8
        elif symbol == "7":
            return 7
        else:
            return None
        
    def __get_name(self, symbol):
        if symbol == "A":
            return "Ass"
        elif symbol == "J":
            return "Jack" 
        elif symbol == "Q":
            return "Queen"
        elif symbol == "10":
            return "Ten"
        elif symbol == "K":
            return "King"
        elif symbol == "9":
            return "Nine"
        elif symbol == "8":
            return "Eight"
        elif symbol == "7":
            return "Seven"
        else:
            return None