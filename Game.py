#
#  Card/Game rules/generic stuff to be moved to separate files
#
import random
from enum import Enum
import numpy as np


class GameState(Enum):
    BIDDING = 0
    TRICK = 1


class Rank:
    def __init__(self, rank, points, name, shortname=''):
        self.rank = rank
        self.points = points
        self.name = name
        if (shortname == ''):
            self.shortname = name[0]
        else:
            self.shortname = shortname


class Suit:
    def __init__(self, name, shortname=''):
        self.name = name
        if (shortname == ''):
            self.shortname = name[0]
        else:
            self.shortname = shortname


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __str__(self):
        return "{0} di {1}".format(self.rank.name, self.suit.name)

    def card_rank(self):
        return self.rank.rank

    def points(self):
        return self.rank.points

    def shortname(self):
        return "{0}{1}".format(self.rank.shortname, self.suit.shortname)


class Deck:
    ranks = [
        Rank(9, 11, "Asso", "A"),
        Rank(8, 10, "Tre", "3"),
        Rank(7, 4, "Re", "R"),
        Rank(6, 3, "Cavallo", "C"),
        Rank(5, 2, "Fante", "F"),
        Rank(4, 0, "Sette", "7"),
        Rank(3, 0, "Sei", "6"),
        Rank(2, 0, "Cinque", "5"),
        Rank(1, 0, "Quattro", "4"),
        Rank(0, 0, "Due", "2")
    ]

    suits = [
        Suit("Denari"),
        Suit("Spade"),
        Suit("Bastoni"),
        Suit("Coppe")
    ]

    def __init__(self):
        self.deck = [Card(r, s) for s in self.suits for r in self.ranks]


# Utility class for functionality specific to the game rules
class Rules:
    NUM_PLAYERS = 5

    def __init__(self):
        pass

    # Accepts a list of 5 cards in the order in which they were played
    # during the game, and the trump suit.
    # Returns a tuple with:
    #  - the position of the winning card in trick
    #  - the total points in the trick
    def winning_card(self, trick, trump):
        assert (len(trick) == self.NUM_PLAYERS)
        win_card_index = 0
        win_card = trick[win_card_index]
        for i in range(1, len(trick)):
            c_beats = False
            c = trick[i]
            c_beats = (c.suit == trump and win_card.suit != trump) or \
                      (c.suit == win_card.suit and c.card_rank() > win_card.card_rank())

            if (c_beats):
                win_card_index = i
                win_card = trick[win_card_index]

        total_points = sum([x.points() for x in trick])
        return (win_card_index, total_points)


class Player:
    def __init__(self, id):
        self.hand = []
        self.points = 0
        self.id = id


class Game:

    def __init__(self):
        self.rules = Rules()
        self.np = self.rules.NUM_PLAYERS
        self.deck = Deck().deck
        self.players = []

    def init_game(self):
        random.shuffle(self.deck)
        for i in range(self.np):
            p = Player(i)
            p.hand = self.deck[8 * i: 8 * i + 8]
            self.players.append(p)

        self.trump = Deck.suits[np.random.randint(len(Deck.suits))]  # TODO: temporarily fix trump
        self.first_player = np.random.randint(self.np)  # TODO: temp fix
        self.current_player = self.first_player
        self.current_trick = []
        self.n_trick = 0
        self.done = False

    def action_to_card(self, action):
        return self.players[self.current_player].hand[action]

    def is_legal_card(self, card):
        return card in self.players[self.current_player].hand

    def remove_played_card(self, card):
        self.players[self.current_player].hand.remove(card)

    # action comes from current_player
    # TODO: temporarily action is an int representing the index of the card played by current_player
    def step(self, action):
        card = self.action_to_card(action)  # TODO: actions should be symbolic in this class
        # so this conversion from Gym space should happen in the env
        if (not self.is_legal_card(card)):
            Exception("Illegal card played {0}".format(card))
        self.current_trick.append(card)
        self.remove_played_card(card)

        if (self.current_player + 1) % self.np == self.first_player:  # Trick end
            (rel_win_id, points) = self.rules.winning_card(self.current_trick, self.trump)
            abs_win_id = (rel_win_id + self.first_player) % self.np
            self.first_player = abs_win_id
            self.current_player = abs_win_id
            self.players[abs_win_id].points += points
            self.current_trick = []
            self.n_trick += 1
        else:
            self.current_player = (self.current_player + 1) % self.np

        if self.n_trick == 8:
            self.done = True


#
# TESTS
#

def test_shuffle():
    mazzo = Deck().deck
    for c in mazzo:
        print(c)
    random.shuffle(mazzo)
    print("Ora mischio...\n\n")
    for c in mazzo:
        print(c)


def test_winning_card():
    asso = Rank(9, 11, "Asso")
    tre = Rank(8, 10, "Tre")
    re = Rank(7, 4, "Re")
    cavallo = Rank(6, 3, "Cavallo")
    fante = Rank(5, 2, "Fante")
    sette = Rank(4, 0, "Sette")
    sei = Rank(3, 0, "Sei")
    cinque = Rank(2, 0, "Cinque")
    quattro = Rank(1, 0, "Quattro")
    due = Rank(0, 0, "Due")

    denari = Suit("Denari")
    spade = Suit("Spade")
    bastoni = Suit("Bastoni")
    coppe = Suit("Coppe")

    rules = Rules()

    # No trump in trick
    trick = [Card(due, denari), Card(quattro, bastoni), Card(asso, coppe), Card(re, denari), Card(fante, denari)]
    (win, points) = rules.winning_card(trick, spade)
    assert (win == 3 and points == 17)

    # Trump in trick (not first card)
    trick = [Card(due, denari), Card(quattro, bastoni), Card(sette, coppe), Card(re, denari), Card(fante, denari)]
    (win, points) = rules.winning_card(trick, coppe)
    assert (win == 2 and points == 6)

    # Swap trump and otherwise winning card
    trick = [Card(due, denari), Card(quattro, bastoni), Card(re, denari), Card(sette, coppe), Card(fante, denari)]
    (win, points) = rules.winning_card(trick, coppe)
    assert (win == 3 and points == 6)

    # Trump in trick (first card) and no other trumps
    trick = [Card(due, denari), Card(quattro, bastoni), Card(sette, coppe), Card(re, spade), Card(fante, bastoni)]
    (win, points) = rules.winning_card(trick, denari)
    assert (win == 0 and points == 6)

    # 2 trumps in trick (not first card)
    trick = [Card(due, denari), Card(re, bastoni), Card(asso, denari), Card(re, spade), Card(tre, bastoni)]
    (win, points) = rules.winning_card(trick, bastoni)
    assert (win == 4 and points == 29)


if __name__ == "__main__":
    # test_shuffle()
    test_winning_card()
