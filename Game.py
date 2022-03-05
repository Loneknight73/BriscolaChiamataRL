#
#  Card/Game rules/generic stuff to be moved to separate files
#
import random
from enum import Enum, IntEnum
import numpy as np


class GameState(IntEnum):
    BIDDING = 0,
    CHOOSE_TRUMP = 1,
    TRICK = 2


class Rank:
    def __init__(self, rank, points, name, shortname=''):
        self.rank = rank
        self.points = points
        self.name = name
        if (shortname == ''):
            self.shortname = name[0]
        else:
            self.shortname = shortname

    def __eq__(self, other):
        if (isinstance(other, Rank)):
            return self.rank == other.rank

    def __lt__(self, other):
        if (isinstance(other, Rank)):
            return self.rank < other.rank

class Suit:
    def __init__(self, name, shortname=''):
        self.name = name
        if (shortname == ''):
            self.shortname = name[0]
        else:
            self.shortname = shortname

    def __eq__(self, other):
        if (isinstance(other, Suit)):
            return self.name == other.name


class Card:
    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit

    def __eq__(self, other):
        if (isinstance(other, Card)):
            return self.rank == other.rank and self.suit == other.suit

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
        Rank(0, 0, "Due", "2"),
        Rank(1, 0, "Quattro", "4"),
        Rank(2, 0, "Cinque", "5"),
        Rank(3, 0, "Sei", "6"),
        Rank(4, 0, "Sette", "7"),
        Rank(5, 2, "Fante", "F"),
        Rank(6, 3, "Cavallo", "C"),
        Rank(7, 4, "Re", "R"),
        Rank(8, 10, "Tre", "3"),
        Rank(9, 11, "Asso", "A")
    ]

    suits = [
        Suit("Denari"),
        Suit("Spade"),
        Suit("Bastoni"),
        Suit("Coppe")
    ]

    def __init__(self):
        self.deck = [Card(r, s) for s in self.suits for r in self.ranks]

    @staticmethod
    def get_indexes(card):
        """
        :param card:
        :return: tuple with the indexes of the Rank and Suit of card
        """
        ri = Deck.ranks.index(card.rank)
        si = Deck.suits.index(card.suit)
        return ri, si

    @staticmethod
    def get_card_from_index(i):
        r = Deck.ranks[i % 10]
        s = Deck.suits[i // 10]
        return Card(r, s)

    @staticmethod
    def get_suit_from_index(i):
        return Deck.suits[i]

    @staticmethod
    def get_rank_from_index(i):
        return Deck.ranks[i]

    @staticmethod
    def get_index_from_card(c):
        ri, si = Deck.get_indexes(c)
        return si * 10 + ri


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

class BidType(Enum):
    NONE = 0,
    RANK = 1,
    PASS = 2

class Bid:
    def __init__(self, type, rank=None):
        self.type = type
        self.rank = rank

    def __str__(self):
        if (self.type == BidType.NONE):
            return "NONE"
        elif (self.type == BidType.PASS):
            return "PASS"
        else:
            return str(self.rank.shortname)

class TrickInfo:
    def __init__(self, cards, first_player, winner, points):
        self.cards = cards
        self.first_player = first_player
        self.winner = winner
        self.points = points

class GameAction:
    def __init__(self, phase, action):
        if (phase == GameState.BIDDING):
            self.bid = action
        elif (phase == GameState.CHOOSE_TRUMP):
            self.trump = action
        elif (phase == GameState.TRICK):
            self.card = action

    def get_card(self):
        return self.card

    def get_bid(self):
        return self.bid

    def get_trump(self):
        return self.trump

class Game:

    def __init__(self, seed=None):
        self.rules = Rules()
        self.np = self.rules.NUM_PLAYERS
        self.deck = Deck().deck
        self.players = []
        self.rng = random

    def seed(self, seed=None):
        if (seed is None):
            self.rng = random
        else:
            self.rng = random.Random(seed)

    def init_game(self):
        self.deck = Deck().deck
        self.rng.shuffle(self.deck)
        self.players = []
        for i in range(self.np):
            p = Player(i)
            p.hand = self.deck[8 * i: 8 * i + 8]
            p.hand.sort(key = lambda c: -Deck.get_index_from_card(c))
            self.players.append(p)

        # TODO: temp fix; not clear actually how the first_player should be set; maybe it should be passed from
        # whoever builds the environment
        self.first_player = self.rng.randrange(0, self.np)
        self.current_player = self.first_player
        self.tricks = []  # List of TrickInfo objects describing already completed tricks
        self.current_trick = []
        self.n_trick = 0
        self.done = False
        self.gamestate = GameState.BIDDING
        self.bid_round = [Bid(BidType.NONE) for i in range(self.np)]
        self.highest_bid = Bid(BidType.NONE)
        self.caller = None
        self.partner = None
        self.trump = None
        self.partner_card = None
        self.highest_bidder = None
        self.game_points = [0 for i in range(self.np)]
        self.caller_won = None

    def is_legal_card(self, card):
        hand = self.players[self.current_player].hand
        b = card in hand
        return b

    def is_start_of_trick(self):
        return self.current_player == self.first_player

    def is_start_of_trick_phase(self):
        return self.n_trick == 0 and self.is_start_of_trick()

    def remove_played_card(self, card):
        hand = self.players[self.current_player].hand
        b = card in hand
        hand.remove(card)

    def get_player_hand(self, i):
        return self.players[i].hand

    #
    # Bid phase related functions
    #

    def is_legal_bid(self, bid):
        if (bid.type == BidType.NONE):
            raise Exception("Player {0}: Bid type NONE".format(self.current_player))
        if (bid.type == BidType.PASS):
            # Pass is always legal provided the bidding phase is ongoing
            return True
        else: # bid.type == RANK
            last_bid = self.bid_round[self.current_player]
            if (last_bid.type == BidType.PASS):
                # If the player passed last time, it cannot make an actual bid
                return False
            actual_bids = list(filter(lambda b: b.type == BidType.RANK, self.bid_round))
            if (len(actual_bids) == 0):
                return True
            else:
                highest_bid = min(actual_bids, key=lambda b: b.rank)
                if (bid.rank < highest_bid.rank):
                    return True
                else:
                    return False

    def update_bid_round(self, bid):
        self.bid_round[self.current_player] = bid
        if (bid.type == BidType.RANK):
            self.highest_bid = bid
            self.highest_bidder = self.current_player

    def step_bidding(self, action):
        bid = action.get_bid()
        if not self.is_legal_bid(bid):
            raise Exception("Player {0}: Illegal bid {1}".format(self.current_player, bid))
        self.update_bid_round(bid)
        # Is bidding phase done? TODO: must consider the case where all players pass
        actual_bids = list(filter(lambda b: b.type == BidType.RANK, self.bid_round))
        pass_bids = list(filter(lambda b: b.type == BidType.PASS, self.bid_round))
        if (len(actual_bids) == 1 and len(pass_bids) == self.np - 1):
            # The bidding phase ends here, the trick phase begins
            self.caller = self.highest_bidder
            self.gamestate = GameState.CHOOSE_TRUMP
            self.current_player = self.caller
        else:
            self.current_player = (self.current_player + 1) % self.np

    #
    # Choose trump related functions
    #
    def step_choose_trump(self, action):
        self.trump = action.get_trump()
        self.partner_card = Card(self.highest_bid.rank, self.trump)
        self.current_player = self.first_player
        self.gamestate = GameState.TRICK


    #
    # Trick phase related functions
    #

    def manage_end_game(self):
        self.done = True
        # sum points for caller and partner (if they are different)
        # and set points accordingly
        caller_points = self.players[self.caller].points
        solo_game = True if (self.caller == self.partner) else False
        if (not solo_game):
            caller_points += self.players[self.partner].points
        other_points = 0
        for p in self.players:
            if (p.id != self.caller and p.id != self.partner):
                other_points += p.points
        if (caller_points + other_points != 120):
            raise Exception("Bug: total number of points != 120")

        self.caller_won = caller_points > other_points
        if (solo_game): # Solo game
            self.game_points = [4 if p.id == self.caller else -1 for p in self.players]
            self.game_points = [-x if not self.caller_won else x for x in self.game_points]
        else: # 2 (caller + partner) vs 3 (others)
            for p in self.players:
                if p.id == self.caller:
                    self.game_points[p.id] = 2
                elif p.id == self.partner:
                    self.game_points[p.id] = 1
                else:
                    self.game_points[p.id] = -1
            self.game_points = [-x if not self.caller_won else x for x in self.game_points]


    def step_trick(self, action):
        card = action.get_card()
        # so this conversion from Gym space should happen in the env
        # print("a) Curr = {0}, card = {1}".format(self.current_player,card))
        if not self.is_legal_card(card):
            raise Exception("Player {0}: Illegal card played {1}".format(self.current_player, card))
        self.current_trick.append(card)
        # print("Trick: {0} Curr = {1}, card = {2}".format(self.n_trick, self.current_player,card))
        self.remove_played_card(card)
        # Has partner been unveiled?
        if (card == self.partner_card):
            self.partner = self.current_player

        # Manage normal trick
        if (self.current_player + 1) % self.np == self.first_player:  # Trick end
            (rel_win_id, points) = self.rules.winning_card(self.current_trick, self.trump)
            abs_win_id = (rel_win_id + self.first_player) % self.np
            trick_info = TrickInfo(self.current_trick, self.first_player, abs_win_id, points)
            self.tricks.append(trick_info)
            self.first_player = abs_win_id
            self.current_player = abs_win_id
            self.players[abs_win_id].points += points
            self.current_trick = []
            self.n_trick += 1
        else:
            self.current_player = (self.current_player + 1) % self.np

        if self.n_trick == 8:
            self.manage_end_game()


    # action comes from current_player
    def step(self, action):
        if (self.gamestate == GameState.BIDDING):
            self.step_bidding(action)
        elif (self.gamestate == GameState.CHOOSE_TRUMP):
            self.step_choose_trump(action)
        elif (self.gamestate == GameState.TRICK):
            self.step_trick(action)


#
# Utils
#

def two_hot_encode_card(card):
    encoding = np.zeros(14)
    ri, si = Deck.get_indexes(card)
    encoding[ri] = 1
    encoding[10 + si] = 1
    return encoding


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
