import random

import numpy as np
from gym.spaces import Discrete, Dict, Box
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

#
# Env definition
#
from Game import Game, Deck, GameAction, GameState, Bid, BidType


def env():
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = BriscolaChiamataEnv()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


# Offsets in action space
# TODO: should they be inside the class?
TRICK_ACTIONS = 40
BID_OFFSET = TRICK_ACTIONS
BID_ACTIONS = 11
TOTAL_ACTIONS = BID_OFFSET + BID_ACTIONS


#
# TODO: this Env should be as independent as possible from
# NN models peculiarities. For these, build a wrapper instead.
# For the moment I keep the 2 concerns intertwined because I
# don't know how to decouple the NN aspect from the env.
# For example, rllib suggests to use wrappers around the environment,
# but how can this work if the agents are mixed (e.g. 2 NN and 3 humans)?
#
class BriscolaChiamataEnv(AECEnv):
    '''
    The metadata holds environment constants. From gym, we inherit the "render.modes",
    metadata which specifies which modes can be put into the render() method.
    At least human mode should be supported.
    The "name" metadata allows the environment to be pretty printed.
    '''
    metadata = {'render.modes': ['human'], "name": "bc_v0"}

    def __init__(self):
        '''
        The init method takes in environment arguments and
         should define the following attributes:
        - possible_agents
        - action_spaces
        - observation_spaces

        These attributes should not be changed after initialization.
        '''
        super().__init__()
        self.rng_seed = random.randint(0, 2 ** 32 - 1)
        self.game = Game()
        self.agents = ["player_" + str(r) for r in range(self.game.np)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(len(self.agents)))))

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: Dict({
            GameState.BIDDING: Discrete(BID_ACTIONS),  # 10 cards + pass. TODO: make it multidiscrete?
            GameState.TRICK: Discrete(TRICK_ACTIONS)
        }) for agent in self.agents}
        self.observation_spaces = {agent: Dict({
            'observation': Dict({
                'gamestate': Discrete(len(GameState)),
                'player_hand': Box(low=0, high=1, shape=(TRICK_ACTIONS,), dtype=bool)
            }),
            'action_mask': Dict({
                GameState.BIDDING: Box(low=0, high=1, shape=(BID_ACTIONS,), dtype=bool),
                GameState.TRICK: Box(low=0, high=1, shape=(TOTAL_ACTIONS,), dtype=bool)
            })
        }) for agent in self.agents}
        self.reward_range = (-4, 4)  # TODO: adjust

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=1):
        self.rng_seed = seed

    def reset(self):
        """
         Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - dones
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.

        """
        self.agents = self.possible_agents[:]
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}

        self.game.seed(self.rng_seed)
        self.game.init_game()
        self.agent_selection = self.agents[self.game.current_player]

    def convert_action(self, action):
        state = self.game.gamestate
        a = action[state]
        if (state == GameState.BIDDING):
            if (a == 10):
                x = Bid(BidType.PASS)
            else:
                x = Bid(BidType.RANK, a)
        else:
            x = Deck.get_card_from_index(a)

        ga = GameAction(self.game.gamestate, x)
        return ga

    def step(self, action):
        """
         step(action) takes in an action for the current agent (specified by
         agent_selection) and needs to update
         - rewards
         - _cumulative_rewards (accumulating the rewards)
         - dones
         - infos
         - agent_selection (to the next agent)
         And any internal state used by observe() or render()
        """

        if self.dones[self.agent_selection]:
            # handles stepping an agent which is already done
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next done agent,  or if there are no more done agents, to the next live agent
            return self._was_done_step(action)

        if not self.action_space(self.agent_selection).contains(action):
            raise Exception("Action not in action_space")
        agent = self.agent_name_mapping[self.agent_selection]

        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0

        game_action = self.convert_action(action)
        self.game.step(game_action)
        self.agent_selection = self.agents[self.game.current_player]
        if (self.game.done):
            self.dones = {agent: True for agent in self.agents}
            for i, agent in enumerate(self.agents):
                self.rewards[agent] = self.game.game_points[i]
        else:
            self._clear_rewards()

        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def observe(self, agent):
        # TODO: convert Game observation to the format of observation_spaces
        # as defined in __init__
        # Observation
        bid_mask   = np.zeros(BID_ACTIONS, 'bool')
        trick_mask = np.zeros(TRICK_ACTIONS, 'bool')

        game = self.game
        # Action masks
        if (game.gamestate == GameState.BIDDING):
            bid_mask[BID_ACTIONS - 1] = 1  # PASS is always legal
            # If player was still in play, then bidding for a lower card is legal
            last_bid = game.bid_round[game.current_player]
            if (last_bid.type != BidType.PASS):
                highest_bid = game.highest_bid
                if (highest_bid.type == BidType.NONE):
                    top_bid = 10
                else:
                    top_bid = highest_bid.rank
                for i in range(top_bid):
                    bid_mask[i] = 1
        elif (game.gamestate == GameState.TRICK):
            hand = game.get_player_hand(self.agent_name_mapping[agent])
            for c in hand:
                trick_mask[Deck.get_index_from_card(c)] = 1

        mask_dict = {
            GameState.BIDDING: bid_mask,
            GameState.TRICK: trick_mask
        }
        obs_dict = {
            'gamestate': game.gamestate,
            'player_hand': trick_mask
        }
        return {'observation': obs_dict, 'action_mask': mask_dict}

    def render_bidding_round(self):
        s = "Bidding round:\n"
        s += "{0:^10} {1:^10} {2:^10} {3:^10} {4:^10}\n".format(*range(self.game.np))
        for b in self.game.bid_round:
            s += "{0:^10} ".format(str(b))
        s += "\n"
        return s

    def render_trick(self):
        game = self.game
        if (game.done):
            for i in range(game.np):
                print("Player {0} total points: {1}".format(i, game.players[i].points))
            s = "Game points:\n"
            s += "{0:^10} {1:^10} {2:^10} {3:^10} {4:^10}\n".format(*range(game.np))
            s += "{0:^10} {1:^10} {2:^10} {3:^10} {4:^10}\n".format(*game.game_points)
            print(s)
        else:
            for i in range(game.np):
                print("Player {0} cards: ".format(i), end='')
                for c in game.players[i].hand:
                    print("{0} ".format(c.shortname()), end='')
                print("")
            s = "Current trick:\n"
            s += "{0:^10} {1:^10} {2:^10} {3:^10} {4:^10}\n".format(*range(game.np))
            trick = ["" for i in range(game.np)]
            for i, c in enumerate(game.current_trick):
                trick[(game.first_player + i) % game.np] = c.shortname()
            s += "{0:^10} {1:^10} {2:^10} {3:^10} {4:^10}\n".format(*trick)
            print(s)
            if (len(self.game.current_trick) == 0 and
                    self.game.n_trick > 0):
                print("Trick won by {0}".format(self.game.first_player))
            print("")

    def render(self, mode='human'):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        game = self.game
        s = ""
        if (game.gamestate == GameState.BIDDING):
            s += self.render_bidding_round()
            print(s)
            # TODO: the last pass in the bidding round is not printed
        else:
            s = ""
            if (game.n_trick == 0 and game.current_player == game.first_player):
                # Print the last bidding round and the bidding results
                s += self.render_bidding_round()
                s += "\nCaller = {}".format(game.caller)
                s += "\n\n"
                print(s)
            self.render_trick()


def close(self):
    '''
    Close should release any graphical displays, subprocesses, network connections
    or any other environment data which should not be kept around after the
    user is no longer using the environment.
    '''
    pass
