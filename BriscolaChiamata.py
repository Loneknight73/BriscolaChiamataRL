from gym.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

#
# Env definition
#
from Game import Game


def env():
    '''
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = BriscolaChiamataEnv()
    # This wrapper is only for environments which print results to the terminal
    env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


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
        self.game = Game()
        self.possible_agents = ["player_" + str(r) for r in range(self.game.np)]
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(len(self.possible_agents)))))

        # Gym spaces are defined and documented here: https://gym.openai.com/docs/#spaces
        self.action_spaces = {agent: Discrete(8) for agent in
                              self.possible_agents}  # TODO: should be a dict where keys are the phases of the game
        self.observation_spaces = {agent: Discrete(8) for agent in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

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
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}

        self.game.init_game()
        self.agent_selection = self.agents[0]  # TODO

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

        # TODO: delegate the mechanics of the game to a different class, and keep here only things
        # related to PettingZoo env management
        agent = self.agent_name_mapping[self.agent_selection]

        self._cumulative_rewards[agent] = 0
        self.game.step(action)
        print("Curr = {0}".format(self.game.current_player))
        self.agent_selection = self.agents[self.game.current_player]
        if (self.game.done):
            self.dones = {agent: True for agent in self.agents}


    def observe(self, agent):
        return self.game  # TODO for the moment, pass the whole state of the game

    def render(self, mode='human'):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """

        # TODO: For now print only the cards for each player
        if (self.game.done):
            for i in range(self.game.np):
                print("Player {0} total points: {1}".format(i, self.game.players[i].points))
        else:
            for i in range(self.game.np):
                print("Player {0} cards: ".format(i), end='')
                for c in self.game.players[i].hand:
                    print("{0} ".format(c.shortname()), end='')
                print("")



    def close(self):
        '''
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        '''
        pass
