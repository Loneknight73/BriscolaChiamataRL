import numpy as np


class RandomAgent():
    def __init__(self, player_id):
        self.player_id = player_id

    def reset(self):
        pass

    def act(self, obs):
        hand = obs.players[self.player_id].hand
        index = np.random.randint(len(hand))
        return index  # TODO
