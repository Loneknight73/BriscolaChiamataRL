import numpy as np
import random

from Game import GameState


class RandomAgent():
    def __init__(self, player_id):
        self.player_id = player_id

    def reset(self):
        pass

    def act(self, obs):
        d = {}
        for s, am in obs['action_mask'].items():
            tmp = np.flatnonzero(am)
            d[s] = 0 if (tmp.size == 0) else np.random.choice(tmp)
        return d
