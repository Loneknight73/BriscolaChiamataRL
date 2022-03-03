import numpy as np
import random


class RandomAgent():
    def __init__(self, player_id):
        self.player_id = player_id

    def reset(self):
        pass

    def act(self, obs):
        c = random.choice(np.flatnonzero(obs['action_mask']))
        return c
