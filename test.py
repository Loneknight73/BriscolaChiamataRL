
from pettingzoo.test import api_test, seed_test

from BriscolaChiamata import BriscolaChiamataEnv

def bc_api_test():
    env = BriscolaChiamataEnv()
    api_test(env, num_cycles=10, verbose_progress=True)

def bc_seed_test():
    env_fn = BriscolaChiamataEnv
    seed_test(env_fn, num_cycles=10, test_kept_state=True)


if __name__ == "__main__":
    bc_api_test()
    #bc_seed_test()

