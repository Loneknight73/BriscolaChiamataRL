
from pettingzoo.test import api_test, seed_test

import BriscolaChiamata


def bc_api_test():
    env = BriscolaChiamata.env()
    api_test(env, num_cycles=10, verbose_progress=True)

def bc_seed_test():
    env_fn = BriscolaChiamata.env
    seed_test(env_fn, num_cycles=10, test_kept_state=True)


if __name__ == "__main__":
    bc_api_test()
    bc_seed_test()

