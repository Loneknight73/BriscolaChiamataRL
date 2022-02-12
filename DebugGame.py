from BriscolaChiamata import BriscolaChiamataEnv
from RandomAgent import RandomAgent


def main():
    e = BriscolaChiamataEnv()
    e.reset()

    players = [RandomAgent(i) for i in range(5)]
    winner = -1
    e.render()
    for i, agent in enumerate(e.agent_iter()):
        print(i)
        obs, rew, done, info = e.last()
        if done:
            continue
        e.step(players[e.agent_name_mapping[agent]].act(obs))
        if all(e.dones.values()):
            winner = agent
            break
        e.render()
    e.render()


if __name__ == "__main__":
    main()
