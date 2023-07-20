import random
from argparse import ArgumentParser

import laserhockey.hockey_env as h_env
import numpy as np
import torch
from agents.agent import Agent
from agents.decqn_fabrice import DecQN_Fabrice
from agents.dueling_jens import Dueling_Jens
from tqdm import trange


def get_agent_from_player(player: str) -> Agent | h_env.BasicOpponent:
    """
    Returns agent, given a player.

    Args:
        player (str): Player (Jens, Fabrice, Christoph, Weak, Strong).

    Raises:
        NotImplementedError: Christoph needs to add his agent.
        ValueError: Unknown player.

    Returns:
        Agent | h_env.BasicOpponent: Player's agent.
    """

    if player == "Jens":
        return Dueling_Jens()
    elif player == "Fabrice":
        return DecQN_Fabrice()
    elif player == "Christoph":
        # TODO
        raise NotImplementedError
    elif player == "Weak":
        return h_env.BasicOpponent(weak=True)
    elif player == "Strong":
        return h_env.BasicOpponent(weak=False)
    else:
        raise ValueError(f"Unknown player: {player}")


def main() -> None:
    main_parser = ArgumentParser(
        "AlphaPuck | Evaluation",
        description="Main script for evaluation of AlphaPuck agents.",
    )
    main_parser.add_argument(
        "--player-1",
        type=str,
        required=True,
        help="Name of player 1: Jens, Fabrice, Christoph, Weak, Strong.",
    )
    main_parser.add_argument(
        "--player-2",
        type=str,
        required=True,
        help="Name of player 1: Jens, Fabrice, Christoph, Weak, Strong.",
    )
    main_parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes.",
    )
    main_parser.add_argument(
        "--rng-seed",
        type=int,
        default=7,
        help="Random number generator seed. Set to negative values to generate a random seed.",
    )
    main_parser.add_argument(
        "--disable-rendering",
        default=False,
        action="store_true",
        help="Disables graphical rendering.",
    )
    main_parser.add_argument(
        "--disable-progress-bar",
        default=False,
        action="store_true",
        help="Disables progress bar.",
    )

    args = main_parser.parse_args()

    if args.rng_seed >= 0:
        random.seed(args.rng_seed)
        np.random.seed(args.rng_seed)
        torch.manual_seed(args.rng_seed)
    else:
        rng_seed = torch.seed()
        random.seed(rng_seed)
        np.random.seed(rng_seed)

    agent_p1 = get_agent_from_player(args.player_1)
    agent_p2 = get_agent_from_player(args.player_2)

    env = h_env.HockeyEnv(mode=h_env.HockeyEnv.NORMAL)

    win_stats = np.zeros((args.num_episodes,))
    for episode_idx in trange(args.num_episodes, disable=args.disable_progress_bar):
        env.seed(np.random.randint(0, 100_000_000))
        state_p1, _ = env.reset()
        state_p2 = env.obs_agent_two()
        terminal = False

        while not terminal:
            if not args.disable_rendering:
                env.render()

            action_c_p1 = agent_p1.act(state_p1)
            action_c_p2 = agent_p2.act(state_p2)

            state_p1, _, terminal, _, info = env.step(np.hstack([action_c_p1, action_c_p2]))
            state_p2 = env.obs_agent_two()

            if terminal:
                win_stats[episode_idx] = info["winner"]

    num_wins = (win_stats == 1).sum()
    num_draws = (win_stats == 0).sum()
    num_defeats = (win_stats == -1).sum()
    winning_percentage = (num_wins + 0.5 * num_draws) / args.num_episodes

    print(f"Wins: {num_wins} / {args.num_episodes} ({num_wins / args.num_episodes:.1%})")
    print(f"Draws: {num_draws} / {args.num_episodes} ({num_draws / args.num_episodes:.1%})")
    print(f"Defeats: {num_defeats} / {args.num_episodes} ({num_defeats / args.num_episodes:.1%})")
    print(f"Winning percentage: {winning_percentage:.1%}")

    env.close()


if __name__ == '__main__':
    main()
