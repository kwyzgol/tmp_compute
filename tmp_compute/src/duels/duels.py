import random
from dataclasses import dataclass

import pandas as pd

from src.agents.agents import Agent
from src.env.game_env import GameEnv
from src.utils.duel_logger import DuelLogger
from src.utils.enums import AgentType, EnvType
from src.utils.settings import GlobalSettings


@dataclass
class SeedItem:
    seed: int | str
    reverse_positions: bool


class SeedBuffer:
    def __init__(
        self,
        seeds: list[int | str],
        reverse_positions: bool,
        rng: random.Random,
    ):
        self.rng = rng
        self.base_items = self._build_items(seeds, reverse_positions)
        self.available: list[SeedItem] = []
        self.used: list[SeedItem] = []

        self._refill()

    @staticmethod
    def _build_items(
        seeds: list[int | str],
        reverse_positions: bool,
    ) -> list[SeedItem]:
        items = []

        for seed in seeds:
            items.append(SeedItem(seed=seed, reverse_positions=False))

            if reverse_positions:
                items.append(SeedItem(seed=seed, reverse_positions=True))

        return items

    def _refill(self):
        self.available = self.used if self.used else self.base_items.copy()
        self.used = []
        self.rng.shuffle(self.available)

    def draw(self) -> SeedItem:
        if len(self.available) == 0:
            self._refill()

        item = self.available.pop()
        self.used.append(item)

        return item


def _select_agent1_action(env: GameEnv, agent1: Agent, obs):
    if agent1.agent_type == AgentType.BASIC:
        basic_obs = env.get_basic_obs(player_id=1)
        return agent1.predict(basic_obs)

    if agent1.agent_type == AgentType.RL:
        return agent1.predict(obs)

    raise ValueError(f"Nieznany typ agenta: {agent1.agent_type}")


def _prepare_env_for_agent2(
    agent2: Agent,
    settings: GlobalSettings,
    seed_item: SeedItem,
) -> GameEnv:
    if agent2.agent_type == AgentType.BASIC:
        return GameEnv(
            settings=settings,
            seed=seed_item.seed,
            default_agent=agent2,
            custom_agent=None,
            use_default=True,
            reverse_positions=seed_item.reverse_positions,
        )

    if agent2.agent_type == AgentType.RL:
        return GameEnv(
            settings=settings,
            seed=seed_item.seed,
            default_agent=None,
            custom_agent=agent2,
            use_default=False,
            reverse_positions=seed_item.reverse_positions,
        )

    raise ValueError(f"Nieznany typ agenta: {agent2.agent_type}")


def _run_single_duel(
    agent1: Agent,
    agent2: Agent,
    settings: GlobalSettings,
    seed_item: SeedItem,
) -> tuple[bool, bool, int]:
    env = _prepare_env_for_agent2(
        agent2=agent2,
        settings=settings,
        seed_item=seed_item,
    )

    obs = env.reset(
        seed=seed_item.seed,
        use_default=(agent2.agent_type == AgentType.BASIC),
        reverse_positions=seed_item.reverse_positions,
    )

    done = False
    won = False
    steps = 0

    while not done:
        action = _select_agent1_action(
            env=env,
            agent1=agent1,
            obs=obs,
        )

        obs, reward, done, won = env.step_rl(action)
        steps += 1

    player1_win = env.winner_id == 1
    player2_win = env.winner_id == 2

    return player1_win, player2_win, steps


def duels(
    agent1: tuple[str, Agent],
    agent2: tuple[str, Agent],
    training_seeds: list[int | str],
    test_seeds: list[int | str],
    settings: GlobalSettings,
    reverse_positions: bool,
    num_duels_training: int,
    num_duels_test: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    agent1_name, agent1_object = agent1
    agent2_name, agent2_object = agent2

    logger = DuelLogger(
        player1_name=agent1_name,
        player2_name=agent2_name,
    )

    train_seed_buffer = SeedBuffer(
        seeds=training_seeds,
        reverse_positions=reverse_positions,
        rng=random.Random(f"duels_{agent1_name}_vs_{agent2_name}_train"),
    )

    test_seed_buffer = SeedBuffer(
        seeds=test_seeds,
        reverse_positions=reverse_positions,
        rng=random.Random(f"duels_{agent1_name}_vs_{agent2_name}_test"),
    )

    for _ in range(num_duels_training):
        seed_item = train_seed_buffer.draw()

        player1_win, player2_win, steps = _run_single_duel(
            agent1=agent1_object,
            agent2=agent2_object,
            settings=settings,
            seed_item=seed_item,
        )

        logger.log(
            player1_win=player1_win,
            player2_win=player2_win,
            env_type=EnvType.TRAIN,
            steps=steps,
        )

    for _ in range(num_duels_test):
        seed_item = test_seed_buffer.draw()

        player1_win, player2_win, steps = _run_single_duel(
            agent1=agent1_object,
            agent2=agent2_object,
            settings=settings,
            seed_item=seed_item,
        )

        logger.log(
            player1_win=player1_win,
            player2_win=player2_win,
            env_type=EnvType.TEST,
            steps=steps,
        )

    return logger.get_result()