import os
import random
from dataclasses import dataclass

import pandas as pd
import torch

from src.agents.base_agent import DecisionTreeAgent
from src.agents.dueling_dqn_agent import DuelingDqnAgent
from src.env.game_env import GameEnv
from src.utils.enums import AlgoName, AlgoType, EnvType, EnemyType
from src.utils.settings import GlobalSettings
from src.utils.training_logger import TrainingLogger


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


def _save_model(agent: DuelingDqnAgent, e_name: str, steps: int) -> str:
    output_dir = os.path.join("outputs", "models", e_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, f"{e_name}_s{steps}.pt")
    agent.save(model_path)

    return model_path


def _run_eval_episode(
    env: GameEnv,
    agent: DuelingDqnAgent,
    seed_item: SeedItem,
    use_default: bool,
    custom_agent,
) -> tuple[bool, float, int]:
    env.custom_agent = custom_agent

    obs = env.reset(
        seed=seed_item.seed,
        use_default=use_default,
        reverse_positions=seed_item.reverse_positions,
    )

    done = False
    won = False
    episode_reward = 0.0
    episode_steps = 0

    while not done:
        action = agent.predict(obs)

        obs, reward, done, won = env.step_rl(action)

        episode_reward += float(reward)
        episode_steps += 1

    return won, episode_reward, episode_steps


def _evaluate(
    logger: TrainingLogger,
    env: GameEnv,
    agent: DuelingDqnAgent,
    previous_agent: DuelingDqnAgent | None,
    train_seed_buffer: SeedBuffer,
    test_seed_buffer: SeedBuffer,
    settings: GlobalSettings,
):
    agent.model.eval()

    # ==========================
    # TRAIN ENV vs BASIC
    # ==========================

    for _ in range(settings.eval_episodes_training_base_agent):
        seed_item = train_seed_buffer.draw()

        won, episode_reward, episode_steps = _run_eval_episode(
            env=env,
            agent=agent,
            seed_item=seed_item,
            use_default=True,
            custom_agent=None,
        )

        logger.log(
            agent_win=won,
            episode_reward=episode_reward,
            steps=episode_steps,
            env_type=EnvType.TRAIN,
            enemy_type=EnemyType.BASIC,
        )

    # ==========================
    # TEST ENV vs BASIC
    # ==========================

    for _ in range(settings.eval_episodes_test_base_agent):
        seed_item = test_seed_buffer.draw()

        won, episode_reward, episode_steps = _run_eval_episode(
            env=env,
            agent=agent,
            seed_item=seed_item,
            use_default=True,
            custom_agent=None,
        )

        logger.log(
            agent_win=won,
            episode_reward=episode_reward,
            steps=episode_steps,
            env_type=EnvType.TEST,
            enemy_type=EnemyType.BASIC,
        )

    # Jeśli nie ma jeszcze previous model albo advanced training nie jest aktywny,
    # pomijamy ewaluację przeciwko poprzedniemu modelowi.
    if previous_agent is None:
        return

    if not logger.is_advanced_training_active():
        return

    previous_agent.model.eval()

    # ==========================
    # TRAIN ENV vs PREVIOUS
    # ==========================

    for _ in range(settings.eval_episodes_training_previous_agent):
        seed_item = train_seed_buffer.draw()

        won, episode_reward, episode_steps = _run_eval_episode(
            env=env,
            agent=agent,
            seed_item=seed_item,
            use_default=False,
            custom_agent=previous_agent,
        )

        logger.log(
            agent_win=won,
            episode_reward=episode_reward,
            steps=episode_steps,
            env_type=EnvType.TRAIN,
            enemy_type=EnemyType.PREVIOUS_MODEL,
        )

    # ==========================
    # TEST ENV vs PREVIOUS
    # ==========================

    for _ in range(settings.eval_episodes_test_previous_agent):
        seed_item = test_seed_buffer.draw()

        won, episode_reward, episode_steps = _run_eval_episode(
            env=env,
            agent=agent,
            seed_item=seed_item,
            use_default=False,
            custom_agent=previous_agent,
        )

        logger.log(
            agent_win=won,
            episode_reward=episode_reward,
            steps=episode_steps,
            env_type=EnvType.TEST,
            enemy_type=EnemyType.PREVIOUS_MODEL,
        )


def train_dueling_dqn(
    e_name: str,
    training_seeds: list[int | str],
    test_seeds: list[int | str],
    settings: GlobalSettings,
    reverse_positions: bool = True,
) -> pd.DataFrame:
    rng = random.Random(e_name)

    training_seed_buffer = SeedBuffer(
        seeds=training_seeds,
        reverse_positions=reverse_positions,
        rng=random.Random(f"{e_name}_training"),
    )

    test_seed_buffer = SeedBuffer(
        seeds=test_seeds,
        reverse_positions=reverse_positions,
        rng=random.Random(f"{e_name}_test"),
    )

    default_agent = DecisionTreeAgent(settings=settings)

    first_seed = training_seed_buffer.draw()

    env = GameEnv(
        settings=settings,
        seed=first_seed.seed,
        default_agent=default_agent,
        custom_agent=None,
        use_default=True,
        reverse_positions=first_seed.reverse_positions,
    )

    obs = env.reset(
        seed=first_seed.seed,
        use_default=True,
        reverse_positions=first_seed.reverse_positions,
    )

    input_size = int(obs.numel())

    agent = DuelingDqnAgent(
        input_size=input_size,
        settings=settings,
        name="Dueling DQN",
    )

    logger = TrainingLogger(
        experiment_name=e_name,
        settings=settings,
        algo_name=AlgoName.DUELING_DQN,
        algo_type=AlgoType.Q_VALUE,
    )

    previous_agent: DuelingDqnAgent | None = None

    training_steps = 0
    next_eval_step = settings.next_eval_step(training_steps)

    while training_steps < settings.total_max_training_steps:
        seed_item = training_seed_buffer.draw()

        advanced_training = logger.is_advanced_training_active()

        use_previous = (
            advanced_training
            and previous_agent is not None
            and rng.random() < settings.advanced_model_chance
        )

        use_default = not use_previous
        custom_agent = previous_agent if use_previous else None

        env.custom_agent = custom_agent

        obs = env.reset(
            seed=seed_item.seed,
            use_default=use_default,
            reverse_positions=seed_item.reverse_positions,
        )

        done = False

        while not done and training_steps < settings.total_max_training_steps:
            action = agent.select_action(obs, step=training_steps)

            next_obs, reward, done, won = env.step_rl(action)

            agent.remember(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                done=done,
            )

            if training_steps % settings.optimize_every == 0:
                agent.optimize()

            training_steps += 1
            agent.training_steps = training_steps

            if training_steps % settings.target_update_freq == 0:
                agent.update_target_model()

            obs = next_obs

        # Ewaluacja po epizodzie, gdy przekroczono próg.
        if training_steps >= next_eval_step:
            model_path = _save_model(
                agent=agent,
                e_name=e_name,
                steps=training_steps,
            )

            logger.new_evaluation(
                steps=training_steps,
                model_path=model_path,
            )

            _evaluate(
                logger=logger,
                env=env,
                agent=agent,
                previous_agent=previous_agent,
                train_seed_buffer=training_seed_buffer,
                test_seed_buffer=test_seed_buffer,
                settings=settings,
            )

            logger.calculate_eval()

            previous_agent = agent.clone_for_eval()

            next_eval_step = settings.next_eval_step(training_steps)

    return logger.get_result()
