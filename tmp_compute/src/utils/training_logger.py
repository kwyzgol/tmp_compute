from enum import Enum

import numpy as np
import pandas as pd

from src.utils.enums import AlgoName, AlgoType, EnvType, EnemyType


class TrainingLogger:
    def __init__(
        self,
        experiment_name: str,
        settings,
        algo_name: AlgoName,
        algo_type: AlgoType,
    ):
        self.experiment_name = experiment_name
        self.settings = settings
        self.algo_name = algo_name
        self.algo_type = algo_type

        # DataFrame z gotowymi, zagregowanymi wynikami kolejnych ewaluacji.
        self.df = pd.DataFrame()

        # Dane aktualnie trwającej ewaluacji.
        self.current_training_steps: int | None = None
        self.current_model_path: str | None = None

        # Stan zaawansowanego treningu.
        self.advanced_training_counter = 0
        self.advanced_training = False

        self._reset_eval_state()

    # =============================
    # Reset stanu jednej ewaluacji
    # =============================

    def _reset_eval_state(self):
        # Liczniki liczby rozegranych epizodów w danej kategorii.
        self.counter_train_basic = 0
        self.counter_train_previous = 0
        self.counter_test_basic = 0
        self.counter_test_previous = 0

        # Liczniki zwycięstw agenta w danej kategorii.
        self.agent_win_train_basic = 0
        self.agent_win_train_previous = 0
        self.agent_win_test_basic = 0
        self.agent_win_test_previous = 0

        # Listy nagród z epizodów.
        self.rewards_train_basic = []
        self.rewards_train_previous = []
        self.rewards_test_basic = []
        self.rewards_test_previous = []

        # Listy długości epizodów.
        self.episode_steps_train_basic = []
        self.episode_steps_train_previous = []
        self.episode_steps_test_basic = []
        self.episode_steps_test_previous = []

    # =============================
    # Nowa ewaluacja
    # =============================

    def new_evaluation(self, steps: int, model_path: str):
        """
        Rozpoczyna zbieranie danych dla nowej ewaluacji modelu.

        Parametr steps oznacza liczbę kroków treningu od początku treningu,
        np. 50_000, 100_000, 150_000.
        """
        self.current_training_steps = steps
        self.current_model_path = model_path
        self._reset_eval_state()

    # =============================
    # Logowanie jednego epizodu ewaluacyjnego
    # =============================

    def log(
        self,
        agent_win: bool,
        episode_reward: float,
        steps: int,
        env_type: EnvType,
        enemy_type: EnemyType,
    ):
        """
        Zapisuje wynik jednego epizodu ewaluacyjnego.

        Parametr steps oznacza długość konkretnego epizodu,
        a nie liczbę kroków treningu modelu.
        """
        if self.current_training_steps is None:
            raise RuntimeError("Najpierw wywołaj new_evaluation(steps, model_path).")

        if env_type == EnvType.TRAIN and enemy_type == EnemyType.BASIC:
            self.counter_train_basic += 1
            self.agent_win_train_basic += int(agent_win)
            self.rewards_train_basic.append(episode_reward)
            self.episode_steps_train_basic.append(steps)

        elif env_type == EnvType.TRAIN and enemy_type == EnemyType.PREVIOUS_MODEL:
            self.counter_train_previous += 1
            self.agent_win_train_previous += int(agent_win)
            self.rewards_train_previous.append(episode_reward)
            self.episode_steps_train_previous.append(steps)

        elif env_type == EnvType.TEST and enemy_type == EnemyType.BASIC:
            self.counter_test_basic += 1
            self.agent_win_test_basic += int(agent_win)
            self.rewards_test_basic.append(episode_reward)
            self.episode_steps_test_basic.append(steps)

        elif env_type == EnvType.TEST and enemy_type == EnemyType.PREVIOUS_MODEL:
            self.counter_test_previous += 1
            self.agent_win_test_previous += int(agent_win)
            self.rewards_test_previous.append(episode_reward)
            self.episode_steps_test_previous.append(steps)

        else:
            raise ValueError(f"Nieznana kombinacja env_type={env_type}, enemy_type={enemy_type}")

    # =============================
    # Funkcje pomocnicze
    # =============================

    @staticmethod
    def _winrate(wins: int, counter: int) -> float:
        if counter == 0:
            return 0.0
        return wins / counter

    @staticmethod
    def _stats(values: list[float]) -> tuple[float, float, float]:
        if len(values) == 0:
            return 0.0, 0.0, 0.0

        return (
            float(np.mean(values)),
            float(np.median(values)),
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        )

    @staticmethod
    def _combine(a: list, b: list) -> list:
        return a + b

    @staticmethod
    def _enum_value(value):
        if isinstance(value, Enum):
            return value.value
        return value

    # =============================
    # Advanced training
    # =============================

    def _update_advanced_training_state(self, record: dict):
        if self.advanced_training:
            return

        if not self.settings.advanced_training_possible:
            return

        threshold = self.settings.advanced_training_winrate_threshold
        required_evaluations = self.settings.advanced_training_required_evaluations

        current_winrate = record["winrate_train_basic"]

        if current_winrate >= threshold:
            self.advanced_training_counter += 1
        else:
            self.advanced_training_counter = 0

        if self.advanced_training_counter >= required_evaluations:
            self.advanced_training = True

    # =============================
    # Obliczenie wyniku jednej ewaluacji
    # =============================

    def calculate_eval(self) -> dict:
        if self.current_training_steps is None:
            raise RuntimeError("Najpierw wywołaj new_evaluation(steps, model_path).")

        train_wins = self.agent_win_train_basic + self.agent_win_train_previous
        train_counter = self.counter_train_basic + self.counter_train_previous

        test_wins = self.agent_win_test_basic + self.agent_win_test_previous
        test_counter = self.counter_test_basic + self.counter_test_previous

        basic_wins = self.agent_win_train_basic + self.agent_win_test_basic
        basic_counter = self.counter_train_basic + self.counter_test_basic

        previous_wins = self.agent_win_train_previous + self.agent_win_test_previous
        previous_counter = self.counter_train_previous + self.counter_test_previous

        rewards_train_basic = self._stats(self.rewards_train_basic)
        rewards_train_previous = self._stats(self.rewards_train_previous)
        rewards_test_basic = self._stats(self.rewards_test_basic)
        rewards_test_previous = self._stats(self.rewards_test_previous)

        rewards_train = self._stats(
            self._combine(self.rewards_train_basic, self.rewards_train_previous)
        )
        rewards_test = self._stats(
            self._combine(self.rewards_test_basic, self.rewards_test_previous)
        )
        rewards_basic = self._stats(
            self._combine(self.rewards_train_basic, self.rewards_test_basic)
        )
        rewards_previous = self._stats(
            self._combine(self.rewards_train_previous, self.rewards_test_previous)
        )

        episode_steps_train_basic = self._stats(self.episode_steps_train_basic)
        episode_steps_train_previous = self._stats(self.episode_steps_train_previous)
        episode_steps_test_basic = self._stats(self.episode_steps_test_basic)
        episode_steps_test_previous = self._stats(self.episode_steps_test_previous)

        episode_steps_train = self._stats(
            self._combine(self.episode_steps_train_basic, self.episode_steps_train_previous)
        )
        episode_steps_test = self._stats(
            self._combine(self.episode_steps_test_basic, self.episode_steps_test_previous)
        )
        episode_steps_basic = self._stats(
            self._combine(self.episode_steps_train_basic, self.episode_steps_test_basic)
        )
        episode_steps_previous = self._stats(
            self._combine(self.episode_steps_train_previous, self.episode_steps_test_previous)
        )

        record = {
            # Podstawowe informacje o ewaluacji.
            "experiment_name": self.experiment_name,
            "training_steps": self.current_training_steps,
            "algo_name": self._enum_value(self.algo_name),
            "algo_type": self._enum_value(self.algo_type),

            # Stałe ustawienia architektury / algorytmu.
            "activation": self._enum_value(self.settings.f_activation),
            "hidden_dims": self.settings.hidden_dims,
            "hidden_layers": self.settings.hidden_layers,
            "n_step": self.settings.n_step,
            "model_path": self.current_model_path,

            # Winrate według środowiska i przeciwnika.
            "winrate_train_basic": self._winrate(self.agent_win_train_basic, self.counter_train_basic),
            "winrate_train_previous": self._winrate(self.agent_win_train_previous, self.counter_train_previous),
            "winrate_test_basic": self._winrate(self.agent_win_test_basic, self.counter_test_basic),
            "winrate_test_previous": self._winrate(self.agent_win_test_previous, self.counter_test_previous),

            # Winrate zagregowany.
            "winrate_train": self._winrate(train_wins, train_counter),
            "winrate_test": self._winrate(test_wins, test_counter),
            "winrate_basic": self._winrate(basic_wins, basic_counter),
            "winrate_previous": self._winrate(previous_wins, previous_counter),

            # Liczba epizodów ewaluacyjnych.
            "episodes_train_basic": self.counter_train_basic,
            "episodes_train_previous": self.counter_train_previous,
            "episodes_test_basic": self.counter_test_basic,
            "episodes_test_previous": self.counter_test_previous,
            "episodes_train": train_counter,
            "episodes_test": test_counter,
            "episodes_basic": basic_counter,
            "episodes_previous": previous_counter,

            # Nagrody - train/basic.
            "reward_mean_train_basic": rewards_train_basic[0],
            "reward_median_train_basic": rewards_train_basic[1],
            "reward_std_train_basic": rewards_train_basic[2],

            # Nagrody - train/previous.
            "reward_mean_train_previous": rewards_train_previous[0],
            "reward_median_train_previous": rewards_train_previous[1],
            "reward_std_train_previous": rewards_train_previous[2],

            # Nagrody - test/basic.
            "reward_mean_test_basic": rewards_test_basic[0],
            "reward_median_test_basic": rewards_test_basic[1],
            "reward_std_test_basic": rewards_test_basic[2],

            # Nagrody - test/previous.
            "reward_mean_test_previous": rewards_test_previous[0],
            "reward_median_test_previous": rewards_test_previous[1],
            "reward_std_test_previous": rewards_test_previous[2],

            # Nagrody zagregowane.
            "reward_mean_train": rewards_train[0],
            "reward_median_train": rewards_train[1],
            "reward_std_train": rewards_train[2],
            "reward_mean_test": rewards_test[0],
            "reward_median_test": rewards_test[1],
            "reward_std_test": rewards_test[2],
            "reward_mean_basic": rewards_basic[0],
            "reward_median_basic": rewards_basic[1],
            "reward_std_basic": rewards_basic[2],
            "reward_mean_previous": rewards_previous[0],
            "reward_median_previous": rewards_previous[1],
            "reward_std_previous": rewards_previous[2],

            # Długość epizodu - train/basic.
            "episode_length_mean_train_basic": episode_steps_train_basic[0],
            "episode_length_median_train_basic": episode_steps_train_basic[1],
            "episode_length_std_train_basic": episode_steps_train_basic[2],

            # Długość epizodu - train/previous.
            "episode_length_mean_train_previous": episode_steps_train_previous[0],
            "episode_length_median_train_previous": episode_steps_train_previous[1],
            "episode_length_std_train_previous": episode_steps_train_previous[2],

            # Długość epizodu - test/basic.
            "episode_length_mean_test_basic": episode_steps_test_basic[0],
            "episode_length_median_test_basic": episode_steps_test_basic[1],
            "episode_length_std_test_basic": episode_steps_test_basic[2],

            # Długość epizodu - test/previous.
            "episode_length_mean_test_previous": episode_steps_test_previous[0],
            "episode_length_median_test_previous": episode_steps_test_previous[1],
            "episode_length_std_test_previous": episode_steps_test_previous[2],

            # Długość epizodu zagregowana.
            "episode_length_mean_train": episode_steps_train[0],
            "episode_length_median_train": episode_steps_train[1],
            "episode_length_std_train": episode_steps_train[2],
            "episode_length_mean_test": episode_steps_test[0],
            "episode_length_median_test": episode_steps_test[1],
            "episode_length_std_test": episode_steps_test[2],
            "episode_length_mean_basic": episode_steps_basic[0],
            "episode_length_median_basic": episode_steps_basic[1],
            "episode_length_std_basic": episode_steps_basic[2],
            "episode_length_mean_previous": episode_steps_previous[0],
            "episode_length_median_previous": episode_steps_previous[1],
            "episode_length_std_previous": episode_steps_previous[2],

            # Advanced training zostanie zaktualizowany przed dopisaniem rekordu.
            "advanced_training": False,
        }

        self._update_advanced_training_state(record)
        record["advanced_training"] = self.advanced_training

        self.df = pd.concat(
            [self.df, pd.DataFrame([record])],
            ignore_index=True,
        )

        return record

    # =============================
    # Sprawdzenie trybu zaawansowanego treningu
    # =============================

    def is_advanced_training_active(self) -> bool:
        """
        Zwraca informację, czy zaawansowany trening został już aktywowany.

        Metoda może być używana przez główny skrypt treningowy do wyboru trybu treningu,
        np. czy agent ma trenować tylko przeciwko bazowemu przeciwnikowi,
        czy również przeciwko poprzednim wersjom modelu.
        """
        return self.advanced_training

    # =============================
    # Pobranie wyników
    # =============================

    def get_result(self) -> pd.DataFrame:
        return self.df.copy()
