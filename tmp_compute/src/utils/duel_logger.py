import pandas as pd
import numpy as np

from src.utils.enums import EnvType


class DuelLogger:
    def __init__(self, player1_name: str, player2_name: str):
        self.player1_name = player1_name
        self.player2_name = player2_name

        self.counter_train = 0
        self.counter_test = 0

        self.player1_win_train = 0
        self.player1_win_test = 0

        self.player2_win_train = 0
        self.player2_win_test = 0

        self.steps_train = []
        self.steps_test = []

    def log(
        self,
        player1_win: bool,
        player2_win: bool,
        env_type: EnvType,
        steps: int,
    ):

        if env_type == EnvType.TRAIN:
            self.counter_train += 1
            self.steps_train.append(steps)

            self.player1_win_train += int(player1_win)
            self.player2_win_train += int(player2_win)

        elif env_type == EnvType.TEST:
            self.counter_test += 1
            self.steps_test.append(steps)

            self.player1_win_test += int(player1_win)
            self.player2_win_test += int(player2_win)

        else:
            raise ValueError(f"Nieznany typ środowiska: {env_type}")

    def steps_stats(self, steps: list[int]) -> tuple[float, float, float]:
        if len(steps) == 0:
            return 0, 0, 0

        return (
            float(np.mean(steps)),
            float(np.median(steps)),
            float(np.std(steps, ddof=1)) if len(steps) > 1 else 0.0,
        )

    def winrate(self, wins: int, counter: int) -> float:
        if counter == 0:
            return 0

        return wins / counter

    def get_result(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_mean, train_median, train_std = self.steps_stats(self.steps_train)
        test_mean, test_median, test_std = self.steps_stats(self.steps_test)

        df_player1 = pd.DataFrame([{
            "player1_name": self.player1_name,
            "player2_name": self.player2_name,

            "winrate_train": self.winrate(self.player1_win_train, self.counter_train),
            "winrate_test": self.winrate(self.player1_win_test, self.counter_test),

            "episode_length_mean_train": train_mean,
            "episode_length_median_train": train_median,
            "episode_length_std_train": train_std,

            "episode_length_mean_test": test_mean,
            "episode_length_median_test": test_median,
            "episode_length_std_test": test_std,

            "duels_train": self.counter_train,
            "duels_test": self.counter_test,
        }])

        df_player2 = pd.DataFrame([{
            "player1_name": self.player2_name,
            "player2_name": self.player1_name,

            "winrate_train": self.winrate(self.player2_win_train, self.counter_train),
            "winrate_test": self.winrate(self.player2_win_test, self.counter_test),

            "episode_length_mean_train": train_mean,
            "episode_length_median_train": train_median,
            "episode_length_std_train": train_std,

            "episode_length_mean_test": test_mean,
            "episode_length_median_test": test_median,
            "episode_length_std_test": test_std,

            "duels_train": self.counter_train,
            "duels_test": self.counter_test,
        }])

        return df_player1, df_player2