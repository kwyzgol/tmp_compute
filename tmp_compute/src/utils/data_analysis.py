import os
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scipy.stats import kruskal, mannwhitneyu
from src.utils.enums import SignificantStatus


def check_test_value (
        p: float,
        significant_threshold: float = 0.05,
        very_significant_threshold: float = 0.01,
        print_result: bool = True) -> SignificantStatus:

    significant = p < significant_threshold
    very_significant = p < very_significant_threshold

    if not significant:
        result = SignificantStatus.NOT_SIGNIFICANT
    elif very_significant:
        result = SignificantStatus.VERY_SIGNIFICANT
    else:
        result = SignificantStatus.SIGNIFICANT

    if print_result:
        if result == SignificantStatus.NOT_SIGNIFICANT:
            print(f'[p = {p:.7g}] [p >= {significant_threshold}] Różnice nie są istotne statystycznie.')
        elif result == SignificantStatus.VERY_SIGNIFICANT:
            print(f'[p = {p:.7g}] [p < {significant_threshold}] [p < {very_significant_threshold}] Różnice są bardzo istotne statystycznie.')
        elif result == SignificantStatus.SIGNIFICANT:
            print(f'[p = {p:.7g}] [p < {significant_threshold}] Różnice są istotne statystycznie.')

    return result


def test_kruskal(
        groups: list[tuple[str, list[float]]],
        significant_threshold: float = 0.05,
        very_significant_threshold: float = 0.01,
        print_result: bool = True) -> tuple[SignificantStatus, float, float]:

    names = [name for name, _ in groups]
    values = [values for _, values in groups]

    stat, p = kruskal(*values)

    if print_result:
        print('\n[ Test Kruskala-Wallisa ]')
        print('[Porównywane grupy] ' + ', '.join(names))
        print(f'[Statystyka H = {stat:.7g}]')

    result = check_test_value(
        p=p,
        significant_threshold=significant_threshold,
        very_significant_threshold=very_significant_threshold,
        print_result=print_result
    )

    return result, stat, p


def test_mannwhitneyu(
        groups: list[tuple[str, list[float]]],
        significant_threshold: float = 0.05,
        very_significant_threshold: float = 0.01,
        print_result: bool = True,
        subtitle: str | None = None) -> pd.DataFrame:

    results = []

    for (group1_name, group1_values), (group2_name, group2_values) in combinations(groups, 2):

        u, p = mannwhitneyu(
            group1_values,
            group2_values,
            alternative='two-sided'
        )

        significant_status = check_test_value(
            p=p,
            significant_threshold=significant_threshold,
            very_significant_threshold=very_significant_threshold,
            print_result=False
        )

        results.append({
            'group1': group1_name,
            'group2': group2_name,
            'u': u,
            'p': p,
            'significant_status': significant_status
        })

    df = pd.DataFrame(results)

    if print_result:
        show_mannwhitneyu_table(df, groups, subtitle)

    return df


def show_mannwhitneyu_table(
        df: pd.DataFrame,
        groups: list[tuple[str, list[float]]],
        subtitle: str | None = None) -> None:

    group_names = [name for name, _ in groups]

    cell_values = []
    cell_colors = []

    for row_group in group_names:
        row_values = []
        row_colors = []

        for col_group in group_names:

            if row_group == col_group:
                row_values.append('—')
                row_colors.append('#C8C8C8')
                continue

            pair = df[
                ((df['group1'] == row_group) & (df['group2'] == col_group)) |
                ((df['group1'] == col_group) & (df['group2'] == row_group))
            ]

            if pair.empty:
                row_values.append('')
                row_colors.append('white')
                continue

            result = pair.iloc[0]
            status = result['significant_status']
            p = result['p']
            u = result['u']

            if status == SignificantStatus.NOT_SIGNIFICANT:
                label = 'Różnice nieistotne'
                color = '#F4C1C6'
            elif status == SignificantStatus.SIGNIFICANT:
                label = 'Różnice istotne'
                color = '#BFE6BF'
            else:
                label = 'Różnice bardzo istotne'
                color = '#8FD19A'

            row_values.append(
                f'{label}<br>'
                f'p = {p:.7g}<br>'
                f'U = {u:.7g}'
            )

            row_colors.append(color)

        cell_values.append(row_values)
        cell_colors.append(row_colors)

    table_values = [
        [f'<b>{name}</b>' for name in group_names],
        *cell_values
    ]

    table_colors = [
        ['#FFD966' for _ in group_names],
        *cell_colors
    ]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[''] + [f'<b>{name}</b>' for name in group_names],
                    fill_color='#FFD966',
                    font=dict(color='black'),
                    align='center',
                    line_color='black',
                    line_width=2,
                ),
                cells=dict(
                    values=[
                        [f'<b>{name}</b>' for name in group_names],
                        *list(map(list, zip(*cell_values)))
                    ],
                    fill_color=[
                        ['#FFD966' for _ in group_names],
                        *list(map(list, zip(*cell_colors)))
                    ],
                    font=dict(color='black'),
                    align='center',
                    height=55,
                    line_color='black',
                    line_width=2,
                )
            )
        ]
    )

    title = '<b>[Test Manna-Whitneya U]</b>'

    if subtitle is not None:
        title += f'<br>{subtitle}'

    fig.update_layout(
        title=title,
        title_x=0.5,
        height = max(120 + len(group_names) * 85, 420)
    )

    fig.show()

def sort_groups_by_mean_desc(
        groups: list[tuple[str, list[float]]]) -> list[tuple[str, list[float]]]:
    return sorted(
        groups,
        key=lambda x: sum(x[1]) / len(x[1]),
        reverse=True
    )


def _load_training_results(training_file) -> pd.DataFrame:
    if isinstance(training_file, pd.DataFrame):
        return training_file.copy()

    if isinstance(training_file, (str, os.PathLike)):
        return pd.read_csv(training_file)

    raise TypeError(
        "training_file must be a pandas DataFrame or a path to a CSV file."
    )


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float) != 0

    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "t", "yes", "y"})
    )


def _mean_metric(
        training_file,
        column: str,
        *,
        previous: bool = False,
        episode_count_column: str | None = None) -> float:
    df = _load_training_results(training_file)

    if column not in df.columns:
        raise ValueError(f"Brak wymaganej kolumny: {column}")

    if previous:
        if "advanced_training" not in df.columns:
            return 0.0

        advanced_mask = _bool_series(df["advanced_training"])
        advanced_positions = np.flatnonzero(advanced_mask.to_numpy())

        if len(advanced_positions) == 0:
            return 0.0

        df = df.iloc[advanced_positions[0]:]

        if episode_count_column is not None and episode_count_column in df.columns:
            episode_counts = pd.to_numeric(
                df[episode_count_column],
                errors="coerce",
            ).fillna(0)
            df = df[episode_counts > 0]

    values = pd.to_numeric(df[column], errors="coerce").dropna()

    if values.empty:
        return 0.0

    return float(values.mean())


def calculate_mean_winrate_train_basic(training_file) -> float:
    return _mean_metric(training_file, "winrate_train_basic")


def calculate_mean_winrate_test_basic(training_file) -> float:
    return _mean_metric(training_file, "winrate_test_basic")


def calculate_mean_winrate_train_previous(training_file) -> float:
    return _mean_metric(
        training_file,
        "winrate_train_previous",
        previous=True,
        episode_count_column="episodes_train_previous",
    )


def calculate_mean_winrate_test_previous(training_file) -> float:
    return _mean_metric(
        training_file,
        "winrate_test_previous",
        previous=True,
        episode_count_column="episodes_test_previous",
    )


def calculate_mean_episode_length_train_basic(training_file) -> float:
    return _mean_metric(training_file, "episode_length_mean_train_basic")


def calculate_mean_episode_length_test_basic(training_file) -> float:
    return _mean_metric(training_file, "episode_length_mean_test_basic")


def calculate_mean_episode_length_train_previous(training_file) -> float:
    return _mean_metric(
        training_file,
        "episode_length_mean_train_previous",
        previous=True,
        episode_count_column="episodes_train_previous",
    )


def calculate_mean_episode_length_test_previous(training_file) -> float:
    return _mean_metric(
        training_file,
        "episode_length_mean_test_previous",
        previous=True,
        episode_count_column="episodes_test_previous",
    )


def calculate_mean_mean_reward_train_basic(training_file) -> float:
    return _mean_metric(training_file, "reward_mean_train_basic")


def calculate_mean_mean_reward_test_basic(training_file) -> float:
    return _mean_metric(training_file, "reward_mean_test_basic")


def calculate_mean_mean_reward_train_previous(training_file) -> float:
    return _mean_metric(
        training_file,
        "reward_mean_train_previous",
        previous=True,
        episode_count_column="episodes_train_previous",
    )


def calculate_mean_mean_reward_test_previous(training_file) -> float:
    return _mean_metric(
        training_file,
        "reward_mean_test_previous",
        previous=True,
        episode_count_column="episodes_test_previous",
    )


def calculate_auc_reward_test_basic(training_file) -> float:
    """
    Liczy znormalizowane AUC średniej łącznej nagrody za epizod
    dla środowiska testowego przeciwko modelowi bazowemu.

    Metryka:
        reward_mean_test_basic

    Wynik:
        AUC / (max_training_steps - min_training_steps)

    Czyli średnia wartość krzywej nagrody w czasie treningu.
    """
    training_results = _load_training_results(training_file)

    required_columns = ["training_steps", "reward_mean_test_basic"]

    for column in required_columns:
        if column not in training_results.columns:
            raise ValueError(f"Brak wymaganej kolumny: {column}")

    df = training_results.sort_values("training_steps")

    x = df["training_steps"].to_numpy(dtype=float)
    y = df["reward_mean_test_basic"].to_numpy(dtype=float)

    if len(x) < 2:
        return 0.0

    x_range = x.max() - x.min()

    if x_range == 0:
        return 0.0

    auc = np.trapezoid(y, x)

    return float(auc / x_range)
