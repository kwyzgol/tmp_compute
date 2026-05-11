import statistics
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


def training_compare_table(
        groups: list[tuple[str, list[float]]],
        subtitle: str | None = None) -> None:

    group_names = [name for name, _ in groups]
    values_lists = [values for _, values in groups]

    max_runs = max(len(values) for values in values_lists)

    header_color = '#FFD966'

    headers = (
        ['<b>Grupa</b>']
        + [f'<b>Próba {i + 1}</b>' for i in range(max_runs)]
        + ['<b>Średnia</b>', '<b>Mediana</b>', '<b>Odch. std.</b>']
    )

    table_columns = []

    # kolumna nazw grup
    table_columns.append([f'<b>{name}</b>' for name in group_names])

    # kolumny prób
    for run_idx in range(max_runs):
        column = []

        for values in values_lists:
            if run_idx < len(values):
                column.append(f'{values[run_idx]:.7g}')
            else:
                column.append('—')

        table_columns.append(column)

    # statystyki końcowe
    means = []
    medians = []
    stds = []

    for values in values_lists:
        means.append(f'{statistics.mean(values):.7g}')
        medians.append(f'{statistics.median(values):.7g}')

        if len(values) >= 2:
            stds.append(f'{statistics.stdev(values):.7g}')
        else:
            stds.append('—')

    table_columns.append(means)
    table_columns.append(medians)
    table_columns.append(stds)

    # kolory komórek
    fill_colors = []

    # kolumna grup
    fill_colors.append([header_color for _ in group_names])

    # reszta kolumn
    for _ in range(max_runs + 3):
        fill_colors.append(['white' for _ in group_names])

    title = '<b>[Porównanie wyników treningu]</b>'

    if subtitle is not None:
        title += f'<br>{subtitle}'

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=headers,
                    fill_color=header_color,
                    font=dict(color='black'),
                    align='center',
                    line_color='black',
                    line_width=2,
                ),
                cells=dict(
                    values=table_columns,
                    fill_color=fill_colors,
                    font=dict(color='black'),
                    align='center',
                    height=35,
                    line_color='black',
                    line_width=2,
                )
            )
        ]
    )

    group_count = len(group_names)

    if group_count == 1:
        multiplayer = 150
    elif group_count == 2:
        multiplayer = 90
    elif group_count == 3:
        multiplayer = 70
    elif group_count == 4 or group_count == 5:
        multiplayer = 60
    elif group_count == 6:
        multiplayer = 55
    else:
        multiplayer = 50

    fig.update_layout(
        title=title,
        title_x=0.5,
        height=120 + len(group_names) * multiplayer
    )

    fig.show()


# =====================================================================================================


def _show_heatmap(df, value_column: str, title: str, text_format: str = ".2f"):
    heatmap_df = df.pivot(
        index="player1_name",
        columns="player2_name",
        values=value_column,
    )

    fig = px.imshow(
        heatmap_df,
        text_auto=text_format,
        aspect="auto",
        color_continuous_scale="Viridis",
    )

    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.5,
            "y": 0.95,
            "xanchor": "center",
            "font": {
                "size": 24,
            }
        },

        width=1000,
        height=700,

        margin=dict(
            l=120,
            r=40,
            t=140,
            b=80,
        ),

        font=dict(
            size=16,
        ),

        xaxis_title="<b>Przeciwnik</b>",
        yaxis_title="<b>Agent</b>",

        plot_bgcolor="white",
    )

    fig.update_xaxes(
        side="top",
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )

    fig.update_yaxes(
        tickfont=dict(size=16),
        title_font=dict(size=18),
    )

    fig.update_traces(
        textfont_size=16,
    )

    fig.show()


def show_heatmap_winrate_train(df):
    _show_heatmap(
        df=df,
        value_column="winrate_train",
        title="Odsetek zwycięstw - środowisko treningowe",
        text_format=".2f",
    )


def show_heatmap_winrate_test(df):
    _show_heatmap(
        df=df,
        value_column="winrate_test",
        title="Odsetek zwycięstw - środowisko testowe",
        text_format=".2f",
    )


def show_heatmap_winrate_diff(df):
    df = df.copy()
    df["winrate_diff"] = df["winrate_test"] - df["winrate_train"]

    _show_heatmap(
        df=df,
        value_column="winrate_diff",
        title="Różnica odsetka zwycięstw (testowe - treningowe)",
        text_format=".2f",
    )


def show_heatmap_episode_train(df):
    _show_heatmap(
        df=df,
        value_column="episode_length_mean_train",
        title="Średnia długość epizodu - środowisko treningowe",
        text_format=".1f",
    )


def show_heatmap_episode_test(df):
    _show_heatmap(
        df=df,
        value_column="episode_length_mean_test",
        title="Średnia długość epizodu - środowisko testowe",
        text_format=".1f",
    )


# =====================================================================================================
# WIZUALIZACJA PRZEBIEGU TRENINGU
# =====================================================================================================

TRAIN_LINE_COLOR = "forestgreen"
TEST_LINE_COLOR = "darkorange"
LINE_WIDTH = 4
MARKER_SIZE = 9

ADVANCED_TRAINING_LINE_COLOR = "red"
ADVANCED_TRAINING_LINE_DASH = "dash"
ADVANCED_TRAINING_LINE_WIDTH = 3


def _get_advanced_training_step(df):
    advanced_rows = df[df["advanced_training"] == True]

    if len(advanced_rows) == 0:
        return None

    return advanced_rows.iloc[0]["training_steps"]


def _add_advanced_training_line(fig, x_position):
    if x_position is None:
        return

    fig.add_vline(
        x=x_position,
        line_width=ADVANCED_TRAINING_LINE_WIDTH,
        line_dash=ADVANCED_TRAINING_LINE_DASH,
        line_color=ADVANCED_TRAINING_LINE_COLOR,
    )


def _style_figure(fig, title):
    annotations = [
        dict(
            text="<b>[- - -] Czerwona linia przerywana</b> - włączenie zaawansowanego treningu",
            x=0.0,
            y=1.13,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            xanchor="left",
            font=dict(size=12, color=ADVANCED_TRAINING_LINE_COLOR),
        )
    ]

    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 22},
        },
        template="plotly_white",
        height=650,
        width=1400,
        margin=dict(t=200, l=80, r=60, b=70),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(size=12),
        ),
        annotations=annotations,
    )


def show_training_summary(global_fig_title, training_results):
    """
    Wyświetla podsumowanie przebiegu treningu.

    Funkcja tworzy 3 przestrzenie robocze Plotly:
        - odsetek zwycięstw,
        - średnia wartość łącznej nagrody,
        - średnia długość epizodu.

    Każda przestrzeń zawiera 2 subploty:
        - środowisko treningowe,
        - środowisko testowe.

    Dodatkowo:
        - jeśli advanced training został aktywowany,
          rysowana jest pionowa przerywana czerwona linia
          w miejscu pierwszej aktywacji.
    """

    df = training_results.copy()

    advanced_training_step = _get_advanced_training_step(df)

    x = df["training_steps"]

    # ==================================================
    # ODSETEK ZWYCIĘSTW
    # ==================================================

    fig_winrate = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Odsetek zwycięstw - środowisko treningowe",
            "Odsetek zwycięstw - środowisko testowe",
        ),
        horizontal_spacing=0.12,
    )

    fig_winrate.add_trace(
        go.Scatter(
            x=x,
            y=df["winrate_train"],
            mode="lines+markers",
            line=dict(width=LINE_WIDTH, color=TRAIN_LINE_COLOR),
            marker=dict(size=MARKER_SIZE),
            name="Środowisko treningowe",
        ),
        row=1,
        col=1,
    )

    fig_winrate.add_trace(
        go.Scatter(
            x=x,
            y=df["winrate_test"],
            mode="lines+markers",
            line=dict(width=LINE_WIDTH, color=TEST_LINE_COLOR),
            marker=dict(size=MARKER_SIZE),
            name="Środowisko testowe",
        ),
        row=1,
        col=2,
    )

    _add_advanced_training_line(fig_winrate, advanced_training_step)

    fig_winrate.update_yaxes(title_text="<b>Odsetek zwycięstw</b>", range=[0, 1], row=1, col=1)
    fig_winrate.update_yaxes(title_text="<b>Odsetek zwycięstw</b>", range=[0, 1], row=1, col=2)

    fig_winrate.update_xaxes(title_text="<b>Kroki treningu</b>", row=1, col=1)
    fig_winrate.update_xaxes(title_text="<b>Kroki treningu</b>", row=1, col=2)

    _style_figure(
        fig_winrate,
        f"<b>{global_fig_title}</b><br>Odsetek zwycięstw",
    )

    # ==================================================
    # ŚREDNIA WARTOŚĆ ŁĄCZNEJ NAGRODY
    # ==================================================

    fig_reward = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Średnia wartość łącznej nagrody - środowisko treningowe",
            "Średnia wartość łącznej nagrody - środowisko testowe",
        ),
        horizontal_spacing=0.12,
    )

    fig_reward.add_trace(
        go.Scatter(
            x=x,
            y=df["reward_mean_train"],
            mode="lines+markers",
            line=dict(width=LINE_WIDTH, color=TRAIN_LINE_COLOR),
            marker=dict(size=MARKER_SIZE),
            name="Środowisko treningowe",
        ),
        row=1,
        col=1,
    )

    fig_reward.add_trace(
        go.Scatter(
            x=x,
            y=df["reward_mean_test"],
            mode="lines+markers",
            line=dict(width=LINE_WIDTH, color=TEST_LINE_COLOR),
            marker=dict(size=MARKER_SIZE),
            name="Środowisko testowe",
        ),
        row=1,
        col=2,
    )

    _add_advanced_training_line(fig_reward, advanced_training_step)

    fig_reward.update_yaxes(title_text="<b>Średnia wartość łącznej nagrody</b>", row=1, col=1)
    fig_reward.update_yaxes(title_text="<b>Średnia wartość łącznej nagrody</b>", row=1, col=2)

    fig_reward.update_xaxes(title_text="<b>Kroki treningu</b>", row=1, col=1)
    fig_reward.update_xaxes(title_text="<b>Kroki treningu</b>", row=1, col=2)

    _style_figure(
        fig_reward,
        f"<b>{global_fig_title}</b><br>Średnia wartość łącznej nagrody",
    )

    # ==================================================
    # ŚREDNIA DŁUGOŚĆ EPIZODU
    # ==================================================

    fig_episode_length = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Średnia długość epizodu - środowisko treningowe",
            "Średnia długość epizodu - środowisko testowe",
        ),
        horizontal_spacing=0.12,
    )

    fig_episode_length.add_trace(
        go.Scatter(
            x=x,
            y=df["episode_length_mean_train"],
            mode="lines+markers",
            line=dict(width=LINE_WIDTH, color=TRAIN_LINE_COLOR),
            marker=dict(size=MARKER_SIZE),
            name="Środowisko treningowe",
        ),
        row=1,
        col=1,
    )

    fig_episode_length.add_trace(
        go.Scatter(
            x=x,
            y=df["episode_length_mean_test"],
            mode="lines+markers",
            line=dict(width=LINE_WIDTH, color=TEST_LINE_COLOR),
            marker=dict(size=MARKER_SIZE),
            name="Środowisko testowe",
        ),
        row=1,
        col=2,
    )

    _add_advanced_training_line(fig_episode_length, advanced_training_step)

    fig_episode_length.update_yaxes(title_text="<b>Średnia długość epizodu</b>", row=1, col=1)
    fig_episode_length.update_yaxes(title_text="<b>Średnia długość epizodu</b>", row=1, col=2)

    fig_episode_length.update_xaxes(title_text="<b>Kroki treningu</b>", row=1, col=1)
    fig_episode_length.update_xaxes(title_text="<b>Kroki treningu</b>", row=1, col=2)

    _style_figure(
        fig_episode_length,
        f"<b>{global_fig_title}</b><br>Średnia długość epizodu",
    )

    fig_winrate.show()
    fig_reward.show()
    fig_episode_length.show()
