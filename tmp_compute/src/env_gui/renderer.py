from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import imageio.v2 as imageio
import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame

from src.agents.agents import Agent
from src.env.objects import Trap, Turret, TurretProjectile, Wall
from src.utils.action_helpers import action_to_direction, direction_to_delta, is_melee_action
from src.utils.enums import Actions, AgentType, Direction
from src.utils.settings import GlobalSettings


if TYPE_CHECKING:
    from src.env.game_env import GameEnv


Color = tuple[int, int, int]


class EnvRenderer:
    """
    Offline renderer for the current GameEnv state.

    Example:
        renderer = EnvRenderer(settings)
        frame = renderer.render_frame(env)

        render_duel(
            agent1=("DQN", dqn_agent),
            agent2=("DecisionTree", decision_tree_agent),
            seed=123,
            settings=settings,
            output_path="outputs/videos/dqn_vs_tree.mp4",
        )
    """

    def __init__(
        self,
        settings: GlobalSettings,
        tile_size: int = 40,
        hud_height: int = 148,
        margin: int = 16,
        min_board_width: int = 15,
        min_board_height: int = 15,
        player_names: tuple[str, str] | None = None,
    ):
        self.settings = settings
        self.tile_size = tile_size
        self.hud_height = hud_height
        self.margin = margin
        self.min_board_width = min_board_width
        self.min_board_height = min_board_height
        self.player_names = player_names or ("Player 1", "Player 2")

        self.render_board_width = max(settings.width, min_board_width)
        self.render_board_height = max(settings.height, min_board_height)

        self.width = self.render_board_width * tile_size + 2 * margin
        self.height = hud_height + self.render_board_height * tile_size + margin
        self.board_x = margin + (self.render_board_width - settings.width) * tile_size // 2
        self.board_y = hud_height

        pygame.font.init()
        self.font = pygame.font.Font(None, 22)
        self.small_font = pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 26)
        self.sword_images = self._load_sword_images()

    def render_frame(self, env: GameEnv) -> np.ndarray:
        surface = pygame.Surface((self.width, self.height))
        surface.fill((0, 0, 0))

        self._draw_hud(surface, env)
        self._draw_board(surface, env)
        self._draw_projectiles(surface, env)
        self._draw_players(surface, env)
        self._draw_melee_attacks(surface, env)

        frame = pygame.surfarray.array3d(surface)
        return np.transpose(frame, (1, 0, 2)).astype(np.uint8)

    def _draw_hud(self, surface: pygame.Surface, env: GameEnv) -> None:
        pygame.draw.rect(surface, (12, 12, 12), (0, 0, self.width, self.hud_height))
        pygame.draw.line(
            surface,
            (55, 55, 55),
            (0, self.hud_height - 1),
            (self.width, self.hud_height - 1),
        )

        left_x = self.margin
        right_x = max(self.margin, self.width - self.margin - 220)

        self._draw_player_hud(surface, getattr(env, "player1", None), self.player_names[0], left_x, 12, 1, env)
        self._draw_player_hud(surface, getattr(env, "player2", None), self.player_names[1], right_x, 12, 2, env)
        self._draw_center_hud(surface, env)

    def _draw_player_hud(
        self,
        surface: pygame.Surface,
        player: Any,
        name: str,
        x: int,
        y: int,
        player_id: int,
        env: GameEnv,
    ) -> None:
        color = self._player_color(player_id)
        self._draw_text(surface, name, x, y, color, self.title_font)

        bar_y = y + 28
        self._draw_resource_bar(
            surface,
            "HP",
            self._num(player, "hp"),
            self._setting("max_hp", 100.0),
            x,
            bar_y,
            self._setting("hp_bar_color", (220, 40, 40)),
        )
        self._draw_resource_bar(
            surface,
            "STR",
            self._num(player, "strength"),
            self._setting("max_strength", 100.0),
            x,
            bar_y + 20,
            self._setting("strength_bar_color", (230, 210, 40)),
        )
        self._draw_resource_bar(
            surface,
            "MANA",
            self._num(player, "mana"),
            self._setting("max_mana", 100.0),
            x,
            bar_y + 40,
            self._setting("mana_bar_color", (40, 110, 230)),
        )
        self._draw_resource_bar(
            surface,
            "STA",
            self._num(player, "stamina"),
            self._setting("max_stamina", 100.0),
            x,
            bar_y + 60,
            self._setting("stamina_bar_color", (40, 190, 80)),
        )

        statuses = self._status_labels(player)
        status_text = " ".join(statuses) if statuses else "no status"
        status_color = (230, 230, 230) if statuses else (120, 120, 120)
        self._draw_text(surface, status_text, x, bar_y + 80, status_color, self.small_font)

        score = self._player_score(player, env)
        self._draw_text(surface, f"score: {score:.2f}", x, bar_y + 96, (210, 210, 210), self.small_font)

    def _draw_center_hud(self, surface: pygame.Surface, env: GameEnv) -> None:
        center_x = self.width // 2
        current_step = int(getattr(env, "current_step", 0) or 0)
        max_steps = int(getattr(self.settings, "episode_max_steps", 0) or 0)
        remaining_steps = max(0, max_steps - current_step) if max_steps else 0
        fps = max(1, int(self.settings.tm.fps()))
        remaining_seconds = remaining_steps / fps

        lines = [
            f"time: {self._format_mm_ss(remaining_seconds)}",
            f"step: {current_step}",
            f"winner: {self._winner_label(env)}",
        ]

        if bool(getattr(env, "is_over", False)):
            lines.append("finished")

        for idx, line in enumerate(lines):
            text = self.font.render(line, True, (230, 230, 230))
            rect = text.get_rect(center=(center_x, 24 + idx * 22))
            surface.blit(text, rect)

    def _draw_board(self, surface: pygame.Surface, env: GameEnv) -> None:
        for row in range(self._grid_height(env)):
            for col in range(self._grid_width(env)):
                rect = self._tile_rect(row, col)
                obj = self._grid_get(getattr(env, "terrain_grid", []), row, col)
                color = self._terrain_color(obj)
                pygame.draw.rect(surface, color, rect)
                pygame.draw.rect(surface, (28, 28, 28), rect, 1)

                if self._is_turret(obj):
                    self._draw_turret(surface, rect, obj)

    def _draw_players(self, surface: pygame.Surface, env: GameEnv) -> None:
        self._draw_player(surface, getattr(env, "player1", None), 1)
        self._draw_player(surface, getattr(env, "player2", None), 2)

    def _draw_player(self, surface: pygame.Surface, player: Any, player_id: int) -> None:
        position = self._position(player)
        if position is None:
            return

        row, col = position
        center = self._tile_center(row, col)
        radius = max(6, int(self.tile_size * 0.34))
        color = self._player_color(player_id)

        pygame.draw.circle(surface, color, center, radius)
        pygame.draw.circle(surface, (245, 245, 245), center, radius, 2)

        if bool(getattr(player, "absolute_defence_active", False)):
            pygame.draw.circle(surface, (235, 235, 255), center, radius + 5, 3)
        if bool(getattr(player, "attack_boost_active", False)):
            pygame.draw.circle(surface, (255, 210, 60), center, radius + 9, 2)
        if bool(getattr(player, "slowed_active", False)):
            pygame.draw.circle(surface, (130, 210, 255), center, radius + 13, 2)

        label = self.small_font.render(str(player_id), True, (255, 255, 255))
        label_rect = label.get_rect(center=center)
        surface.blit(label, label_rect)

    def _draw_projectiles(self, surface: pygame.Surface, env: GameEnv) -> None:
        projectile_grid = getattr(env, "projectile_grid", [])

        for row in range(self._grid_height(env)):
            for col in range(self._grid_width(env)):
                projectile = self._grid_get(projectile_grid, row, col)
                if projectile is None:
                    continue

                direction = self._direction(projectile)
                points = self._projectile_points(row, col, direction)
                color = self._projectile_color(projectile)
                pygame.draw.polygon(surface, color, points)
                pygame.draw.polygon(surface, (20, 20, 20), points, 2)

    def _draw_resource_bar(
        self,
        surface: pygame.Surface,
        label: str,
        value: float,
        max_value: float,
        x: int,
        y: int,
        color: Color,
        width: int = 150,
        height: int = 12,
    ) -> None:
        ratio = 0.0 if max_value <= 0 else max(0.0, min(1.0, value / max_value))
        pygame.draw.rect(surface, (45, 45, 45), (x + 48, y, width, height))
        pygame.draw.rect(surface, color, (x + 48, y, int(width * ratio), height))
        pygame.draw.rect(surface, (160, 160, 160), (x + 48, y, width, height), 1)
        self._draw_text(surface, label, x, y - 2, (220, 220, 220), self.small_font)

        value_text = f"{value:.0f}/{max_value:.0f}"
        self._draw_text(surface, value_text, x + 52 + width, y - 2, (210, 210, 210), self.small_font)

    def _draw_melee_attacks(self, surface: pygame.Surface, env: GameEnv) -> None:
        self._draw_player_melee_attack(surface, getattr(env, "player1", None), 1)
        self._draw_player_melee_attack(surface, getattr(env, "player2", None), 2)

    def _draw_player_melee_attack(
        self,
        surface: pygame.Surface,
        player: Any,
        player_id: int,
    ) -> None:
        if player is None or bool(getattr(player, "invalid_action", False)):
            return

        action = self._coerce_action(getattr(player, "pending_action", Actions.NONE))
        if not is_melee_action(action):
            return

        position = self._position(player)
        direction = action_to_direction(action)
        if position is None or direction is None:
            return

        row, col = position
        d_row, d_col = direction_to_delta(direction)
        target_row = row + d_row
        target_col = col + d_col

        if not self._inside(target_row, target_col):
            return

        start = self._tile_center(row, col)
        end = self._tile_center(target_row, target_col)
        center = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        image = self.sword_images.get((player_id, direction))

        if image is None:
            self._draw_fallback_sword(surface, center, direction, player_id)
            return

        size = max(10, int(self.tile_size * 0.9))
        sword = pygame.transform.smoothscale(image, (size, size))
        rect = sword.get_rect(center=center)
        surface.blit(sword, rect)

    def _draw_fallback_sword(
        self,
        surface: pygame.Surface,
        center: tuple[int, int],
        direction: Direction,
        player_id: int,
    ) -> None:
        color = self._player_color(player_id)
        length = max(10, int(self.tile_size * 0.62))
        if direction == Direction.UP:
            points = [(center[0], center[1] - length), (center[0] - 5, center[1] + 6), (center[0] + 5, center[1] + 6)]
        elif direction == Direction.DOWN:
            points = [(center[0], center[1] + length), (center[0] - 5, center[1] - 6), (center[0] + 5, center[1] - 6)]
        elif direction == Direction.LEFT:
            points = [(center[0] - length, center[1]), (center[0] + 6, center[1] - 5), (center[0] + 6, center[1] + 5)]
        else:
            points = [(center[0] + length, center[1]), (center[0] - 6, center[1] - 5), (center[0] - 6, center[1] + 5)]
        pygame.draw.polygon(surface, color, points)
        pygame.draw.polygon(surface, (245, 245, 245), points, 1)

    def _draw_turret(self, surface: pygame.Surface, rect: pygame.Rect, obj: Any) -> None:
        center = rect.center
        radius = max(5, int(self.tile_size * 0.24))
        pygame.draw.circle(surface, (95, 75, 20), center, radius + 4)
        pygame.draw.circle(surface, (255, 225, 90), center, radius)

        cooldown = self._num(obj, "cooldown", None)
        interval = self._setting("turret_shoot_interval", 0)
        if cooldown is None or interval <= 0:
            return

        charge = 1.0 - max(0.0, min(1.0, cooldown / interval))
        charge_width = int((rect.width - 8) * charge)
        pygame.draw.rect(surface, (45, 45, 45), (rect.x + 4, rect.bottom - 7, rect.width - 8, 4))
        pygame.draw.rect(surface, (255, 235, 80), (rect.x + 4, rect.bottom - 7, charge_width, 4))

    def _load_sword_images(self) -> dict[tuple[int, Direction], pygame.Surface]:
        images: dict[tuple[int, Direction], pygame.Surface] = {}
        base_path = Path(__file__).resolve().parent / "images"
        names = {
            Direction.UP: "up",
            Direction.DOWN: "down",
            Direction.LEFT: "left",
            Direction.RIGHT: "right",
        }

        for player_id in (1, 2):
            for direction, name in names.items():
                path = base_path / f"sword-player{player_id}-{name}.png"
                if not path.exists():
                    continue
                try:
                    images[(player_id, direction)] = pygame.image.load(str(path))
                except pygame.error:
                    continue

        return images

    def _projectile_points(
        self,
        row: int,
        col: int,
        direction: Direction | None,
    ) -> list[tuple[int, int]]:
        center_x, center_y = self._tile_center(row, col)
        size = max(7, int(self.tile_size * 0.32))
        side = max(5, int(self.tile_size * 0.20))

        if direction == Direction.UP:
            return [(center_x, center_y - size), (center_x - side, center_y + side), (center_x + side, center_y + side)]
        if direction == Direction.DOWN:
            return [(center_x, center_y + size), (center_x - side, center_y - side), (center_x + side, center_y - side)]
        if direction == Direction.LEFT:
            return [(center_x - size, center_y), (center_x + side, center_y - side), (center_x + side, center_y + side)]
        return [(center_x + size, center_y), (center_x - side, center_y - side), (center_x - side, center_y + side)]

    def _projectile_color(self, projectile: Any) -> Color:
        owner_id = getattr(projectile, "owner_id", None)
        if owner_id == 1:
            return self._player_color(1)
        if owner_id == 2:
            return self._player_color(2)
        if isinstance(projectile, TurretProjectile) or "turret" in projectile.__class__.__name__.lower():
            return self._setting("turret_tile_color", (245, 180, 40))
        return (235, 235, 235)

    def _terrain_color(self, obj: Any) -> Color:
        if self._is_turret(obj):
            return self._setting("turret_tile_color", (245, 180, 40))
        if self._is_wall(obj):
            return self._setting("wall_tile_color", (55, 55, 55))
        if self._is_trap(obj):
            return self._setting("trap_tile_color", (190, 40, 40))
        return self._setting("empty_tile_color", (120, 200, 120))

    def _status_labels(self, player: Any) -> list[str]:
        if player is None:
            return []

        labels = []
        if bool(getattr(player, "slowed_active", False)):
            labels.append("[SLOWED]")
        if bool(getattr(player, "attack_boost_active", False)):
            labels.append("[ATK BOOST]")
        if bool(getattr(player, "absolute_defence_active", False)):
            labels.append("[ABS_DEF]")
        return labels

    @staticmethod
    def _format_mm_ss(seconds: float) -> str:
        total_seconds = max(0, int(seconds))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _winner_label(env: GameEnv) -> str:
        winner_id = getattr(env, "winner_id", 0) or 0
        try:
            winner_id = int(winner_id)
        except (TypeError, ValueError):
            return "none"

        if winner_id in (1, 2):
            return f"P{winner_id}"
        return "none"

    def _player_score(self, player: Any, env: GameEnv) -> float:
        if player is None:
            return 0.0

        current_step = int(getattr(env, "current_step", 0) or 0)
        step_penalty = self._setting("step_penalty", 0.0)
        return self._num(player, "episode_reward") + current_step * float(step_penalty)

    def _draw_text(
        self,
        surface: pygame.Surface,
        text: str,
        x: int,
        y: int,
        color: Color,
        font: pygame.font.Font,
    ) -> None:
        rendered = font.render(text, True, color)
        surface.blit(rendered, (x, y))

    def _tile_rect(self, row: int, col: int) -> pygame.Rect:
        return pygame.Rect(
            self.board_x + col * self.tile_size,
            self.board_y + row * self.tile_size,
            self.tile_size,
            self.tile_size,
        )

    def _tile_center(self, row: int, col: int) -> tuple[int, int]:
        return (
            self.board_x + col * self.tile_size + self.tile_size // 2,
            self.board_y + row * self.tile_size + self.tile_size // 2,
        )

    def _grid_width(self, env: GameEnv) -> int:
        return int(getattr(getattr(env, "settings", self.settings), "width", self.settings.width))

    def _grid_height(self, env: GameEnv) -> int:
        return int(getattr(getattr(env, "settings", self.settings), "height", self.settings.height))

    def _inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.settings.height and 0 <= col < self.settings.width

    def _position(self, obj: Any) -> tuple[int, int] | None:
        if obj is None:
            return None

        position = getattr(obj, "position", None)
        row = getattr(position, "row", None)
        col = getattr(position, "col", None)

        if row is None:
            row = getattr(obj, "row", None)
        if col is None:
            col = getattr(obj, "col", None)

        if row is None or col is None:
            return None

        return int(row), int(col)

    def _direction(self, obj: Any) -> Direction | None:
        direction = getattr(obj, "direction", None)
        if isinstance(direction, Direction):
            return direction

        if isinstance(direction, str):
            normalized = direction.upper()
            return Direction.__members__.get(normalized)

        if isinstance(direction, int):
            try:
                return Direction(direction)
            except ValueError:
                return None

        value = getattr(direction, "value", None)
        if isinstance(value, int):
            try:
                return Direction(value)
            except ValueError:
                return None

        return None

    def _coerce_action(self, action: Any) -> Actions:
        if isinstance(action, Actions):
            return action

        item = getattr(action, "item", None)
        if callable(item):
            return self._coerce_action(item())

        if isinstance(action, str):
            if action in Actions.__members__:
                return Actions[action]
            try:
                return Actions(action)
            except ValueError:
                return Actions.NONE

        try:
            return Actions(action)
        except (TypeError, ValueError):
            return Actions.NONE

    def _is_wall(self, obj: Any) -> bool:
        return isinstance(obj, Wall) or obj.__class__.__name__.lower().endswith("wall")

    def _is_trap(self, obj: Any) -> bool:
        return isinstance(obj, Trap) or "trap" in obj.__class__.__name__.lower()

    def _is_turret(self, obj: Any) -> bool:
        return isinstance(obj, Turret) or "turret" in obj.__class__.__name__.lower()

    def _player_color(self, player_id: int) -> Color:
        if player_id == 1:
            return self._setting("player_1_color", (75, 119, 209))
        return self._setting("player_2_color", (234, 51, 35))

    def _setting(self, name: str, default: Any) -> Any:
        return getattr(self.settings, name, default)

    @staticmethod
    def _num(obj: Any, name: str, default: float = 0.0) -> float:
        value = getattr(obj, name, default)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _grid_get(grid: Any, row: int, col: int) -> Any:
        try:
            return grid[row][col]
        except (IndexError, KeyError, TypeError):
            return None


def render_frame(env: GameEnv, settings: GlobalSettings | None = None) -> np.ndarray:
    renderer = EnvRenderer(settings or env.settings)
    return renderer.render_frame(env)


def render_duel(
    agent1: tuple[str, Agent],
    agent2: tuple[str, Agent],
    seed: int | str,
    settings: GlobalSettings,
    output_path: str | None = None,
    *,
    filename: str | None = None,
    reverse_positions: bool = False,
    include_initial_frame: bool = True,
    max_steps: int | None = None,
    return_success: bool = False,
) -> str | bool:
    output = output_path or filename or "example.mp4"

    try:
        env = _prepare_env_for_agent2(
            agent=agent2[1],
            settings=settings,
            seed=seed,
            reverse_positions=reverse_positions,
        )

        obs = env.reset(
            seed=seed,
            use_default=(_agent_type(agent2[1]) == AgentType.BASIC),
            reverse_positions=reverse_positions,
        )

        renderer = EnvRenderer(settings=settings, player_names=(agent1[0], agent2[0]))
        frames: list[np.ndarray] = []

        if include_initial_frame:
            frames.append(renderer.render_frame(env))

        done = bool(getattr(env, "is_over", False))
        step_limit = max_steps if max_steps is not None else int(settings.episode_max_steps)
        steps = 0

        while not done and steps < step_limit:
            action = _select_agent1_action(env=env, agent=agent1[1], obs=obs)
            obs, _reward, done, _won = env.step_rl(action)
            frames.append(renderer.render_frame(env))
            steps += 1

        if not frames:
            frames.append(renderer.render_frame(env))

        output_file = Path(output)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fps = max(1, int(settings.tm.fps()))
        frames.extend([frames[-1]] * (fps * 3))
        imageio.mimsave(str(output_file), frames, fps=fps)

    except Exception:
        if return_success:
            return False
        raise

    if return_success:
        return True

    return str(output)


def _prepare_env_for_agent2(
    agent: Agent,
    settings: GlobalSettings,
    seed: int | str,
    reverse_positions: bool,
) -> GameEnv:
    agent_type = _agent_type(agent)
    game_env_cls = _get_game_env_class()

    if agent_type == AgentType.BASIC:
        return game_env_cls(
            settings=settings,
            seed=seed,
            default_agent=agent,
            custom_agent=None,
            use_default=True,
            reverse_positions=reverse_positions,
        )

    if agent_type == AgentType.RL:
        return game_env_cls(
            settings=settings,
            seed=seed,
            default_agent=None,
            custom_agent=agent,
            use_default=False,
            reverse_positions=reverse_positions,
        )

    raise ValueError(f"Nieznany typ agenta: {agent_type}")


def _get_game_env_class():
    try:
        from src.env.game_env import GameEnv as game_env_cls
    except ImportError as exc:
        raise ImportError(
            "Nie mozna zaimportowac GameEnv. W projekcie jest prawdopodobny "
            "cykliczny import miedzy src/env/game_env.py i src/agents/base_agent.py "
            "(BASIC_OBS_COLUMNS / DecisionTreeAgent). Renderer klatek mozna nadal "
            "uzyc na istniejacym obiekcie env, ale render_duel wymaga dzialajacego "
            "importu GameEnv."
        ) from exc

    return game_env_cls


def _select_agent1_action(env: GameEnv, agent: Agent, obs: Any) -> Actions:
    agent_type = _agent_type(agent)

    if agent_type == AgentType.BASIC:
        action = agent.predict(env.get_basic_obs(player_id=1))
    elif agent_type == AgentType.RL:
        action = agent.predict(obs)
    else:
        raise ValueError(f"Nieznany typ agenta: {agent_type}")

    return _coerce_action(action)


def _agent_type(agent: Agent) -> AgentType:
    agent_type = getattr(agent, "agent_type", AgentType.BASIC)

    if isinstance(agent_type, AgentType):
        return agent_type

    if isinstance(agent_type, str):
        normalized = agent_type.upper()
        if normalized in AgentType.__members__:
            return AgentType[normalized]

    try:
        return AgentType(agent_type)
    except (TypeError, ValueError):
        raise ValueError(f"Nieznany typ agenta: {agent_type}") from None


def _coerce_action(action: Any) -> Actions:
    if isinstance(action, Actions):
        return action

    item = getattr(action, "item", None)
    if callable(item):
        return _coerce_action(item())

    if isinstance(action, str):
        if action in Actions.__members__:
            return Actions[action]
        try:
            return Actions(action)
        except ValueError:
            return Actions.NONE

    try:
        return Actions(action)
    except (TypeError, ValueError):
        return Actions.NONE
