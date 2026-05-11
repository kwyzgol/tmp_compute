import random

import numpy as np
import torch

from src.agents.agents import Agent, LazyAgent
from src.env.objects import EmptyField, PlayerProjectile, Trap, Turret, Wall
from src.env.player import Player
from src.env.position import Position
from src.utils.action_helpers import (
    action_to_direction,
    direction_to_delta,
    is_dash_action,
    is_melee_action,
    is_move_action,
    is_ranged_action,
)
from src.utils.enums import Actions
from src.utils.grid_pathfinding import path_exists
from src.utils.settings import GlobalSettings


BASIC_OBS_COLUMNS = [
    "hp_bucket",
    "strength_bucket",
    "mana_bucket",
    "stamina_bucket",

    "absolute_defence_active",
    "attack_boost_active",
    "slowed_active",

    "available_field_up",
    "available_field_down",
    "available_field_left",
    "available_field_right",

    "can_move",
    "can_melee",
    "can_ranged",
    "can_dash",
    "can_absolute_defence",
    "can_attack_boost",

    "opponent_adjacent_up",
    "opponent_adjacent_down",
    "opponent_adjacent_left",
    "opponent_adjacent_right",

    "opponent_inline_up",
    "opponent_inline_down",
    "opponent_inline_left",
    "opponent_inline_right",

    "enemy_projectile_near_up",
    "enemy_projectile_near_down",
    "enemy_projectile_near_left",
    "enemy_projectile_near_right",
]


class GameEnv:
    def __init__(
        self,
        settings: GlobalSettings,
        seed: int | str | None = None,
        default_agent: Agent | None = None,
        custom_agent: Agent | None = None,
        use_default: bool = True,
        reverse_positions: bool = False,
    ):
        self.settings = settings
        self.seed = seed
        self.rng = random.Random(seed)

        if default_agent is None:
            from src.agents.base_agent import DecisionTreeAgent

            default_agent = DecisionTreeAgent(settings)

        self.default_agent = default_agent
        self.custom_agent = custom_agent
        self.use_default = use_default
        self.reverse_positions = reverse_positions

        self.current_step = 0
        self.score_player1 = 0
        self.score_player2 = 0

        self.player1: Player | None = None
        self.player2: Player | None = None

        self.terrain_grid = []
        self.projectile_grid = []

        self.is_over = False
        self.winner_id = 0
        self.last_obs_player1 = None

        self.reset(
            seed=seed,
            use_default=use_default,
            reverse_positions=reverse_positions,
        )

    # ==========================================================================
    # RESET / GENEROWANIE
    # ==========================================================================

    def reset(
        self,
        seed: int | str | None = None,
        use_default: bool | None = None,
        reverse_positions: bool | None = None,
    ):
        if seed is not None:
            self.seed = seed
            self.rng = random.Random(seed)

        if use_default is not None:
            self.use_default = use_default

        if reverse_positions is not None:
            self.reverse_positions = reverse_positions

        if not self.use_default and self.custom_agent is None:
            raise ValueError("custom_agent nie może być None, gdy use_default=False.")

        self.current_step = 0
        self.score_player1 = 0
        self.score_player2 = 0
        self.is_over = False
        self.winner_id = 0

        self._generate_valid_map()

        self.last_obs_player1 = self.get_rl_obs(player_id=1)
        return self.last_obs_player1

    def _generate_valid_map(self) -> None:

        while True:
            self.terrain_grid = self._generate_terrain_grid_once()
            self.projectile_grid = self._generate_empty_projectile_grid()

            pos1, pos2 = self._draw_start_positions()

            if self.reverse_positions:
                pos1, pos2 = pos2, pos1

            self.player1 = Player(player_id=1, position=pos1, settings=self.settings)
            self.player2 = Player(player_id=2, position=pos2, settings=self.settings)

            if self._players_have_path():
                return


    def _generate_terrain_grid_once(self):
        width = self.settings.width
        height = self.settings.height
        cells_count = width * height

        wall_count = int(cells_count * self.settings.walls_ratio)
        turret_count = int(cells_count * self.settings.turret_ratio)
        trap_count = int(cells_count * self.settings.trap_ratio)
        empty_count = cells_count - wall_count - turret_count - trap_count

        if empty_count < 2:
            raise ValueError("Mapa musi mieć przynajmniej 2 wolne pola dla graczy.")

        object_types = (
            [EmptyField for _ in range(empty_count)]
            + [Wall for _ in range(wall_count)]
            + [Turret for _ in range(turret_count)]
            + [Trap for _ in range(trap_count)]
        )

        self.rng.shuffle(object_types)

        grid = []

        for row in range(height):
            grid_row = []

            for col in range(width):
                index = row * width + col
                object_cls = object_types[index]
                position = Position(row, col)
                grid_row.append(object_cls(self, position))

            grid.append(grid_row)

        return grid

    def _generate_empty_projectile_grid(self):
        return [
            [None for _ in range(self.settings.width)]
            for _ in range(self.settings.height)
        ]

    def _draw_start_positions(self) -> tuple[Position, Position]:
        free_positions = []

        for row in range(self.settings.height):
            for col in range(self.settings.width):
                position = Position(row, col)
                if self.is_walkable(position):
                    free_positions.append(position)

        if len(free_positions) < 2:
            raise ValueError("Za mało wolnych pól dla graczy.")

        pos1, pos2 = self.rng.sample(free_positions, 2)
        return pos1, pos2

    def _players_have_path(self) -> bool:
        grid = self.get_lite_grid(player_id=1)
        return path_exists(
            grid=grid,
            start=self.player1.position,
            goal=self.player2.position,
        )

    # ==========================================================================
    # STEP
    # ==========================================================================

    def step(self, action: Actions):
        if self.is_over:
            return self._build_step_result_for_player1()

        self.player1.reset_step_reward()
        self.player2.reset_step_reward()

        self.player1.pending_action = action
        self.player2.pending_action = self._select_player2_action()

        players_order = [self.player1, self.player2]
        self.rng.shuffle(players_order)

        for player in players_order:
            if not player.is_alive:
                continue

            self._update_player_action(player)

            if self._has_dead_player():
                break

        if not self._has_dead_player():
            self._update_terrain_grid()

        if not self._has_dead_player():
            self._update_projectile_grid()

        if not self._has_dead_player():
            self.player1.update_timers_and_regen()
            self.player2.update_timers_and_regen()

        self.current_step += 1

        self.check_is_over()

        if self.is_over:
            self._apply_final_rewards()

        return self._build_step_result_for_player1()

    def step_rl(self, action: Actions) -> tuple[torch.Tensor, float, bool, bool]:
        obs, reward, is_over, won = self.step(action)
        return obs, reward, is_over, won

    def _build_step_result_for_player1(self) -> tuple[torch.Tensor, float, bool, bool]:
        obs = self.get_rl_obs(player_id=1)
        self.last_obs_player1 = obs
        return obs, float(self.player1.next_reward), self.is_over, self.winner_id == 1

    def _select_player2_action(self) -> Actions:
        agent = self.default_agent if self.use_default else self.custom_agent

        if agent is None:
            raise ValueError("Brak agenta dla player2.")

        if self.use_default:
            obs = self.get_basic_obs(player_id=2)
        else:
            obs = self.get_rl_obs(player_id=2)

        return agent.predict(obs)

    def _update_player_action(self, player: Player) -> None:
        action = player.pending_action

        if action == Actions.NONE:
            return

        if is_move_action(action):
            self._try_move(player, action)
            return

        if is_melee_action(action):
            self._try_melee_attack(player, action)
            return

        if is_ranged_action(action):
            self._try_ranged_attack(player, action)
            return

        if is_dash_action(action):
            self._try_dash(player, action)
            return

        if action == Actions.ABSOLUTE_DEFENSE_ABILITY:
            self._try_absolute_defence(player)
            return

        if action == Actions.ATTACK_BOOST_ABILITY:
            self._try_attack_boost(player)
            return

        self._mark_invalid_action(player)

    def _try_move(self, player: Player, action: Actions) -> None:
        if player.move_cooldown > 0:
            self._mark_invalid_action(player)
            return

        direction = action_to_direction(action)
        d_row, d_col = direction_to_delta(direction)
        target = player.position.moved(d_row, d_col)

        if not self.is_position_available_for_player(target):
            self._mark_invalid_action(player)
            return

        player.position = target
        self._resolve_projectile_collision_at_player(player)
        player.move_cooldown = self._apply_slow_multiplier(self.settings.move_cooldown, player)

    def _try_melee_attack(self, player: Player, action: Actions) -> None:
        if player.melee_cooldown > 0:
            self._mark_invalid_action(player)
            return

        if player.strength < self.settings.melee_strength_cost:
            self._mark_invalid_action(player)
            return

        direction = action_to_direction(action)
        d_row, d_col = direction_to_delta(direction)
        target = player.position.moved(d_row, d_col)

        if not self.is_inside(target):
            self._mark_invalid_action(player)
            return

        player.strength -= self.settings.melee_strength_cost
        player.melee_cooldown = self._apply_slow_multiplier(self.settings.melee_cooldown, player)

        target_player = self.get_player_at(target)

        if target_player is None or target_player.player_id == player.player_id:
            return

        self.damage_player(
            player=target_player,
            damage=self.settings.melee_damage,
            attacker=player,
            apply_slow=self.settings.melee_applies_slow,
        )

    def _try_ranged_attack(self, player: Player, action: Actions) -> None:
        if player.ranged_cooldown > 0:
            self._mark_invalid_action(player)
            return

        if player.mana < self.settings.ranged_mana_cost:
            self._mark_invalid_action(player)
            return

        direction = action_to_direction(action)
        d_row, d_col = direction_to_delta(direction)
        target = player.position.moved(d_row, d_col)

        if not self.is_inside(target):
            self._mark_invalid_action(player)
            return

        if self.terrain_grid[target.row][target.col].blocks_projectiles():
            self._mark_invalid_action(player)
            return

        if self.projectile_grid[target.row][target.col] is not None:
            self._mark_invalid_action(player)
            return

        player.mana -= self.settings.ranged_mana_cost
        player.ranged_cooldown = self._apply_slow_multiplier(self.settings.ranged_cooldown, player)

        target_player = self.get_player_at(target)
        if target_player is not None and target_player.player_id != player.player_id:
            self.damage_player(
                player=target_player,
                damage=self.settings.projectile_damage,
                attacker=player,
                apply_slow=self.settings.projectile_applies_slow,
            )
            return

        self.projectile_grid[target.row][target.col] = PlayerProjectile(
            env=self,
            position=target,
            direction=direction,
            owner_id=player.player_id,
        )

    def _try_dash(self, player: Player, action: Actions) -> None:
        if player.dash_cooldown > 0:
            self._mark_invalid_action(player)
            return

        if player.stamina < self.settings.dash_stamina_cost:
            self._mark_invalid_action(player)
            return

        direction = action_to_direction(action)
        d_row, d_col = direction_to_delta(direction)

        first_target = player.position.moved(d_row, d_col)
        second_target = first_target.moved(d_row, d_col)

        if not self.is_position_available_for_player(first_target):
            self._mark_invalid_action(player)
            return

        # Dash próbuje przesunąć gracza o 2 pola.
        # Jeśli drugie pole jest niedostępne, ale pierwsze jest wolne, gracz robi krótszy dash o 1 pole.
        if self.is_position_available_for_player(second_target):
            player.position = second_target
        else:
            player.position = first_target

        self._resolve_projectile_collision_at_player(player)
        player.stamina -= self.settings.dash_stamina_cost
        player.dash_cooldown = self._apply_slow_multiplier(self.settings.dash_cooldown, player)

    def _try_absolute_defence(self, player: Player) -> None:
        if player.absolute_defence_cooldown > 0:
            self._mark_invalid_action(player)
            return

        if player.mana < self.settings.absolute_defence_mana_cost:
            self._mark_invalid_action(player)
            return

        player.mana -= self.settings.absolute_defence_mana_cost
        player.absolute_defence_timer = self.settings.absolute_defence_duration
        player.absolute_defence_cooldown = self.settings.absolute_defence_cooldown

    def _try_attack_boost(self, player: Player) -> None:
        if player.attack_boost_cooldown > 0:
            self._mark_invalid_action(player)
            return

        if player.stamina < self.settings.attack_boost_stamina_cost:
            self._mark_invalid_action(player)
            return

        player.stamina -= self.settings.attack_boost_stamina_cost
        player.attack_boost_timer = self.settings.attack_boost_duration
        player.attack_boost_cooldown = self.settings.attack_boost_cooldown

    def _mark_invalid_action(self, player: Player) -> None:
        player.invalid_action = True
        player.add_reward(self.settings.invalid_action_penalty)
        player.pending_action = Actions.NONE

    def _apply_slow_multiplier(self, cooldown: int, player: Player) -> int:
        if not player.slowed_active:
            return cooldown

        return int(round(cooldown * self.settings.slowed_cooldown_multiplier))

    def _resolve_projectile_collision_at_player(self, player: Player) -> None:
        projectile = self.projectile_grid[player.row][player.col]

        if projectile is None:
            return

        if not projectile.can_damage_player(player.player_id):
            return

        self.projectile_grid[player.row][player.col] = None
        self.damage_player(
            player=player,
            damage=projectile.get_damage(),
            attacker=projectile.get_attacker_player(),
            apply_slow=True,
        )

    # ==========================================================================
    # UPDATE GRIDÓW
    # ==========================================================================

    def _update_terrain_grid(self) -> None:
        for row in range(self.settings.height):
            for col in range(self.settings.width):
                self.terrain_grid[row][col].update()

    def _update_projectile_grid(self) -> None:
        # Snapshot pozycji, żeby pocisk przesunięty w tej samej turze nie wykonał update drugi raz.
        positions = []

        for row in range(self.settings.height):
            for col in range(self.settings.width):
                if self.projectile_grid[row][col] is not None:
                    positions.append(Position(row, col))

        for position in positions:
            projectile = self.projectile_grid[position.row][position.col]

            if projectile is None:
                continue

            projectile.update()

    # ==========================================================================
    # ZASADY / KOLIZJE / OBRAŻENIA
    # ==========================================================================

    def is_inside(self, position: Position) -> bool:
        return (
            0 <= position.row < self.settings.height
            and 0 <= position.col < self.settings.width
        )

    def is_walkable(self, position: Position) -> bool:
        if not self.is_inside(position):
            return False

        return self.terrain_grid[position.row][position.col].is_walkable()

    def is_position_available_for_player(self, position: Position) -> bool:
        if not self.is_walkable(position):
            return False

        if self.get_player_at(position) is not None:
            return False

        return True

    def get_player_at(self, position: Position) -> Player | None:
        if self.player1 is not None and self.player1.position == position:
            return self.player1

        if self.player2 is not None and self.player2.position == position:
            return self.player2

        return None

    def get_player(self, player_id: int) -> Player:
        if player_id == 1:
            return self.player1

        if player_id == 2:
            return self.player2

        raise ValueError(f"Nieznane player_id: {player_id}")

    def get_opponent(self, player_id: int) -> Player:
        if player_id == 1:
            return self.player2

        if player_id == 2:
            return self.player1

        raise ValueError(f"Nieznane player_id: {player_id}")

    def damage_player(
        self,
        player: Player,
        damage: float,
        attacker: Player | None,
        apply_slow: bool = True,
    ) -> None:
        if player.absolute_defence_active:
            return

        if attacker is not None and attacker.attack_boost_active:
            damage *= self.settings.attack_boost_multiplier

        actual_damage = min(player.hp, damage)
        player.hp -= actual_damage

        player.add_reward(-actual_damage * self.settings.damage_taken_penalty_scale)

        if attacker is not None:
            attacker.add_reward(actual_damage * self.settings.damage_dealt_reward_scale)

        if apply_slow:
            player.slowed_timer = self.settings.slowed_duration

    # ==========================================================================
    # KONIEC GRY
    # ==========================================================================

    def _has_dead_player(self) -> bool:
        return self.player1.hp <= 0 or self.player2.hp <= 0

    def check_is_over(self) -> bool:
        if self._has_dead_player():
            self.is_over = True
            self.winner_id = self.choose_winner()
            return True

        if self.current_step >= self.settings.episode_max_steps:
            self.is_over = True
            self.winner_id = self.choose_winner()
            return True

        return False

    def choose_winner(self) -> int:
        if self.player1.hp <= 0 and self.player2.hp <= 0:
            return 0

        if self.player1.hp <= 0:
            return 2

        if self.player2.hp <= 0:
            return 1

        if self.player1.hp > self.player2.hp:
            return 1

        if self.player2.hp > self.player1.hp:
            return 2

        return 0

    def _apply_final_rewards(self) -> None:
        if self.winner_id == 1:
            self.player1.add_reward(self.settings.win_reward_bonus)
            self.score_player1 += 1
        elif self.winner_id == 2:
            self.player2.add_reward(self.settings.win_reward_bonus)
            self.score_player2 += 1

        if self.winner_id in (1, 2):
            winner = self.get_player(self.winner_id)
            remaining_ratio = max(
                0.0,
                (self.settings.episode_max_steps - self.current_step) / self.settings.episode_max_steps,
            )
            winner.add_reward(remaining_ratio * self.settings.time_reward_bonus)

    # ==========================================================================
    # OBSERWACJE
    # ==========================================================================

    def get_lite_grid(self, player_id: int = 1) -> np.ndarray:
        grid = np.zeros((self.settings.height, self.settings.width), dtype=np.int64)

        for row in range(self.settings.height):
            for col in range(self.settings.width):
                grid[row, col] = self.terrain_grid[row][col].to_num_value_lite()

        # Nałożenie pocisków.
        for row in range(self.settings.height):
            for col in range(self.settings.width):
                projectile = self.projectile_grid[row][col]
                if projectile is not None:
                    grid[row, col] = projectile.to_num_value_lite()

        # Nałożenie graczy z perspektywy player_id:
        # 1 = player1/gracz, 2 = przeciwnik.
        current_player = self.get_player(player_id)
        opponent = self.get_opponent(player_id)

        grid[current_player.row, current_player.col] = 1
        grid[opponent.row, opponent.col] = 2

        return grid

    def get_terrain_grid_full(self) -> np.ndarray:
        grid = np.zeros((self.settings.height, self.settings.width, 3), dtype=np.float32)

        for row in range(self.settings.height):
            for col in range(self.settings.width):
                grid[row, col] = self.terrain_grid[row][col].to_num_value_full()

        return grid

    def get_projectile_grid_full(self, player_id: int = 1) -> np.ndarray:
        grid = np.zeros((self.settings.height, self.settings.width, 3), dtype=np.float32)

        for row in range(self.settings.height):
            for col in range(self.settings.width):
                projectile = self.projectile_grid[row][col]

                if projectile is None:
                    continue

                projectile_type, direction, charge = projectile.to_num_value_full()

                if isinstance(projectile, PlayerProjectile):
                    if projectile.owner_id == player_id:
                        projectile_type = 1.0
                    else:
                        projectile_type = 2.0

                grid[row, col] = projectile_type, direction, charge

        return grid

    def get_rl_obs(self, player_id: int = 1) -> torch.Tensor:
        terrain = self.get_terrain_grid_full().reshape(-1)
        projectiles = self.get_projectile_grid_full(player_id=player_id).reshape(-1)

        player = self.get_player(player_id)
        opponent = self.get_opponent(player_id)

        player_stats = np.array(player.to_stats_vector(relative_to_player_id=player_id), dtype=np.float32)
        opponent_stats = np.array(opponent.to_stats_vector(relative_to_player_id=player_id), dtype=np.float32)

        obs = np.concatenate([
            terrain.astype(np.float32),
            projectiles.astype(np.float32),
            player_stats,
            opponent_stats,
        ])

        return torch.tensor(obs, dtype=torch.float32)

    def get_basic_obs(self, player_id: int = 1):
        player = self.get_player(player_id)

        adjacent_flags = self._get_opponent_adjacent_flags(player_id)
        inline_flags = self._get_opponent_inline_flags(player_id)
        projectile_flags = self._get_enemy_projectile_near_flags(player_id)

        stats = [
            # zasoby
            self._bucket_resource(player.hp, self.settings.max_hp),
            self._bucket_resource(player.strength, self.settings.max_strength),
            self._bucket_resource(player.mana, self.settings.max_mana),
            self._bucket_resource(player.stamina, self.settings.max_stamina),

            # stany specjalne
            int(player.absolute_defence_active),
            int(player.attack_boost_active),
            int(player.slowed_active),

            # dostępne pola wokół gracza
            int(self._is_available_field(player, -1, 0)),
            int(self._is_available_field(player, 1, 0)),
            int(self._is_available_field(player, 0, -1)),
            int(self._is_available_field(player, 0, 1)),

            # możliwości akcji
            int(player.move_cooldown <= 0),
            int(player.melee_cooldown <= 0 and player.strength >= self.settings.melee_strength_cost),
            int(player.ranged_cooldown <= 0 and player.mana >= self.settings.ranged_mana_cost),
            int(player.dash_cooldown <= 0 and player.stamina >= self.settings.dash_stamina_cost),
            int(player.absolute_defence_cooldown <= 0 and player.mana >= self.settings.absolute_defence_mana_cost),
            int(player.attack_boost_cooldown <= 0 and player.stamina >= self.settings.attack_boost_stamina_cost),

            # przeciwnik obok
            *adjacent_flags,

            # przeciwnik w linii
            *inline_flags,

            # pociski w pobliżu
            *projectile_flags,
        ]

        grid = self.get_lite_grid(player_id=player_id)

        return stats, grid

    def _is_available_field(self, player: Player, d_row: int, d_col: int) -> bool:
        target = player.position.moved(d_row, d_col)
        return self.is_position_available_for_player(target)

    def _get_opponent_adjacent_flags(self, player_id: int) -> list[int]:
        player = self.get_player(player_id)
        opponent = self.get_opponent(player_id)

        return [
            int(opponent.position == player.position.moved(-1, 0)),
            int(opponent.position == player.position.moved(1, 0)),
            int(opponent.position == player.position.moved(0, -1)),
            int(opponent.position == player.position.moved(0, 1)),
        ]

    def _get_opponent_inline_flags(self, player_id: int) -> list[int]:
        player = self.get_player(player_id)
        opponent = self.get_opponent(player_id)

        up = 0
        down = 0
        left = 0
        right = 0

        # góra
        if opponent.col == player.col and opponent.row < player.row:
            blocked = False

            for row in range(opponent.row + 1, player.row):
                if self.terrain_grid[row][player.col].blocks_projectiles():
                    blocked = True
                    break

            if not blocked:
                up = 1

        # dół
        if opponent.col == player.col and opponent.row > player.row:
            blocked = False

            for row in range(player.row + 1, opponent.row):
                if self.terrain_grid[row][player.col].blocks_projectiles():
                    blocked = True
                    break

            if not blocked:
                down = 1

        # lewo
        if opponent.row == player.row and opponent.col < player.col:
            blocked = False

            for col in range(opponent.col + 1, player.col):
                if self.terrain_grid[player.row][col].blocks_projectiles():
                    blocked = True
                    break

            if not blocked:
                left = 1

        # prawo
        if opponent.row == player.row and opponent.col > player.col:
            blocked = False

            for col in range(player.col + 1, opponent.col):
                if self.terrain_grid[player.row][col].blocks_projectiles():
                    blocked = True
                    break

            if not blocked:
                right = 1

        return [up, down, left, right]

    def _get_enemy_projectile_near_flags(self, player_id: int, distance: int = 3) -> list[int]:
        player = self.get_player(player_id)

        up = 0
        down = 0
        left = 0
        right = 0

        for step in range(1, distance + 1):
            # góra
            row = player.row - step
            if row >= 0:
                projectile = self.projectile_grid[row][player.col]
                if self._is_dangerous_projectile(projectile, player_id):
                    up = 1

            # dół
            row = player.row + step
            if row < self.settings.height:
                projectile = self.projectile_grid[row][player.col]
                if self._is_dangerous_projectile(projectile, player_id):
                    down = 1

            # lewo
            col = player.col - step
            if col >= 0:
                projectile = self.projectile_grid[player.row][col]
                if self._is_dangerous_projectile(projectile, player_id):
                    left = 1

            # prawo
            col = player.col + step
            if col < self.settings.width:
                projectile = self.projectile_grid[player.row][col]
                if self._is_dangerous_projectile(projectile, player_id):
                    right = 1

        return [up, down, left, right]

    @staticmethod
    def _is_dangerous_projectile(projectile, player_id: int) -> bool:
        if projectile is None:
            return False

        if isinstance(projectile, PlayerProjectile):
            return projectile.owner_id != player_id

        return True

    @staticmethod
    def _bucket_resource(value: float, max_value: float) -> int:
        ratio = value / max_value if max_value > 0 else 0.0

        if ratio <= 0.25:
            return 0
        if ratio <= 0.50:
            return 1
        if ratio <= 0.75:
            return 2
        return 3
