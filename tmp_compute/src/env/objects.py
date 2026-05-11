from src.env.position import Position
from src.utils.action_helpers import direction_to_delta
from src.utils.enums import Direction


class GameObject:
    def __init__(self, env, position: Position):
        self.env = env
        self.position = position

    def update(self) -> None:
        raise NotImplementedError

    def is_walkable(self) -> bool:
        raise NotImplementedError

    def blocks_projectiles(self) -> bool:
        raise NotImplementedError

    def to_num_value_lite(self) -> int:
        raise NotImplementedError

    def to_num_value_full(self) -> tuple[float, float, float]:
        raise NotImplementedError


class EmptyField(GameObject):
    def update(self) -> None:
        pass

    def is_walkable(self) -> bool:
        return True

    def blocks_projectiles(self) -> bool:
        return False

    def to_num_value_lite(self) -> int:
        return 3

    def to_num_value_full(self) -> tuple[float, float, float]:
        # ogólny typ: pole, szczegółowy typ: empty, cooldown: 0
        return 3.0, 0.0, 0.0


class Trap(EmptyField):
    def update(self) -> None:
        player = self.env.get_player_at(self.position)

        if player is None:
            return

        self.env.damage_player(
            player=player,
            damage=self.env.settings.trap_damage,
            attacker=None,
            apply_slow=False,
        )

    def to_num_value_lite(self) -> int:
        return 3

    def to_num_value_full(self) -> tuple[float, float, float]:
        # ogólny typ: pole, szczegółowy typ: trap, cooldown: 0
        return 3.0, 1.0, 0.0


class Wall(GameObject):
    def update(self) -> None:
        pass

    def is_walkable(self) -> bool:
        return False

    def blocks_projectiles(self) -> bool:
        return True

    def to_num_value_lite(self) -> int:
        return 4

    def to_num_value_full(self) -> tuple[float, float, float]:
        # ogólny typ: blokada, szczegółowy typ: wall, cooldown: 0
        return 4.0, 0.0, 0.0


class Turret(Wall):
    def __init__(self, env, position: Position):
        super().__init__(env, position)
        self.cooldown = env.settings.turret_shoot_interval

    def update(self) -> None:
        self.cooldown -= 1

        if self.cooldown > 0:
            return

        self.cooldown = self.env.settings.turret_shoot_interval
        self._shoot_in_all_directions()

    def _shoot_in_all_directions(self) -> None:
        for direction in Direction:
            d_row, d_col = direction_to_delta(direction)
            target = self.position.moved(d_row, d_col)

            if not self.env.is_inside(target):
                continue

            if not self.env.is_walkable(target):
                continue

            player = self.env.get_player_at(target)
            if player is not None:
                self.env.damage_player(
                    player=player,
                    damage=self.env.settings.turret_projectile_damage,
                    attacker=None,
                    apply_slow=True,
                )
                continue

            if self.env.projectile_grid[target.row][target.col] is not None:
                continue

            self.env.projectile_grid[target.row][target.col] = TurretProjectile(
                env=self.env,
                position=target,
                direction=direction,
            )

    def to_num_value_lite(self) -> int:
        return 4

    def to_num_value_full(self) -> tuple[float, float, float]:
        charge = 1.0 - (self.cooldown / self.env.settings.turret_shoot_interval)
        charge = max(0.0, min(1.0, charge))

        # ogólny typ: blokada, szczegółowy typ: turret, cooldown/charge: 0-1
        return 4.0, 1.0, charge


class Projectile(GameObject):
    def __init__(self, env, position: Position, direction: Direction):
        super().__init__(env, position)
        self.direction = direction
        self.cooldown = env.settings.projectile_move_cooldown

    def update(self) -> None:
        self.cooldown -= 1

        if self.cooldown > 0:
            return

        self.cooldown = self.env.settings.projectile_move_cooldown
        self._move_or_hit()

    def _move_or_hit(self) -> None:
        current_player = self.env.get_player_at(self.position)
        if current_player is not None and self.can_damage_player(current_player.player_id):
            self.env.projectile_grid[self.position.row][self.position.col] = None
            self.env.damage_player(
                player=current_player,
                damage=self.get_damage(),
                attacker=self.get_attacker_player(),
                apply_slow=True,
            )
            return

        d_row, d_col = direction_to_delta(self.direction)
        target = self.position.moved(d_row, d_col)

        self.env.projectile_grid[self.position.row][self.position.col] = None

        if not self.env.is_inside(target):
            return

        if self.env.terrain_grid[target.row][target.col].blocks_projectiles():
            return

        other_projectile = self.env.projectile_grid[target.row][target.col]
        if other_projectile is not None:
            self.env.projectile_grid[target.row][target.col] = None
            return

        player = self.env.get_player_at(target)
        if player is not None and self.can_damage_player(player.player_id):
            self.env.damage_player(
                player=player,
                damage=self.get_damage(),
                attacker=self.get_attacker_player(),
                apply_slow=True,
            )
            return

        self.position = target
        self.env.projectile_grid[target.row][target.col] = self

    def is_walkable(self) -> bool:
        return True

    def blocks_projectiles(self) -> bool:
        return False

    def to_num_value_lite(self) -> int:
        return 6

    def get_charge(self) -> float:
        charge = 1.0 - (self.cooldown / self.env.settings.projectile_move_cooldown)
        return max(0.0, min(1.0, charge))

    def get_direction_value(self) -> float:
        return self.direction.value / 3.0

    def to_num_value_full(self) -> tuple[float, float, float]:
        return 0.0, self.get_direction_value(), self.get_charge()

    def can_damage_player(self, player_id: int) -> bool:
        raise NotImplementedError

    def get_damage(self) -> float:
        raise NotImplementedError

    def get_attacker_player(self):
        return None


class PlayerProjectile(Projectile):
    def __init__(self, env, position: Position, direction: Direction, owner_id: int):
        super().__init__(env, position, direction)
        self.owner_id = owner_id

    def can_damage_player(self, player_id: int) -> bool:
        return player_id != self.owner_id

    def get_damage(self) -> float:
        return self.env.settings.projectile_damage

    def get_attacker_player(self):
        return self.env.get_player(self.owner_id)

    def to_num_value_full(self) -> tuple[float, float, float]:
        return float(self.owner_id), self.get_direction_value(), self.get_charge()


class TurretProjectile(Projectile):
    def can_damage_player(self, player_id: int) -> bool:
        return True

    def get_damage(self) -> float:
        return self.env.settings.turret_projectile_damage

    def to_num_value_full(self) -> tuple[float, float, float]:
        return 3.0, self.get_direction_value(), self.get_charge()
