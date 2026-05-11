from dataclasses import dataclass

from src.env.position import Position
from src.utils.enums import Actions
from src.utils.settings import GlobalSettings


@dataclass
class Player:
    player_id: int
    position: Position
    settings: GlobalSettings

    hp: float = 0.0
    strength: float = 0.0
    mana: float = 0.0
    stamina: float = 0.0

    move_cooldown: int = 0
    melee_cooldown: int = 0
    ranged_cooldown: int = 0
    dash_cooldown: int = 0
    absolute_defence_cooldown: int = 0
    attack_boost_cooldown: int = 0

    absolute_defence_timer: int = 0
    attack_boost_timer: int = 0
    slowed_timer: int = 0

    pending_action: Actions = Actions.NONE
    next_reward: float = 0.0
    episode_reward: float = 0.0
    invalid_action: bool = False

    def __post_init__(self):
        self.hp = self.settings.initial_hp
        self.strength = self.settings.initial_strength
        self.mana = self.settings.initial_mana
        self.stamina = self.settings.initial_stamina

    @property
    def row(self) -> int:
        return self.position.row

    @property
    def col(self) -> int:
        return self.position.col

    @property
    def is_alive(self) -> bool:
        return self.hp > 0

    @property
    def absolute_defence_active(self) -> bool:
        return self.absolute_defence_timer > 0

    @property
    def attack_boost_active(self) -> bool:
        return self.attack_boost_timer > 0

    @property
    def slowed_active(self) -> bool:
        return self.slowed_timer > 0

    def reset_step_reward(self) -> None:
        self.next_reward = self.settings.step_penalty
        self.invalid_action = False

    def add_reward(self, value: float) -> None:
        self.next_reward += value
        self.episode_reward += value

    def update_timers_and_regen(self) -> None:
        self._decrease_cooldowns()
        self._decrease_effect_timers()
        self._regenerate_resources()

    def _decrease_cooldowns(self) -> None:
        self.move_cooldown = max(0, self.move_cooldown - 1)
        self.melee_cooldown = max(0, self.melee_cooldown - 1)
        self.ranged_cooldown = max(0, self.ranged_cooldown - 1)
        self.dash_cooldown = max(0, self.dash_cooldown - 1)
        self.absolute_defence_cooldown = max(0, self.absolute_defence_cooldown - 1)
        self.attack_boost_cooldown = max(0, self.attack_boost_cooldown - 1)

    def _decrease_effect_timers(self) -> None:
        self.absolute_defence_timer = max(0, self.absolute_defence_timer - 1)
        self.attack_boost_timer = max(0, self.attack_boost_timer - 1)
        self.slowed_timer = max(0, self.slowed_timer - 1)

    def _regenerate_resources(self) -> None:
        self.strength = min(self.settings.max_strength, self.strength + self.settings.strength_regen)
        self.mana = min(self.settings.max_mana, self.mana + self.settings.mana_regen)
        self.stamina = min(self.settings.max_stamina, self.stamina + self.settings.stamina_regen)

    @staticmethod
    def _ratio(value: float, maximum: float) -> float:
        if maximum <= 0:
            return 1.0

        return max(0.0, min(1.0, value / maximum))

    @staticmethod
    def _cooldown_readiness(cooldown: int, maximum: int) -> float:
        if maximum <= 0:
            return 1.0

        return 1.0 - Player._ratio(cooldown, maximum)

    def to_stats_vector(self, relative_to_player_id: int) -> list[float]:
        is_self = self.player_id == relative_to_player_id
        relative_id = 1 if is_self else 2

        return [
            float(relative_id),
            self._ratio(self.hp, self.settings.max_hp),
            self._ratio(self.strength, self.settings.max_strength),
            self._ratio(self.mana, self.settings.max_mana),
            self._ratio(self.stamina, self.settings.max_stamina),
            self.row / max(1, self.settings.height - 1),
            self.col / max(1, self.settings.width - 1),
            float(self.absolute_defence_active),
            float(self.attack_boost_active),
            float(self.slowed_active),
            self._cooldown_readiness(self.move_cooldown, self.settings.move_cooldown),
            self._cooldown_readiness(self.melee_cooldown, self.settings.melee_cooldown),
            self._cooldown_readiness(self.ranged_cooldown, self.settings.ranged_cooldown),
            self._cooldown_readiness(self.dash_cooldown, self.settings.dash_cooldown),
            self._cooldown_readiness(
                self.absolute_defence_cooldown,
                self.settings.absolute_defence_cooldown,
            ),
            self._cooldown_readiness(
                self.attack_boost_cooldown,
                self.settings.attack_boost_cooldown,
            ),
            float(self.move_cooldown <= 0),
            float(
                self.melee_cooldown <= 0
                and self.strength >= self.settings.melee_strength_cost
            ),
            float(
                self.ranged_cooldown <= 0
                and self.mana >= self.settings.ranged_mana_cost
            ),
            float(
                self.dash_cooldown <= 0
                and self.stamina >= self.settings.dash_stamina_cost
            ),
            float(
                self.absolute_defence_cooldown <= 0
                and self.mana >= self.settings.absolute_defence_mana_cost
            ),
            float(
                self.attack_boost_cooldown <= 0
                and self.stamina >= self.settings.attack_boost_stamina_cost
            ),
        ]
