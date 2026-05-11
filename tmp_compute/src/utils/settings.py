from dataclasses import dataclass, field
import pickle

from src.utils.enums import *
from src.utils.game_time import TimeManager


@dataclass
class GlobalSettings:
    # ============================= ŚRODOWISKO =============================

    # --- rozmiar mapy ---
    width: int = 10
    height: int = 10

    # --- tworzenie mapy ---
    add_reversed_positions: bool = True
    walls_ratio: float = 0.16
    turret_ratio: float = 0.04
    trap_ratio: float = 0.05

    # --- czas ---
    tm: TimeManager = field(default_factory=TimeManager)
    episode_max_steps: int = 2_400
    eval_after: int = 20_000
    eval_episodes_training_base_agent = 5
    eval_episodes_training_previous_agent = 5
    eval_episodes_test_base_agent = 5
    eval_episodes_test_previous_agent = 5
    total_max_training_steps = 1_000_000

    # --- cooldowny ---
    move_cooldown: int = 2
    melee_cooldown: int = 3
    ranged_cooldown: int = 5
    dash_cooldown: int = 8
    absolute_defence_cooldown: int = 48
    attack_boost_cooldown: int = 40

    # --- czasy trwania efektów ---
    absolute_defence_duration: int = 8
    attack_boost_duration: int = 16
    slowed_duration: int = 12

    # --- mnożniki efektów ---
    attack_boost_multiplier: float = 1.5
    slowed_cooldown_multiplier: float = 1.5

    # --- zasoby ---
    max_hp: float = 100
    max_strength: float = 100
    max_mana: float = 100
    max_stamina: float = 100

    # --- regeneracja zasobów ---
    strength_regen: float = 2.0
    mana_regen: float = 1.25
    stamina_regen: float = 1.75

    # --- koszty akcji ---
    melee_strength_cost: float = 10.0
    ranged_mana_cost: float = 20.0
    dash_stamina_cost: float = 20.0
    absolute_defence_mana_cost: float = 40.0
    attack_boost_stamina_cost: float = 40.0

    # --- obrażenia ---
    melee_damage: float = 3.0
    projectile_damage: float = 7.0
    turret_projectile_damage: float = 3.0
    trap_damage: float = 0.5

    # --- efekty trafień ---
    melee_applies_slow: bool = True
    projectile_applies_slow: bool = True

    # --- zasada działania pocisków/wieżyczek---
    projectile_move_cooldown: int = 1
    turret_shoot_interval: int = 16

    # --- nagrody ---
    damage_dealt_reward_scale: float = 1.0
    damage_taken_penalty_scale: float = 1.0
    invalid_action_penalty: float = -0.001
    step_penalty: float = -0.005

    win_reward_bonus: float = 25.0
    time_reward_bonus: float = 25.0

    # --- inicjalizacja graczy ---
    initial_hp: float = 100
    initial_strength: float = 100
    initial_mana: float = 100
    initial_stamina: float = 100

    # ================================== GUI ==================================

    empty_tile_color: tuple[int, int, int] = (120, 200, 120)
    wall_tile_color: tuple[int, int, int] = (55, 55, 55)
    trap_tile_color: tuple[int, int, int] = (190, 40, 40)
    turret_tile_color: tuple[int, int, int] = (245, 180, 40)

    player_1_color: tuple[int, int, int] = (75, 119, 209)
    player_2_color: tuple[int, int, int] = (234, 51, 35)

    hp_bar_color: tuple[int, int, int] = (220, 40, 40)
    strength_bar_color: tuple[int, int, int] = (230, 210, 40)
    mana_bar_color: tuple[int, int, int] = (40, 110, 230)
    stamina_bar_color: tuple[int, int, int] = (40, 190, 80)

    # ==================== Drzewo decyzyjne (agent bazowy) ====================

    random_movement_chance: float = 0.3

    # ============================ Proces treningu ============================

    advanced_training_possible: bool = True
    advanced_training_winrate_threshold: float = 0.7
    advanced_training_required_evaluations: int = 1
    advanced_model_chance: float = 0.7

    # ================================= AI/RL =================================

    # --- shared ---
    gamma: float = 0.99
    learning_rate: float = 1e-4
    force_cpu_training: bool = True
    torch_num_threads: int = 1

    f_activation: ActivationFunction = ActivationFunction.RELU
    hidden_dims: int = 128
    hidden_layers: int = 4

    # --- value-based shared ---
    replay_buffer_size: int = 100_000
    batch_size: int = 64
    warmup_steps: int = 1_000

    target_update_freq: int = 1_000
    optimize_every: int = 4

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 100_000

    # --- N-step ---
    n_step: int = 1

    # --- PER ---
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_end: float = 1.0
    per_epsilon: float = 1e-6

    # --- Noisy ---
    sigma_init: float = 0.5

    # --- Distributional ---
    num_atoms: int = 51
    v_min: float = -150.0
    v_max: float = 150.0

    # --- policy-gradient shared ---
    rollout_steps: int = 2048
    entropy_coef: float = 0.01

    # --- actor-critic shared ---
    value_loss_coef: float = 0.5
    gae_lambda: float = 0.95

    # --- PPO ---
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4
    ppo_batch_size: int = 64

    def next_eval_step(self, current_training_steps: int) -> int:
        return min(
            current_training_steps + self.eval_after,
            self.total_max_training_steps,
        )

    def save(self, filename: str):
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        with open(filename, "rb") as f:
            return pickle.load(f)
