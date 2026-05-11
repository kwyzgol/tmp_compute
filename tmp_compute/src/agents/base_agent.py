import random

from src.agents.agents import Agent
from src.env.game_env import BASIC_OBS_COLUMNS
from src.utils.enums import Actions
from src.utils.grid_pathfinding import get_bfs_moves
from src.utils.settings import GlobalSettings


class DecisionTreeAgentAdvanced(Agent):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!! OKAZAŁ SIĘ ZBYT TRUDNY DO TRENINGU RL !!!
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

     >> Nowy agent DecisionTreeAgent znajduje się poniżej.

    Prosty, deterministyczny agent bazowy zapisany jako ręczne drzewo decyzyjne.

    Celem klasy nie jest uzyskanie optymalnego zachowania, tylko stworzenie czytelnego
    punktu odniesienia dla metod uczenia ze wzmocnieniem.

    Agent używa obserwacji basic:
        obs = (stats, grid)

    gdzie:
        stats - lista prostych cech opisowych,
        grid  - uproszczona mapa używana przez BFS.

    Hierarchia decyzji:
        1. unikanie pocisków,
        2. obrona absolutna przy niskim HP i bliskim zagrożeniu,
        3. wzmocnienie ataku, jeśli przeciwnik jest obok,
        4. atak wręcz, jeśli przeciwnik jest obok,
        5. atak dystansowy, jeśli przeciwnik jest w tej samej linii,
        6. ruch w stronę przeciwnika według BFS (lub z pewnym prawdopodobieństwem losowo,
        7. brak akcji, jeśli nie da się wyznaczyć ruchu.

    Drzewo jest celowo zapisane prostymi regułami, aby jego działanie było łatwe
    do opisania i uzasadnienia w pracy.
    """

    DIRECTIONS = [
        "up",
        "down",
        "left",
        "right",
    ]

    MELEE_ACTIONS = {
        "up": Actions.ATTACK_MELEE_UP,
        "down": Actions.ATTACK_MELEE_DOWN,
        "left": Actions.ATTACK_MELEE_LEFT,
        "right": Actions.ATTACK_MELEE_RIGHT,
    }

    RANGED_ACTIONS = {
        "up": Actions.ATTACK_RANGED_UP,
        "down": Actions.ATTACK_RANGED_DOWN,
        "left": Actions.ATTACK_RANGED_LEFT,
        "right": Actions.ATTACK_RANGED_RIGHT,
    }

    MOVE_ACTIONS = {
        "up": Actions.MOVE_UP,
        "down": Actions.MOVE_DOWN,
        "left": Actions.MOVE_LEFT,
        "right": Actions.MOVE_RIGHT,
    }

    DASH_ACTIONS = {
        "up": Actions.DASH_UP,
        "down": Actions.DASH_DOWN,
        "left": Actions.DASH_LEFT,
        "right": Actions.DASH_RIGHT,
    }

    # Kierunki ucieczki sprawdzane przy zagrożeniu pociskiem.
    # Dla pocisku z góry lub z dołu preferowany jest ruch w bok.
    # Dla pocisku z lewej lub z prawej preferowany jest ruch pionowy.
    PROJECTILE_ESCAPE_DIRECTIONS = {
        "up": ["left", "right"],
        "down": ["left", "right"],
        "left": ["up", "down"],
        "right": ["up", "down"],
    }

    def __init__(
        self,
        settings: GlobalSettings = GlobalSettings(),
        name: str = "DecisionTreeAdvanced",
    ):
        super().__init__(name)

        self.settings = settings
        self.rng = random.Random()

        self.basic_obs_index = {
            column: idx
            for idx, column in enumerate(BASIC_OBS_COLUMNS)
        }

    # ==================================================================================
    # Główna decyzja agenta
    # ==================================================================================

    def predict(self, obs=None) -> Actions:
        if obs is None:
            return Actions.NONE

        stats, grid = obs

        # ==============================================================================
        # 1. Najwyższy priorytet: unikanie pocisków.
        # ==============================================================================

        for projectile_direction in self.DIRECTIONS:
            escape_action = self._try_escape_projectile(stats, projectile_direction)

            if escape_action is not None:
                return escape_action

        # ==============================================================================
        # 2. Niskie HP + przeciwnik obok lub w tej samej linii => obrona absolutna.
        # ==============================================================================

        # hp_bucket == 0 && can_absolute_defence == true && opponent_adjacent_any == true
        if self._has_low_hp(stats) and self._get(stats, "can_absolute_defence"):
            if self._opponent_adjacent_any(stats):
                return Actions.ABSOLUTE_DEFENSE_ABILITY

        # hp_bucket == 0 && can_absolute_defence == true && opponent_inline_any == true
        if self._has_low_hp(stats) and self._get(stats, "can_absolute_defence"):
            if self._opponent_inline_any(stats):
                return Actions.ABSOLUTE_DEFENSE_ABILITY

        # ==============================================================================
        # 3. Przeciwnik obok + możliwe wzmocnienie + możliwy atak wręcz => boost.
        # ==============================================================================

        # opponent_adjacent_any == true && can_attack_boost == true && can_melee == true
        if self._opponent_adjacent_any(stats):
            if self._get(stats, "can_attack_boost") and self._get(stats, "can_melee"):
                return Actions.ATTACK_BOOST_ABILITY

        # ==============================================================================
        # 4. Przeciwnik obok + możliwy atak wręcz => atak wręcz.
        # ==============================================================================

        for direction in self.DIRECTIONS:
            # can_melee == true && opponent_adjacent_{direction} == true
            if self._get(stats, "can_melee") and self._opponent_adjacent(stats, direction):
                return self.MELEE_ACTIONS[direction]

        # ==============================================================================
        # 5. Przeciwnik w tej samej linii + możliwy atak dystansowy => atak dystansowy.
        # ==============================================================================

        for direction in self.DIRECTIONS:
            # can_ranged == true && opponent_inline_{direction} == true
            if self._get(stats, "can_ranged") and self._opponent_inline(stats, direction):
                return self.RANGED_ACTIONS[direction]

        # ==============================================================================
        # 6. W pozostałych przypadkach ruch według BFS (lub z pewnym prawdopodobieństwem losowo).
        # ==============================================================================

        # path_to_enemy_exists == true
        move_action = self._select_bfs_move_or_random(grid)

        if move_action is not None:
            return move_action

        # ==============================================================================
        # 7. Brak sensownej decyzji (możliwe w przypadku, gdy BFS nie odkrył ścieżki, oznacza błąd środowiska).
        # ==============================================================================

        return Actions.NONE

    # ==================================================================================
    # Unikanie pocisków
    # ==================================================================================

    def _try_escape_projectile(self, stats, projectile_direction: str) -> Actions | None:
        """
        Próbuje uniknąć pocisku nadlatującego z danego kierunku.

        Hierarchia dla każdego kierunku zagrożenia:
            1A/1B. ruch w pierwszy lub drugi kierunek ucieczki,
            1C/1D. dash w pierwszy lub drugi kierunek ucieczki.
        """

        if not self._projectile_threat(stats, projectile_direction):
            return None

        escape_directions = self.PROJECTILE_ESCAPE_DIRECTIONS[projectile_direction]

        for escape_direction in escape_directions:
            # enemy_projectile_near == true && available_field_{escape_direction} == true && can_move == true
            if self._can_move(stats, escape_direction):
                return self.MOVE_ACTIONS[escape_direction]

        for escape_direction in escape_directions:
            # enemy_projectile_near == true && available_field_{escape_direction} == true && can_dash == true
            if self._can_dash(stats, escape_direction):
                return self.DASH_ACTIONS[escape_direction]

        return None

    # ==================================================================================
    # Odczyt cech z obs_basic
    # ==================================================================================

    def _get(self, stats, column: str, default: int = 0) -> int:
        idx = self.basic_obs_index.get(column)

        if idx is None:
            return default

        if idx >= len(stats):
            return default

        return int(stats[idx])

    def _has_low_hp(self, stats) -> bool:
        return bool(
            self._get(stats, "hp_bucket") == 0
        )

    def _projectile_threat(self, stats, direction: str) -> bool:
        return bool(
            self._get(stats, f"enemy_projectile_near_{direction}")
        )

    def _can_move(self, stats, direction: str) -> bool:
        return bool(
            self._get(stats, "can_move")
            and self._get(stats, f"available_field_{direction}")
        )

    def _can_dash(self, stats, direction: str) -> bool:
        return bool(
            self._get(stats, "can_dash")
            and self._get(stats, f"available_field_{direction}")
        )

    def _opponent_adjacent(self, stats, direction: str) -> bool:
        return bool(self._get(stats, f"opponent_adjacent_{direction}"))

    def _opponent_inline(self, stats, direction: str) -> bool:
        return bool(self._get(stats, f"opponent_inline_{direction}"))

    def _opponent_adjacent_any(self, stats) -> bool:
        return any(
            self._opponent_adjacent(stats, direction)
            for direction in self.DIRECTIONS
        )

    def _opponent_inline_any(self, stats) -> bool:
        return any(
            self._opponent_inline(stats, direction)
            for direction in self.DIRECTIONS
        )

    # ==================================================================================
    # Ruch przez BFS lub z pewnym prawdopodobieństwem losowo
    # ==================================================================================

    def _select_bfs_move_or_random(self, grid) -> Actions:
        random_probability = self.settings.random_movement_chance
        use_random_move = self.rng.random() < random_probability

        if use_random_move:
            return self.rng.choice(list(self.MOVE_ACTIONS.values()))
        else:
            bfs_moves = get_bfs_moves(grid)
            return bfs_moves[0]


class DecisionTreeAgent(DecisionTreeAgentAdvanced):
    """
    Uproszczony agent bazowy do treningu RL.

    Agent maksymalizuje kontakt z graczem bez defensywnych zdolnosci, boostow i
    unikow pociskow:
        1. gdy przeciwnik jest obok, atakuje wrecz, jesli moze,
        2. gdy przeciwnik jest obok, ale melee jest niedostepne, czeka,
        3. gdy przeciwnik jest w linii, strzela, jesli moze,
        4. w pozostalych przypadkach idzie BFS-em do przeciwnika.
    """

    def __init__(
        self,
        settings: GlobalSettings = GlobalSettings(),
        name: str = "DecisionTree",
    ):
        super().__init__(settings=settings, name=name)

    def predict(self, obs=None) -> Actions:
        if obs is None:
            return Actions.NONE

        stats, grid = obs

        for direction in self.DIRECTIONS:
            if self._opponent_adjacent(stats, direction):
                if self._get(stats, "can_melee"):
                    return self.MELEE_ACTIONS[direction]

                return Actions.NONE

        for direction in self.DIRECTIONS:
            if self._opponent_inline(stats, direction):
                if self._get(stats, "can_ranged"):
                    return self.RANGED_ACTIONS[direction]

                return self._select_bfs_move_or_random(grid)

        return self._select_bfs_move_or_random(grid)
