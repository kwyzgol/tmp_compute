from src.utils.enums import Actions, Direction


MOVE_ACTIONS = {
    Actions.MOVE_UP,
    Actions.MOVE_DOWN,
    Actions.MOVE_LEFT,
    Actions.MOVE_RIGHT,
}

MELEE_ACTIONS = {
    Actions.ATTACK_MELEE_UP,
    Actions.ATTACK_MELEE_DOWN,
    Actions.ATTACK_MELEE_LEFT,
    Actions.ATTACK_MELEE_RIGHT,
}

RANGED_ACTIONS = {
    Actions.ATTACK_RANGED_UP,
    Actions.ATTACK_RANGED_DOWN,
    Actions.ATTACK_RANGED_LEFT,
    Actions.ATTACK_RANGED_RIGHT,
}

DASH_ACTIONS = {
    Actions.DASH_UP,
    Actions.DASH_DOWN,
    Actions.DASH_LEFT,
    Actions.DASH_RIGHT,
}


ACTION_TO_DIRECTION = {
    Actions.MOVE_UP: Direction.UP,
    Actions.MOVE_DOWN: Direction.DOWN,
    Actions.MOVE_LEFT: Direction.LEFT,
    Actions.MOVE_RIGHT: Direction.RIGHT,

    Actions.ATTACK_MELEE_UP: Direction.UP,
    Actions.ATTACK_MELEE_DOWN: Direction.DOWN,
    Actions.ATTACK_MELEE_LEFT: Direction.LEFT,
    Actions.ATTACK_MELEE_RIGHT: Direction.RIGHT,

    Actions.ATTACK_RANGED_UP: Direction.UP,
    Actions.ATTACK_RANGED_DOWN: Direction.DOWN,
    Actions.ATTACK_RANGED_LEFT: Direction.LEFT,
    Actions.ATTACK_RANGED_RIGHT: Direction.RIGHT,

    Actions.DASH_UP: Direction.UP,
    Actions.DASH_DOWN: Direction.DOWN,
    Actions.DASH_LEFT: Direction.LEFT,
    Actions.DASH_RIGHT: Direction.RIGHT,
}


DIRECTION_TO_DELTA = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}


def direction_to_delta(direction: Direction) -> tuple[int, int]:
    return DIRECTION_TO_DELTA[direction]


def action_to_direction(action: Actions) -> Direction | None:
    return ACTION_TO_DIRECTION.get(action)


def is_move_action(action: Actions) -> bool:
    return action in MOVE_ACTIONS


def is_melee_action(action: Actions) -> bool:
    return action in MELEE_ACTIONS


def is_ranged_action(action: Actions) -> bool:
    return action in RANGED_ACTIONS


def is_dash_action(action: Actions) -> bool:
    return action in DASH_ACTIONS