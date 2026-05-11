from collections import deque

from src.env.position import Position
from src.utils.enums import Actions


BFS_DIRECTIONS = [
    (-1, 0, Actions.MOVE_UP),
    (1, 0, Actions.MOVE_DOWN),
    (0, -1, Actions.MOVE_LEFT),
    (0, 1, Actions.MOVE_RIGHT),
]


WALKABLE_VALUES = {
    1,  # player1
    2,  # player2
    3,  # empty/trap
}


def find_position_by_value(grid, value: int) -> Position | None:
    rows = len(grid)
    cols = len(grid[0])

    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == value:
                return Position(row, col)

    return None


def get_bfs_moves(
    grid,
    start: Position | None = None,
    goal: Position | None = None,
) -> list[Actions]:
    rows = len(grid)
    cols = len(grid[0])

    # Domyślnie BFS szuka trasy od 1 do 2.
    # Dzięki temu DecisionTreeAgent może dostać po prostu grid z perspektywy gracza:
    # 1 = player1/gracz, 2 = przeciwnik, 3 = pole możliwe do przejścia.
    if start is None:
        start = find_position_by_value(grid, 1)

    if goal is None:
        goal = find_position_by_value(grid, 2)

    if start is None or goal is None:
        return [None]

    queue = deque()
    queue.append((start, []))

    visited = {
        (start.row, start.col),
    }

    while queue:
        current_position, current_path = queue.popleft()

        if current_position == goal:
            return current_path

        for d_row, d_col, action in BFS_DIRECTIONS:
            next_row = current_position.row + d_row
            next_col = current_position.col + d_col

            if next_row < 0 or next_row >= rows:
                continue

            if next_col < 0 or next_col >= cols:
                continue

            if (next_row, next_col) in visited:
                continue

            grid_value = grid[next_row][next_col]

            if grid_value not in WALKABLE_VALUES:
                continue

            visited.add((next_row, next_col))

            next_position = Position(next_row, next_col)
            next_path = current_path + [action]

            queue.append((next_position, next_path))

    return [None]


def path_exists(
    grid,
    start: Position | None = None,
    goal: Position | None = None,
) -> bool:
    moves = get_bfs_moves(
        grid=grid,
        start=start,
        goal=goal,
    )

    return moves != [None]
