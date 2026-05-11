from dataclasses import dataclass


@dataclass
class Position:
    row: int
    col: int

    def moved(self, d_row: int, d_col: int) -> "Position":
        return Position(
            row=self.row + d_row,
            col=self.col + d_col,
        )