
from __future__ import annotations
from typing import Optional, Tuple

# Simple deterministic rules:
# - If number cell's remaining mines equals remaining hidden neighbors, flag all
# - If number cell's flagged neighbors equals number, reveal all other hidden neighbors

def pick_move(board) -> Optional[Tuple[str, Tuple[int, int]]]:
    progress = False
    for x, y in board.revealed_number_cells():
        c = board.grid[y][x]
        nbrs = board.neighbors(x, y)
        hidden = [(nx, ny) for nx, ny in nbrs if not board.grid[ny][nx].is_revealed and not board.grid[ny][nx].is_flagged]
        flagged = [(nx, ny) for nx, ny in nbrs if board.grid[ny][nx].is_flagged]
        if not hidden:
            continue
        if c.adj_mines - len(flagged) == len(hidden):
            # All hidden are mines -> flag one
            nx, ny = hidden[0]
            return ('flag', (nx, ny))
        if len(flagged) == c.adj_mines:
            # All other hidden are safe -> reveal one
            nx, ny = hidden[0]
            return ('reveal', (nx, ny))
    return None
