
from __future__ import annotations
from typing import Tuple, List
import numpy as np

# Features for a candidate hidden cell (x, y):
# - counts of neighbor states: revealed numbers 1..8, zeroes, flags, hidden
# - mean and min of clue deficits around it
# - bias term

def extract_cell_features(board, x: int, y: int) -> np.ndarray:
    counts = np.zeros(10, dtype=float)
    clue_deficits: List[float] = []
    for nx, ny in board.neighbors(x, y):
        c = board.grid[ny][nx]
        if c.is_revealed and not c.is_mine:
            if c.adj_mines == 0:
                counts[0] += 1
            else:
                idx = min(c.adj_mines, 8)
                counts[idx] += 1
            # deficit = number - flags around that cell
            flags = sum(1 for ax, ay in board.neighbors(nx, ny) if board.grid[ay][ax].is_flagged)
            deficit = c.adj_mines - flags
            clue_deficits.append(deficit)
        elif c.is_flagged:
            counts[9] += 1
        else:
            # hidden neighbor
            pass
    if not clue_deficits:
        clue_deficits = [0.0]
    features = np.array([
        *counts.tolist(),
        float(len(list(board.hidden_cells()))),
        float(len(list(board.revealed_number_cells()))),
        float(np.mean(clue_deficits)),
        float(np.min(clue_deficits)),
        1.0,  # bias
    ], dtype=float)
    return features

