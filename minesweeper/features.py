
from __future__ import annotations
from typing import List
import numpy as np

# Richer features for a candidate hidden cell (x, y):
# - Local neighborhood counts: revealed numbers 0..8, flags, hidden
# - Local clue stats: sum/mean of numbers, deficits (num - flags), hidden-around-clue stats
# - Deterministic-rule signals: count of sure-mine/sure-safe clues around
# - Global state: remaining cells, flags, est mine density
# - Positional: normalized x,y and distance-to-edge, frontier indicator
# - Bias term

def _count_board_state(board):
    total_hidden = 0
    total_flags = 0
    for row in board.grid:
        for c in row:
            if not c.is_revealed and not c.is_flagged:
                total_hidden += 1
            if c.is_flagged:
                total_flags += 1
    remaining_mines = max(0, board.num_mines - total_flags)
    remaining_cells = max(1, total_hidden)
    density = remaining_mines / float(remaining_cells)
    return total_hidden, total_flags, density


def extract_cell_features(board, x: int, y: int) -> np.ndarray:
    # Local neighbor summaries
    numbers_count = np.zeros(9, dtype=float)  # counts for numbers 0..8
    n_flags = 0
    n_hidden = 0
    clue_deficits: List[float] = []
    hidden_around_clue: List[int] = []
    sure_mine_clues = 0
    sure_safe_clues = 0

    frontier_indicator = 0.0
    for nx, ny in board.neighbors(x, y):
        c = board.grid[ny][nx]
        if c.is_revealed and not c.is_mine:
            frontier_indicator = 1.0
            idx = int(np.clip(c.adj_mines, 0, 8))
            numbers_count[idx] += 1
            flags = sum(1 for ax, ay in board.neighbors(nx, ny) if board.grid[ay][ax].is_flagged)
            hidden = sum(1 for ax, ay in board.neighbors(nx, ny) if not board.grid[ay][ax].is_revealed and not board.grid[ay][ax].is_flagged)
            deficit = c.adj_mines - flags
            clue_deficits.append(deficit)
            hidden_around_clue.append(hidden)
            if hidden > 0 and deficit == hidden:
                sure_mine_clues += 1
            if flags == c.adj_mines and hidden > 0:
                sure_safe_clues += 1
        elif c.is_flagged:
            n_flags += 1
        else:
            n_hidden += 1

    if not clue_deficits:
        clue_deficits = [0.0]
    if not hidden_around_clue:
        hidden_around_clue = [0]

    total_hidden, total_flags, est_density = _count_board_state(board)
    remaining_cells = total_hidden

    # Positional features
    fx = x / max(1, (board.width - 1))
    fy = y / max(1, (board.height - 1))
    dist_edge = min(x, board.width - 1 - x, y, board.height - 1 - y) / max(1, max(board.width, board.height) - 1)

    features = np.array([
        # Local counts
        *numbers_count.tolist(),
        float(n_flags),
        float(n_hidden),
        # Clue stats around
        float(np.sum(clue_deficits)),
        float(np.mean(clue_deficits)),
        float(np.min(clue_deficits)),
        float(np.max(clue_deficits)),
        float(np.mean(hidden_around_clue)),
        float(np.min(hidden_around_clue)),
        float(np.max(hidden_around_clue)),
        float(sure_mine_clues),
        float(sure_safe_clues),
        # Global
        float(remaining_cells),
        float(total_flags),
        float(est_density),
        # Positional
        fx,
        fy,
        float(dist_edge),
        frontier_indicator,
        # Bias
        1.0,
    ], dtype=float)
    return features

