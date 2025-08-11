
from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

Coordinate = Tuple[int, int]

@dataclass
class Cell:
    is_mine: bool = False
    is_revealed: bool = False
    is_flagged: bool = False
    adj_mines: int = 0

class Minesweeper:
    def __init__(self, width: int, height: int, num_mines: int, seed: Optional[int] = None):
        assert 0 < num_mines < width * height
        self.width = width
        self.height = height
        self.num_mines = num_mines
        self.rng = random.Random(int(seed)) if seed is not None else random.Random()
        self.grid: List[List[Cell]] = [[Cell() for _ in range(width)] for _ in range(height)]
        self._mines_placed = False
        self.revealed_count = 0
        self.game_over = False
        self.win = False

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def neighbors(self, x: int, y: int) -> List[Coordinate]:
        coords = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    coords.append((nx, ny))
        return coords

    def _place_mines(self, first_click: Coordinate):
        all_cells = [(x, y) for y in range(self.height) for x in range(self.width)]
        # Ensure first click is safe
        all_cells.remove(first_click)
        mines = set(self.rng.sample(all_cells, self.num_mines))
        for (x, y) in mines:
            self.grid[y][x].is_mine = True
        # Compute adjacencies
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x].is_mine:
                    continue
                self.grid[y][x].adj_mines = sum(1 for (nx, ny) in self.neighbors(x, y) if self.grid[ny][nx].is_mine)
        self._mines_placed = True

    def reveal(self, x: int, y: int) -> None:
        if self.game_over:
            return
        if not self.in_bounds(x, y):
            return
        c = self.grid[y][x]
        if c.is_flagged or c.is_revealed:
            return
        if not self._mines_placed:
            self._place_mines((x, y))
        c.is_revealed = True
        self.revealed_count += 1
        if c.is_mine:
            self.game_over = True
            self.win = False
            return
        if c.adj_mines == 0:
            for (nx, ny) in self.neighbors(x, y):
                if not self.grid[ny][nx].is_revealed:
                    self.reveal(nx, ny)
        # Check win
        if self.revealed_count == self.width * self.height - self.num_mines:
            self.game_over = True
            self.win = True

    def flag(self, x: int, y: int) -> None:
        if self.game_over:
            return
        if not self.in_bounds(x, y):
            return
        c = self.grid[y][x]
        if c.is_revealed:
            return
        c.is_flagged = not c.is_flagged

    def hidden_cells(self) -> Iterable[Coordinate]:
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x].is_revealed and not self.grid[y][x].is_flagged:
                    yield (x, y)

    def revealed_number_cells(self) -> Iterable[Coordinate]:
        for y in range(self.height):
            for x in range(self.width):
                c = self.grid[y][x]
                if c.is_revealed and not c.is_mine and c.adj_mines > 0:
                    yield (x, y)

    def render_ascii(self) -> str:
        rows = []
        for y in range(self.height):
            row = []
            for x in range(self.width):
                c = self.grid[y][x]
                if c.is_flagged:
                    row.append('F')
                elif not c.is_revealed:
                    row.append('#')
                elif c.is_mine:
                    row.append('*')
                elif c.adj_mines == 0:
                    row.append('.')
                else:
                    row.append(str(c.adj_mines))
            rows.append(' '.join(row))
        return '\n'.join(rows)

