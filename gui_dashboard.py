from __future__ import annotations
import pygame as pg
from pathlib import Path
import time
from dataclasses import dataclass

from minesweeper.engine import Minesweeper
from minesweeper.features import extract_cell_features
from minesweeper.agents.gnn_agent import GNNAgent


WHITE = (250, 250, 250)
BLACK = (20, 20, 20)
GRAY = (230, 230, 230)
ACCENT = (66, 133, 244)
RED = (234, 67, 53)
GREEN = (52, 168, 83)
AMBER = (251, 188, 5)

CELL_COLORS = {
    1: (25, 118, 210),
    2: (56, 142, 60),
    3: (211, 47, 47),
    4: (123, 31, 162),
    5: (93, 64, 55),
    6: (0, 151, 167),
    7: (69, 90, 100),
    8: (158, 158, 158),
}


@dataclass
class SessionStats:
    games: int = 0
    wins: int = 0
    reveals: int = 0
    flags: int = 0
    mine_hits: int = 0

    @property
    def win_rate(self) -> float:
        return (self.wins / self.games) if self.games > 0 else 0.0

    @property
    def mine_hit_rate(self) -> float:
        return (self.mine_hits / self.reveals) if self.reveals > 0 else 0.0


def load_agent(feature_dim: int):
    latest = Path('checkpoints') / 'latest.npz'
    if latest.exists():
        try:
            return GNNAgent.load(latest)
        except Exception:
            pass
    return GNNAgent(feature_dim=feature_dim, lr=0.001, l2=1e-6)


class DashboardGUI:
    def __init__(self):
        pg.init()
        self.font = pg.font.SysFont('Inter,Arial', 20)
        self.font_sm = pg.font.SysFont('Inter,Arial', 16)
        self.font_lg = pg.font.SysFont('Inter,Arial', 28, bold=True)
        self.screen = pg.display.set_mode((1000, 720))
        pg.display.set_caption('Mineswept — Dashboard')

        # Default settings
        self.agent_type = 'gnn'
        self.width, self.height, self.mines = 16, 16, 40
        self.cell = 28
        self.padding = 12
        self.delay = 0.03

        self.env = Minesweeper(self.width, self.height, self.mines, seed=None)
        fx = extract_cell_features(self.env, 0, 0)
        self.agent = load_agent(fx.shape[0])
        self.env.reveal(self.width // 2, self.height // 2)

        self.stats = SessionStats()
        self.mode = 'menu'  # 'menu' | 'play' | 'train'
        self.last_step = 0.0
        self.clock = pg.time.Clock()

    # ---------- Rendering ----------
    def draw_panel(self, rect: pg.Rect, title: str):
        pg.draw.rect(self.screen, WHITE, rect, border_radius=10)
        pg.draw.rect(self.screen, GRAY, rect, 1, border_radius=10)
        text = self.font.render(title, True, BLACK)
        self.screen.blit(text, (rect.x + 16, rect.y + 12))

    def draw_button(self, rect: pg.Rect, label: str, color=ACCENT) -> bool:
        mouse = pg.mouse.get_pos()
        pressed = pg.mouse.get_pressed()[0]
        hovered = rect.collidepoint(mouse)
        bg = tuple(min(255, int(c * (1.08 if hovered else 1.0))) for c in color)
        pg.draw.rect(self.screen, bg, rect, border_radius=8)
        txt = self.font.render(label, True, WHITE)
        self.screen.blit(txt, txt.get_rect(center=rect.center))
        return hovered and pressed

    def draw_board(self, origin: tuple[int, int]):
        ox, oy = origin
        for y in range(self.env.height):
            for x in range(self.env.width):
                px = ox + x * self.cell
                py = oy + y * self.cell
                c = self.env.grid[y][x]
                rect = pg.Rect(px, py, self.cell, self.cell)
                if c.is_flagged:
                    pg.draw.rect(self.screen, AMBER, rect, border_radius=4)
                    ftxt = self.font_sm.render('F', True, BLACK)
                    self.screen.blit(ftxt, ftxt.get_rect(center=rect.center))
                elif not c.is_revealed:
                    pg.draw.rect(self.screen, (210, 210, 210), rect, border_radius=4)
                elif c.is_mine:
                    pg.draw.rect(self.screen, RED, rect, border_radius=4)
                    xt = self.font_sm.render('X', True, WHITE)
                    self.screen.blit(xt, xt.get_rect(center=rect.center))
                else:
                    pg.draw.rect(self.screen, WHITE, rect, border_radius=4)
                    if c.adj_mines > 0:
                        color = CELL_COLORS.get(c.adj_mines, BLACK)
                        nt = self.font_sm.render(str(c.adj_mines), True, color)
                        self.screen.blit(nt, nt.get_rect(center=rect.center))
                pg.draw.rect(self.screen, (180, 180, 180), rect, 1, border_radius=4)

    def render_menu(self):
        self.screen.fill((245, 247, 250))
        title = self.font_lg.render('Mineswept', True, BLACK)
        self.screen.blit(title, (40, 30))

        # Left: mode selection
        left = pg.Rect(30, 90, 300, 250)
        self.draw_panel(left, 'Mode')
        if self.draw_button(pg.Rect(left.x + 20, left.y + 60, 260, 44), '▶ Train', ACCENT):
            self.mode = 'train'
        if self.draw_button(pg.Rect(left.x + 20, left.y + 120, 260, 44), '▶ Play', GREEN):
            self.mode = 'play'

        # Middle: agent and board
        mid = pg.Rect(350, 90, 300, 250)
        self.draw_panel(mid, 'Agent & Board')
        txt = self.font.render(f'Agent: {self.agent_type.upper()}  |  Size: {self.width}×{self.height}  Mines: {self.mines}', True, BLACK)
        self.screen.blit(txt, (mid.x + 16, mid.y + 60))
        hint = self.font_sm.render('Use keys: [A]gent, [B]oard, [M]ines to cycle', True, (90,90,90))
        self.screen.blit(hint, (mid.x + 16, mid.y + 100))

        # Right: preview
        right = pg.Rect(670, 90, 300, 250)
        self.draw_panel(right, 'Preview')
        self.draw_board((right.x + 16, right.y + 50))

        # Footer
        footer = self.font_sm.render('Hotkeys: A=Agent, B=Board size, M=Mines, Enter=Start', True, (120,120,120))
        self.screen.blit(footer, (30, 360))

    def render_dashboard(self):
        self.screen.fill((245, 247, 250))
        title = self.font_lg.render(f'Mode: {self.mode.title()}', True, BLACK)
        self.screen.blit(title, (30, 20))

        # Left: Board
        board_panel = pg.Rect(30, 70, 560, 560)
        self.draw_panel(board_panel, 'Board')
        self.draw_board((board_panel.x + 16, board_panel.y + 50))

        # Right: Stats
        stats_panel = pg.Rect(610, 70, 360, 400)
        self.draw_panel(stats_panel, 'Stats')
        s = self.stats
        lines = [
            f'Games: {s.games}',
            f'Wins: {s.wins}',
            f'Win%: {s.win_rate:.3f}',
            f'Reveals: {s.reveals}',
            f'Flags: {s.flags}',
            f'Mine hits: {s.mine_hits}',
            f'Mine hit rate: {s.mine_hit_rate:.3f}',
        ]
        for i, line in enumerate(lines):
            t = self.font.render(line, True, BLACK)
            self.screen.blit(t, (stats_panel.x + 16, stats_panel.y + 50 + i * 34))

        # Right bottom: Controls
        ctrl_panel = pg.Rect(610, 490, 360, 140)
        self.draw_panel(ctrl_panel, 'Controls')
        if self.draw_button(pg.Rect(ctrl_panel.x + 16, ctrl_panel.y + 50, 150, 40), '⏸ Pause', AMBER):
            self.mode = 'menu'
        if self.draw_button(pg.Rect(ctrl_panel.x + 190, ctrl_panel.y + 50, 150, 40), '⏭ Next', ACCENT):
            self._new_game()

    # ---------- Logic ----------
    def _new_game(self):
        self.env = Minesweeper(self.width, self.height, self.mines, seed=None)
        self.env.reveal(self.width // 2, self.height // 2)

    def _reset_agent(self):
        fx = extract_cell_features(self.env, 0, 0)
        self.agent = load_agent(fx.shape[0])
        self.stats = SessionStats()

    def handle_menu_keys(self, event):
        if event.key == pg.K_RETURN:
            self.mode = 'train'
        elif event.key == pg.K_a:
            self._reset_agent()
        elif event.key == pg.K_b:
            # cycle board sizes
            sizes = [(9,9,10), (16,16,40), (30,16,99)]
            idx = next((i for i,t in enumerate(sizes) if (self.width,self.height,self.mines)==t), 0)
            self.width, self.height, self.mines = sizes[(idx+1)%len(sizes)]
            self._new_game()
        elif event.key == pg.K_m:
            # adjust mines
            self.mines = max(5, min(self.width*self.height-1, self.mines + 5))
            self._new_game()

    def step_once(self):
        if self.env.game_over:
            self.stats.games += 1
            if self.env.win:
                self.stats.wins += 1
            # Save progress
            Path('checkpoints').mkdir(parents=True, exist_ok=True)
            self.agent.save(Path('checkpoints') / 'latest.npz')
            self._new_game()
            return
        kind, (x, y) = self.agent.next_action(self.env)
        if kind == 'flag':
            self.env.flag(x, y)
            self.stats.flags += 1
            return
        xfeat = extract_cell_features(self.env, x, y)
        was_mine = self.env.grid[y][x].is_mine
        self.env.reveal(x, y)
        self.agent.update(xfeat, 1.0 if was_mine else 0.0)
        self.stats.reveals += 1
        if was_mine:
            self.stats.mine_hits += 1

    def run(self):
        running = True
        while running:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                elif event.type == pg.KEYDOWN:
                    if self.mode == 'menu':
                        self.handle_menu_keys(event)
                    else:
                        if event.key == pg.K_SPACE:
                            self.mode = 'menu'
                        elif event.key == pg.K_r:
                            self._reset_agent()
                        elif event.key == pg.K_n:
                            self._new_game()

            if self.mode in ('play', 'train'):
                now = time.time()
                if now - self.last_step >= self.delay:
                    self.last_step = now
                    # In both modes, we show the board and let the agent play; in future, play could disable learning
                    self.step_once()
                self.render_dashboard()
            else:
                self.render_menu()

            pg.display.flip()
            self.clock.tick(60)
        pg.quit()


def main():
    gui = DashboardGUI()
    gui.run()


if __name__ == '__main__':
    main()


