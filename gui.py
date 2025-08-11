from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import time
import numpy as np

from minesweeper.engine import Minesweeper
from minesweeper.features import extract_cell_features
from minesweeper.agents.online_lr_agent import OnlineLogisticAgent
from minesweeper.agents.mlp_agent import MLPAgent


CELL_SIZE = 28
PADDING = 10
COLOR_MAP = {
    1: '#1976d2',
    2: '#388e3c',
    3: '#d32f2f',
    4: '#7b1fa2',
    5: '#5d4037',
    6: '#0097a7',
    7: '#455a64',
    8: '#9e9e9e',
}


class MinesweeperGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title('Mineswept GUI')

        # Controls
        control_frame = ttk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        ttk.Label(control_frame, text='Agent').grid(row=0, column=0, sticky='w')
        self.agent_var = tk.StringVar(value='mlp')
        ttk.Combobox(control_frame, textvariable=self.agent_var, values=['mlp', 'lr'], width=6, state='readonly').grid(row=0, column=1)

        ttk.Label(control_frame, text='Width').grid(row=0, column=2, sticky='w')
        self.width_var = tk.IntVar(value=16)
        ttk.Entry(control_frame, textvariable=self.width_var, width=4).grid(row=0, column=3)

        ttk.Label(control_frame, text='Height').grid(row=0, column=4, sticky='w')
        self.height_var = tk.IntVar(value=16)
        ttk.Entry(control_frame, textvariable=self.height_var, width=4).grid(row=0, column=5)

        ttk.Label(control_frame, text='Mines').grid(row=0, column=6, sticky='w')
        self.mines_var = tk.IntVar(value=40)
        ttk.Entry(control_frame, textvariable=self.mines_var, width=5).grid(row=0, column=7)

        ttk.Label(control_frame, text='Speed (ms)').grid(row=0, column=8, sticky='w')
        self.delay_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.delay_var, width=6).grid(row=0, column=9)

        self.btn_start = ttk.Button(control_frame, text='Start', command=self.start)
        self.btn_start.grid(row=0, column=10, padx=4)
        self.btn_stop = ttk.Button(control_frame, text='Stop', command=self.stop, state='disabled')
        self.btn_stop.grid(row=0, column=11, padx=4)
        self.btn_reset = ttk.Button(control_frame, text='Reset Agent', command=self.reset_agent)
        self.btn_reset.grid(row=0, column=12, padx=8)

        # Stats
        stats_frame = ttk.Frame(root)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=2)
        self.label_stats = ttk.Label(stats_frame, text='Games: 0 | Wins: 0 | Win%: 0.000')
        self.label_stats.pack(side=tk.LEFT)

        # Canvas for board
        self.canvas = tk.Canvas(root, bg='#eeeeee')
        self.canvas.pack(side=tk.TOP, padx=PADDING, pady=PADDING)

        self.env: Minesweeper | None = None
        self.agent: OnlineLogisticAgent | MLPAgent | None = None
        self.running = False
        self.after_id = None
        self.games = 0
        self.wins = 0

        # Initialize
        self._init_agent()
        self._new_game()

        self.root.protocol('WM_DELETE_WINDOW', self.on_close)

    def _init_agent(self):
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        mines = int(self.mines_var.get())
        dummy = Minesweeper(width, height, mines, seed=None)
        fx = extract_cell_features(dummy, 0, 0)
        latest = Path('checkpoints') / 'latest.npz'
        if latest.exists():
            # Try load matching type first
            if self.agent_var.get() == 'mlp':
                try:
                    self.agent = MLPAgent.load(latest)
                except Exception:
                    try:
                        ag = OnlineLogisticAgent.load(latest)
                        ag.adapt_feature_dim(fx.shape[0])
                        self.agent = ag
                    except Exception:
                        self.agent = MLPAgent(feature_dim=fx.shape[0], hidden_sizes=(64,64), lr=0.01, l2=1e-5)
            else:
                try:
                    ag = OnlineLogisticAgent.load(latest)
                    ag.adapt_feature_dim(fx.shape[0])
                    self.agent = ag
                except Exception:
                    try:
                        self.agent = MLPAgent.load(latest)
                    except Exception:
                        self.agent = OnlineLogisticAgent(feature_dim=fx.shape[0], lr=0.05, l2=1e-4)
        else:
            if self.agent_var.get() == 'mlp':
                self.agent = MLPAgent(feature_dim=fx.shape[0], hidden_sizes=(64,64), lr=0.01, l2=1e-5)
            else:
                self.agent = OnlineLogisticAgent(feature_dim=fx.shape[0], lr=0.05, l2=1e-4)

    def _new_game(self):
        width = int(self.width_var.get())
        height = int(self.height_var.get())
        mines = int(self.mines_var.get())
        self.env = Minesweeper(width, height, mines, seed=None)
        # First reveal center
        self.env.reveal(width // 2, height // 2)
        self._resize_canvas()
        self._render()

    def _resize_canvas(self):
        w = int(self.width_var.get()) * CELL_SIZE + PADDING * 2
        h = int(self.height_var.get()) * CELL_SIZE + PADDING * 2
        self.canvas.config(width=w, height=h)

    def _render(self):
        assert self.env is not None
        self.canvas.delete('all')
        for y in range(self.env.height):
            for x in range(self.env.width):
                px = PADDING + x * CELL_SIZE
                py = PADDING + y * CELL_SIZE
                c = self.env.grid[y][x]
                if c.is_flagged:
                    fill = '#ffc107'
                    self.canvas.create_rectangle(px, py, px+CELL_SIZE, py+CELL_SIZE, fill=fill, outline='#999')
                    self.canvas.create_text(px+CELL_SIZE/2, py+CELL_SIZE/2, text='🚩', font=('Arial', 12))
                elif not c.is_revealed:
                    self.canvas.create_rectangle(px, py, px+CELL_SIZE, py+CELL_SIZE, fill='#bdbdbd', outline='#9e9e9e')
                elif c.is_mine:
                    self.canvas.create_rectangle(px, py, px+CELL_SIZE, py+CELL_SIZE, fill='#ef5350', outline='#999')
                    self.canvas.create_text(px+CELL_SIZE/2, py+CELL_SIZE/2, text='💣', font=('Arial', 12))
                else:
                    self.canvas.create_rectangle(px, py, px+CELL_SIZE, py+CELL_SIZE, fill='#eeeeee', outline='#ccc')
                    if c.adj_mines > 0:
                        color = COLOR_MAP.get(c.adj_mines, '#212121')
                        self.canvas.create_text(px+CELL_SIZE/2, py+CELL_SIZE/2, text=str(c.adj_mines), fill=color, font=('Helvetica', 12, 'bold'))

    def start(self):
        if self.running:
            return
        try:
            # Reinit agent if agent type changed
            self._init_agent()
            self.running = True
            self.btn_start.config(state='disabled')
            self.btn_stop.config(state='normal')
            self._step_loop()
        except Exception as e:
            messagebox.showerror('Error', str(e))

    def stop(self):
        self.running = False
        self.btn_start.config(state='normal')
        self.btn_stop.config(state='disabled')
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None

    def reset_agent(self):
        self.stop()
        self._init_agent()
        self.games = 0
        self.wins = 0
        self._new_game()
        self._update_stats()

    def _update_stats(self):
        win_rate = (self.wins / self.games) if self.games > 0 else 0.0
        self.label_stats.config(text=f'Games: {self.games} | Wins: {self.wins} | Win%: {win_rate:.3f}')

    def _step_loop(self):
        if not self.running:
            return
        delay = max(0, int(self.delay_var.get()))
        try:
            done = self._step_once()
            self._render()
            if done:
                # Save latest and start a new game
                Path('checkpoints').mkdir(parents=True, exist_ok=True)
                (Path('checkpoints') / 'latest.npz').unlink(missing_ok=True)
                if isinstance(self.agent, (OnlineLogisticAgent, MLPAgent)):
                    self.agent.save(Path('checkpoints') / 'latest.npz')
                self.games += 1
                if self.env and self.env.win:
                    self.wins += 1
                self._update_stats()
                self._new_game()
            self.after_id = self.root.after(delay, self._step_loop)
        except Exception as e:
            self.stop()
            messagebox.showerror('Error', str(e))

    def _step_once(self) -> bool:
        assert self.env is not None and self.agent is not None
        if self.env.game_over:
            return True
        kind, (x, y) = self.agent.next_action(self.env)
        if kind == 'flag':
            self.env.flag(x, y)
            return False
        # reveal
        xfeat = extract_cell_features(self.env, x, y)
        was_mine = self.env.grid[y][x].is_mine
        self.env.reveal(x, y)
        # learn
        y_label = 1.0 if was_mine else 0.0
        self.agent.update(xfeat, y_label)
        return self.env.game_over

    def on_close(self):
        try:
            if isinstance(self.agent, (OnlineLogisticAgent, MLPAgent)):
                Path('checkpoints').mkdir(parents=True, exist_ok=True)
                self.agent.save(Path('checkpoints') / 'latest.npz')
        finally:
            self.root.destroy()


def main():
    root = tk.Tk()
    app = MinesweeperGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()


