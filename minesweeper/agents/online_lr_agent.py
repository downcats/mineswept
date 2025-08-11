
from __future__ import annotations
from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
from ..features import extract_cell_features
from .rule_agent import pick_move as rule_pick

class OnlineLogisticAgent:
    def __init__(self, feature_dim: int, lr: float = 0.05, l2: float = 1e-4, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.w = self.rng.normal(scale=0.01, size=(feature_dim,))
        self.lr = lr
        self.l2 = l2

    def predict_proba(self, x: np.ndarray) -> float:
        z = float(x @ self.w)
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, x: np.ndarray, y: float) -> None:
        p = self.predict_proba(x)
        grad = (p - y) * x + self.l2 * self.w
        self.w -= self.lr * grad

    def next_action(self, board) -> Tuple[str, Tuple[int, int]]:
        # Rule-based deterministic action first
        action = rule_pick(board)
        if action is not None:
            return action
        # Otherwise, choose a reveal with highest predicted safety
        best_score = -1e9
        best_xy: Optional[Tuple[int, int]] = None
        for x, y in board.hidden_cells():
            xfeat = extract_cell_features(board, x, y)
            p_mine = self.predict_proba(xfeat)
            safety = 1.0 - p_mine
            if safety > best_score:
                best_score = safety
                best_xy = (x, y)
        if best_xy is None:
            cells = list(board.hidden_cells())
            best_xy = cells[0]
        return ('reveal', best_xy)

    # Backward-compatible helper
    def choose_cell(self, board) -> Tuple[int, int]:
        kind, xy = self.next_action(board)
        return xy

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, w=self.w, lr=np.array(self.lr), l2=np.array(self.l2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'OnlineLogisticAgent':
        data = np.load(Path(path), allow_pickle=False)
        w = data['w']
        lr = float(data['lr']) if 'lr' in data else 0.05
        l2 = float(data['l2']) if 'l2' in data else 1e-4
        agent = cls(feature_dim=int(w.shape[0]), lr=lr, l2=l2)
        agent.w = w.astype(float)
        return agent

    def weight_stats(self) -> Tuple[float, float]:
        w_norm = float(np.linalg.norm(self.w))
        bias_weight = float(self.w[-1]) if self.w.size > 0 else 0.0
        return w_norm, bias_weight

