from __future__ import annotations
from typing import Tuple, Optional, Union
from pathlib import Path
import numpy as np
from ..features import extract_cell_features
from .rule_agent import pick_move as rule_pick


class MLPAgent:
    def __init__(self, feature_dim: int, hidden_sizes=(64, 64), lr: float = 0.01, l2: float = 1e-5, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.layer_sizes = [feature_dim, *hidden_sizes, 1]
        self.params = self._init_params(self.layer_sizes)
        self.lr = lr
        self.l2 = l2

    def _init_params(self, sizes):
        params = []
        for in_dim, out_dim in zip(sizes[:-1], sizes[1:]):
            limit = np.sqrt(6 / (in_dim + out_dim))
            W = self.rng.uniform(-limit, limit, size=(out_dim, in_dim))
            b = np.zeros((out_dim,))
            params.append((W, b))
        return params

    def _forward(self, x: np.ndarray):
        a = np.asarray(x, dtype=float).ravel()
        a_list = [a]
        z_list = []
        for i, (W, b) in enumerate(self.params):
            z = W @ a + b
            z_list.append(z)
            if i < len(self.params) - 1:
                a = np.maximum(0.0, z)
            else:
                # Stable sigmoid
                a = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
            a_list.append(a)
        return a_list, z_list

    def _backward(self, a_list, z_list, y: float):
        grads = [None] * len(self.params)
        # Output layer
        y_hat = float(a_list[-1][0])
        dz = np.array([y_hat - y], dtype=float)  # shape (1,)
        a_prev = np.asarray(a_list[-2], dtype=float).ravel()
        W_last, _ = self.params[-1]
        dW = dz[:, None] @ a_prev[None, :] + self.l2 * W_last
        db = dz.copy()
        grads[-1] = (dW, db)
        da = (W_last.T @ dz).ravel()
        # Hidden layers
        for l in range(len(self.params) - 2, -1, -1):
            W, _ = self.params[l]
            z = np.asarray(z_list[l], dtype=float).ravel()
            a_prev = np.asarray(a_list[l], dtype=float).ravel()
            relu_mask = (z > 0).astype(float)
            dz = (da * relu_mask).ravel()
            dW = dz[:, None] @ a_prev[None, :] + self.l2 * W
            db = dz
            grads[l] = (dW, db)
            if l > 0:
                da = (W.T @ dz).ravel()
        return grads

    def _apply_grads(self, grads):
        for i, (dW, db) in enumerate(grads):
            W, b = self.params[i]
            self.params[i] = (W - self.lr * dW, b - self.lr * db)

    def predict_proba(self, x: np.ndarray) -> float:
        a_list, _ = self._forward(x)
        return float(a_list[-1][0])

    def update(self, x: np.ndarray, y: float) -> None:
        a_list, z_list = self._forward(x)
        grads = self._backward(a_list, z_list, y)
        self._apply_grads(grads)

    def next_action(self, board) -> Tuple[str, Tuple[int, int]]:
        action = rule_pick(board)
        if action is not None:
            return action
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
            best_xy = list(board.hidden_cells())[0]
        return ('reveal', best_xy)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for i, (W, b) in enumerate(self.params):
            arrays[f'W{i}'] = W
            arrays[f'b{i}'] = b
        arrays['agent'] = np.array('mlp')
        arrays['lr'] = np.array(self.lr)
        arrays['l2'] = np.array(self.l2)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MLPAgent':
        data = np.load(Path(path), allow_pickle=False)
        if 'agent' not in data or str(data['agent']) not in ('mlp', 'MLPAgent'):
            raise ValueError('Checkpoint is not for MLPAgent')
        # Infer layer sizes from arrays
        i = 0
        params = []
        while f'W{i}' in data:
            W = data[f'W{i}']
            b = data[f'b{i}']
            params.append((W.astype(float), b.astype(float)))
            i += 1
        feature_dim = params[0][0].shape[1]
        lr = float(data['lr']) if 'lr' in data else 0.01
        l2 = float(data['l2']) if 'l2' in data else 1e-5
        agent = cls(feature_dim=feature_dim, hidden_sizes=tuple(p[0].shape[0] for p in params[:-1]), lr=lr, l2=l2)
        agent.params = params
        return agent

    def weight_stats(self) -> Tuple[float, float]:
        total_norm = 0.0
        for W, b in self.params:
            total_norm += float(np.linalg.norm(W)) + float(np.linalg.norm(b))
        last_bias = float(self.params[-1][1][0])
        return total_norm, last_bias


