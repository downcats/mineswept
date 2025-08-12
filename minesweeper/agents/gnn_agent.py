from __future__ import annotations
from typing import Tuple, Optional, Union, Dict, List
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..features import extract_cell_features
from .rule_agent import pick_move as rule_pick


class GNNLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_attr_dim: int):
        super().__init__()
        # c->h message
        self.c2h_gate = nn.Sequential(
            nn.Linear(hidden_dim + edge_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.c2h_msg = nn.Linear(hidden_dim, hidden_dim)
        self.h_upd = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.h_norm = nn.LayerNorm(hidden_dim)

        # h->c message
        self.h2c_gate = nn.Sequential(
            nn.Linear(hidden_dim + edge_attr_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.h2c_msg = nn.Linear(hidden_dim, hidden_dim)
        self.c_upd = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.c_norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_hidden: torch.Tensor, h_clue: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor):
        if h_hidden.numel() == 0:
            return h_hidden, h_clue
        # c -> h
        if edges.numel() > 0 and h_clue.numel() > 0:
            c_idx = edges[:, 0].long()
            h_idx = edges[:, 1].long()
            src = h_clue[c_idx]
            cat = torch.cat([src, edge_attr], dim=-1)
            gate = self.c2h_gate(cat)
            msgs = self.c2h_msg(src) * gate
            agg_h = torch.zeros_like(h_hidden)
            agg_h.index_add_(0, h_idx, msgs)
        else:
            agg_h = torch.zeros_like(h_hidden)
        h_hidden = self.h_norm(h_hidden + self.h_upd(torch.cat([h_hidden, agg_h], dim=-1)))

        # h -> c
        if edges.numel() > 0 and h_hidden.numel() > 0 and h_clue.numel() > 0:
            # reverse mapping: use same edge_attr
            c_idx = edges[:, 0].long()
            h_idx = edges[:, 1].long()
            src = h_hidden[h_idx]
            cat = torch.cat([src, edge_attr], dim=-1)
            gate = self.h2c_gate(cat)
            msgs = self.h2c_msg(src) * gate
            agg_c = torch.zeros_like(h_clue)
            agg_c.index_add_(0, c_idx, msgs)
        else:
            agg_c = torch.zeros_like(h_clue)
        h_clue = self.c_norm(h_clue + self.c_upd(torch.cat([h_clue, agg_c], dim=-1)))

        return h_hidden, h_clue


class StackedGNN(nn.Module):
    def __init__(self, hidden_feat_dim: int, clue_feat_dim: int, hidden_dim: int = 128, num_layers: int = 3, edge_attr_dim: int = 3):
        super().__init__()
        self.hidden_encoder = nn.Sequential(
            nn.Linear(hidden_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.clue_encoder = nn.Sequential(
            nn.Linear(clue_feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList([GNNLayer(hidden_dim, edge_attr_dim) for _ in range(num_layers)])
        self.out_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_x: torch.Tensor, clue_x: torch.Tensor, edges: torch.Tensor, edge_attr: torch.Tensor):
        if hidden_x.numel() == 0:
            return torch.empty((0,), dtype=torch.float32, device=hidden_x.device)
        h_hidden = self.hidden_encoder(hidden_x)
        h_clue = self.clue_encoder(clue_x) if clue_x.numel() > 0 else torch.zeros((0, h_hidden.size(-1)), device=h_hidden.device)
        for layer in self.layers:
            h_hidden, h_clue = layer(h_hidden, h_clue, edges, edge_attr)
        logits = self.out_head(h_hidden).squeeze(-1)
        return logits


class GNNAgent:
    def __init__(self, feature_dim: int, lr: float = 0.001, l2: float = 1e-6, seed: int = 0):
        # We'll use the existing extract_cell_features for hidden nodes as input
        # For clue nodes, we build lightweight features
        torch.manual_seed(int(max(0, seed)))
        self.rng = np.random.default_rng(int(max(0, seed)))
        self.hidden_feat_dim = int(feature_dim)
        self.clue_feat_dim = 8  # [number, flagged_cnt, hidden_cnt, deficit, est_density, fx, fy, 1.0]
        self.model = StackedGNN(self.hidden_feat_dim, self.clue_feat_dim, hidden_dim=128, num_layers=3)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
        self.lr = float(lr)
        self.l2 = float(l2)

        # Cache from last next_action for predict_proba/update
        self._cache: Dict[str, object] = {}

    def _build_graph(self, board):
        # Hidden node features and index map
        hidden_coords: List[Tuple[int, int]] = list(board.hidden_cells())
        num_hidden = len(hidden_coords)
        if num_hidden == 0:
            return None
        hidden_feats = [extract_cell_features(board, x, y) for (x, y) in hidden_coords]
        hidden_x = torch.tensor(np.stack(hidden_feats, axis=0), dtype=torch.float32)

        # Clue nodes
        clue_coords: List[Tuple[int, int]] = list(board.revealed_number_cells())
        clue_feats: List[List[float]] = []
        edges: List[Tuple[int, int]] = []  # (clue_idx, hidden_idx)
        edge_attr: List[List[float]] = []

        # Global density for features
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
        est_density = remaining_mines / float(remaining_cells)

        # Build maps
        hidden_index: Dict[Tuple[int, int], int] = {xy: i for i, xy in enumerate(hidden_coords)}

        for (cx, cy) in clue_coords:
            c = board.grid[cy][cx]
            flags = 0
            hidd = 0
            nbrs = board.neighbors(cx, cy)
            for nx, ny in nbrs:
                cc = board.grid[ny][nx]
                if cc.is_flagged:
                    flags += 1
                if not cc.is_revealed and not cc.is_flagged:
                    hidd += 1
            deficit = c.adj_mines - flags
            fx = cx / max(1, (board.width - 1))
            fy = cy / max(1, (board.height - 1))
            clue_feats.append([float(c.adj_mines), float(flags), float(hidd), float(deficit), float(est_density), fx, fy, 1.0])
            ci = len(clue_feats) - 1
            if hidd > 0:
                inv_h = 1.0 / float(hidd)
            else:
                inv_h = 0.0
            for nx, ny in nbrs:
                if (nx, ny) in hidden_index:
                    hi = hidden_index[(nx, ny)]
                    edges.append((ci, hi))
                    edge_attr.append([float(deficit), float(hidd), float(inv_h)])

        if not clue_feats:
            clue_x = torch.zeros((0, self.clue_feat_dim), dtype=torch.float32)
            edges_t = torch.zeros((0, 2), dtype=torch.long)
            edge_attr_t = torch.zeros((0, 3), dtype=torch.float32)
        else:
            clue_x = torch.tensor(np.array(clue_feats, dtype=np.float32))
            edges_t = torch.tensor(np.array(edges, dtype=np.int64)) if edges else torch.zeros((0, 2), dtype=torch.long)
            edge_attr_t = torch.tensor(np.array(edge_attr, dtype=np.float32)) if edge_attr else torch.zeros((0, 3), dtype=torch.float32)

        return {
            'hidden_coords': hidden_coords,
            'hidden_x': hidden_x,
            'clue_x': clue_x,
            'edges': edges_t,
            'edge_attr': edge_attr_t,
        }

    def _predict_all(self, board) -> Dict[Tuple[int, int], float]:
        self.model.eval()
        g = self._build_graph(board)
        if g is None:
            return {}
        with torch.no_grad():
            logits = self.model(g['hidden_x'], g['clue_x'], g['edges'], g['edge_attr'])
            probs = torch.sigmoid(logits).cpu().numpy().astype(float)
        p_map: Dict[Tuple[int, int], float] = {}
        for (xy, p) in zip(g['hidden_coords'], probs.tolist()):
            p_map[xy] = float(p)
        # Cache for update
        self._cache = {'graph': g, 'p_map': p_map}
        return p_map

    def predict_proba(self, x: np.ndarray) -> float:
        # For compatibility with existing pipeline, return the last selected cell's probability if available
        if 'last_selected_p' in self._cache:
            return float(self._cache['last_selected_p'])  # type: ignore
        return 0.5

    def update(self, x: np.ndarray, y: float) -> None:
        # Use cached graph and last selected index to train
        if 'graph' not in self._cache or 'last_selected_idx' not in self._cache:
            return
        g = self._cache['graph']  # type: ignore
        idx = int(self._cache['last_selected_idx'])  # type: ignore
        self.model.train()
        logits = self.model(g['hidden_x'], g['clue_x'], g['edges'], g['edge_attr'])
        if logits.numel() == 0:
            return
        logit = logits[idx:idx+1]
        target = torch.tensor([y], dtype=torch.float32)
        loss = F.binary_cross_entropy_with_logits(logit, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

    def next_action(self, board) -> Tuple[str, Tuple[int, int]]:
        action = rule_pick(board)
        if action is not None:
            return action
        p_map = self._predict_all(board)
        if not p_map:
            # Fallback: pick any hidden
            best_xy = list(board.hidden_cells())[0]
            self._cache['last_selected_idx'] = 0
            self._cache['last_selected_p'] = 0.5
            return ('reveal', best_xy)
        # Choose min mine probability
        best_xy = min(p_map.items(), key=lambda kv: kv[1])[0]
        # Record selection for update/predict_proba
        g = self._cache['graph']  # type: ignore
        idx_map = {xy: i for i, xy in enumerate(g['hidden_coords'])}
        self._cache['last_selected_idx'] = idx_map[best_xy]
        self._cache['last_selected_p'] = p_map[best_xy]
        return ('reveal', best_xy)

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {
            'agent': np.array('gnn'),
            'lr': np.array(self.lr),
            'l2': np.array(self.l2),
            'hidden_feat_dim': np.array(self.hidden_feat_dim),
            'clue_feat_dim': np.array(self.clue_feat_dim),
        }
        # Save parameters by name
        for name, param in self.model.state_dict().items():
            arrays[f'param::{name}'] = param.detach().cpu().numpy()
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'GNNAgent':
        data = np.load(Path(path), allow_pickle=False)
        if 'agent' not in data or str(data['agent']) not in ('gnn', 'GNNAgent'):
            raise ValueError('Checkpoint is not for GNNAgent')
        hidden_feat_dim = int(data['hidden_feat_dim'])
        clue_feat_dim = int(data['clue_feat_dim'])
        lr = float(data['lr']) if 'lr' in data else 1e-3
        l2 = float(data['l2']) if 'l2' in data else 1e-6
        agent = cls(feature_dim=hidden_feat_dim, lr=lr, l2=l2, seed=0)
        agent.clue_feat_dim = clue_feat_dim
        # Load parameters
        state = {}
        for key in data.files:
            if key.startswith('param::'):
                name = key.split('param::', 1)[1]
                state[name] = torch.tensor(data[key])
        agent.model.load_state_dict(state, strict=False)
        return agent

    def weight_stats(self) -> Tuple[float, float]:
        total_norm = 0.0
        last_bias = 0.0
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                total_norm += float(torch.norm(p).cpu())
                if name.endswith('out_head.bias'):
                    if p.numel() > 0:
                        last_bias = float(p.view(-1)[0].cpu())
        return total_norm, last_bias


