
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from minesweeper.engine import Minesweeper
from minesweeper.features import extract_cell_features
from minesweeper.agents.online_lr_agent import OnlineLogisticAgent
from minesweeper.agents.mlp_agent import MLPAgent

FEATURE_DIM = None

def play_episode(agent: OnlineLogisticAgent, width: int, height: int, mines: int, seed: int | None = None, learn: bool = True):
    env = Minesweeper(width, height, mines, seed=seed)
    first = (width // 2, height // 2)
    env.reveal(*first)
    reveal_actions = 0
    flag_actions = 0
    prob_sum = 0.0
    prob_count = 0
    prob_sum_safe = 0.0
    prob_count_safe = 0
    prob_sum_mine = 0.0
    prob_count_mine = 0
    reveal_mines = 0
    # Online loop
    while not env.game_over:
        kind, (x, y) = agent.next_action(env)
        if kind == 'flag':
            env.flag(x, y)
            flag_actions += 1
            continue
        # reveal
        xfeat = extract_cell_features(env, x, y)
        p_mine = agent.predict_proba(xfeat)
        prob_sum += float(p_mine)
        prob_count += 1
        was_mine = env.grid[y][x].is_mine
        env.reveal(x, y)
        reveal_actions += 1
        if learn:
            y_label = 1.0 if was_mine else 0.0
            agent.update(xfeat, y_label)
        if was_mine:
            reveal_mines += 1
            prob_sum_mine += float(p_mine)
            prob_count_mine += 1
        else:
            prob_sum_safe += float(p_mine)
            prob_count_safe += 1
    metrics = {
        'reveal_actions': reveal_actions,
        'flag_actions': flag_actions,
        'total_actions': reveal_actions + flag_actions,
        'final_revealed': env.revealed_count,
        'avg_pred_p_mine': (prob_sum / prob_count) if prob_count > 0 else 0.0,
        'avg_pred_p_mine_safe': (prob_sum_safe / prob_count_safe) if prob_count_safe > 0 else None,
        'avg_pred_p_mine_mine': (prob_sum_mine / prob_count_mine) if prob_count_mine > 0 else None,
        'reveal_mines': reveal_mines,
        'reveal_mine_rate': (reveal_mines / reveal_actions) if reveal_actions > 0 else None,
    }
    return env.win, metrics


def append_csv_row(csv_path: Path, row: dict, header_order: list[str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open('a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header_order)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=16)
    parser.add_argument('--height', type=int, default=16)
    parser.add_argument('--mines', type=int, default=40)
    parser.add_argument('--episodes', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=-1, help='Base RNG seed; <0 uses OS entropy (random every run)')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--agent', type=str, default='mlp', choices=['lr','mlp'])
    parser.add_argument('--eval_every', type=int, default=200)
    parser.add_argument('--log_csv', type=str, default='logs/train_log.csv')
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt_every', type=int, default=200)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--auto_resume_latest', action='store_true', default=True,
                        help='Auto-load checkpoints/latest.npz if present and --resume not set')
    args = parser.parse_args()

    # Infer feature dimension from a dummy board
    dummy = Minesweeper(args.width, args.height, args.mines, seed=args.seed)
    fx = extract_cell_features(dummy, 0, 0)
    latest_path = Path(args.ckpt_dir) / 'latest.npz'
    def init_agent():
        if args.agent == 'lr':
            return OnlineLogisticAgent(feature_dim=fx.shape[0], lr=args.lr, l2=args.l2, seed=args.seed)
        else:
            return MLPAgent(feature_dim=fx.shape[0], hidden_sizes=(64,64), lr=0.01, l2=1e-5, seed=args.seed)
    def load_agent(path: Path):
        # Try MLP first, then LR
        try:
            return MLPAgent.load(path)
        except Exception:
            ag = OnlineLogisticAgent.load(path)
            # If features changed, adapt LR weights
            if hasattr(ag, 'adapt_feature_dim'):
                ag.adapt_feature_dim(fx.shape[0])
            return ag
    if args.resume:
        agent = load_agent(Path(args.resume))
        print(f"[train] Loaded checkpoint: {args.resume}")
    elif args.auto_resume_latest and latest_path.exists():
        # Respect requested agent type: only auto-resume if checkpoint matches, else init new
        try:
            ag = load_agent(latest_path)
            if (args.agent == 'mlp' and isinstance(ag, MLPAgent)) or (args.agent == 'lr' and isinstance(ag, OnlineLogisticAgent)):
                agent = ag
                print(f"[train] Auto-resumed from {latest_path}")
            else:
                agent = init_agent()
                print(f"[train] latest.npz type mismatch for --agent={args.agent}; initialized new agent")
        except Exception:
            agent = init_agent()
            print(f"[train] Failed to load latest.npz; initialized new agent")
    else:
        agent = init_agent()
        print("[train] Initialized new agent")

    rng = np.random.default_rng(None if args.seed < 0 else args.seed)
    wins_recent = 0
    header = [
        'phase','episode','seed','width','height','mines','lr','l2',
        'win','reveal_actions','flag_actions','total_actions','final_revealed','avg_pred_p_mine','avg_pred_p_mine_safe','avg_pred_p_mine_mine','reveal_mines','reveal_mine_rate','w_norm','w_bias',
        'recent_train_win_rate','eval_win_rate'
    ]
    csv_path = Path(args.log_csv)

    try:
        for ep in range(1, args.episodes + 1):
            ep_seed = int(rng.integers(1_000_000_000))
            win, m = play_episode(agent, args.width, args.height, args.mines, seed=ep_seed, learn=True)
            # Always update latest checkpoint for cumulative training
            agent.save(latest_path)
            wins_recent += 1 if win else 0
            if ep % args.log_every == 0:
                w_norm, w_bias = agent.weight_stats()
                row = {
                    'phase': 'train', 'episode': ep, 'seed': ep_seed,
                    'width': args.width, 'height': args.height, 'mines': args.mines,
                    'lr': args.lr, 'l2': args.l2,
                    'win': int(win),
                    **m,
                    'w_norm': w_norm,
                    'w_bias': w_bias,
                    'recent_train_win_rate': None,
                    'eval_win_rate': None,
                }
                append_csv_row(csv_path, row, header)
            if args.ckpt_every and ep % args.ckpt_every == 0:
                ckpt_path = Path(args.ckpt_dir) / f'agent_ep{ep}.npz'
                agent.save(ckpt_path)
            if ep % args.eval_every == 0:
                eval_wins = 0
                for _ in range(200):
                    ww, _ = play_episode(agent, args.width, args.height, args.mines, seed=int(rng.integers(1_000_000_000)), learn=False)
                    eval_wins += 1 if ww else 0
                recent_rate = wins_recent / args.eval_every
                eval_rate = eval_wins / 200
                print(f"Episode {ep}: recent train win% ~ {recent_rate:.3f}, eval win% {eval_rate:.3f}")
                wins_recent = 0
                # Log eval summary row
                row = {
                    'phase': 'eval', 'episode': ep, 'seed': None,
                    'width': args.width, 'height': args.height, 'mines': args.mines,
                    'lr': args.lr, 'l2': args.l2,
                    'win': None,
                    'reveal_actions': None, 'flag_actions': None, 'total_actions': None, 'final_revealed': None, 'avg_pred_p_mine': None,
                    'recent_train_win_rate': recent_rate,
                    'eval_win_rate': eval_rate,
                }
                append_csv_row(csv_path, row, header)
    except KeyboardInterrupt:
        # Ensure latest is saved on interrupt
        agent.save(latest_path)
        print("\n[train] Interrupted. Saved latest checkpoint.")

if __name__ == '__main__':
    main()
