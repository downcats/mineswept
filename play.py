
from __future__ import annotations
import argparse
import time
from minesweeper.engine import Minesweeper
from minesweeper.features import extract_cell_features
from minesweeper.agents.online_lr_agent import OnlineLogisticAgent
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=9)
    parser.add_argument('--height', type=int, default=9)
    parser.add_argument('--mines', type=int, default=10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--l2', type=float, default=1e-4)
    parser.add_argument('--delay', type=float, default=0.05)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--auto_resume_latest', action='store_true', default=True,
                        help='Auto-load checkpoints/latest.npz if present and --resume not set')
    args = parser.parse_args()

    env = Minesweeper(args.width, args.height, args.mines, seed=args.seed)
    fx = extract_cell_features(env, 0, 0)
    latest_path = Path('checkpoints') / 'latest.npz'
    if args.resume:
        agent = OnlineLogisticAgent.load(args.resume)
        print(f"[play] Loaded checkpoint: {args.resume}")
    elif args.auto_resume_latest and latest_path.exists():
        agent = OnlineLogisticAgent.load(latest_path)
        print(f"[play] Auto-resumed from {latest_path}")
    else:
        agent = OnlineLogisticAgent(feature_dim=fx.shape[0], lr=args.lr, l2=args.l2, seed=args.seed)
        print("[play] Initialized new agent")

    # First reveal center
    env.reveal(args.width // 2, args.height // 2)
    print(env.render_ascii())
    print()
    try:
        while not env.game_over:
            kind, (x, y) = agent.next_action(env)
            if kind == 'flag':
                env.flag(x, y)
            else:
                # reveal
                xfeat = extract_cell_features(env, x, y)
                was_mine = env.grid[y][x].is_mine
                env.reveal(x, y)
                y_label = 1.0 if was_mine else 0.0
                agent.update(xfeat, y_label)
            print(env.render_ascii())
            print()
            time.sleep(args.delay)
    finally:
        # Save progress after the game ends (win/lose) or on interruption
        Path('checkpoints').mkdir(parents=True, exist_ok=True)
        agent.save(Path('checkpoints') / 'latest.npz')

    print('WIN' if env.win else 'LOSE')

if __name__ == '__main__':
    main()
