from __future__ import annotations
import argparse
import csv
from pathlib import Path
from statistics import mean


def parse_csv(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def to_float(x):
    if x is None or x == '' or x == 'None':
        return None
    try:
        return float(x)
    except Exception:
        return None


def summarize(rows):
    train = [r for r in rows if r.get('phase') == 'train']
    evals = [r for r in rows if r.get('phase') == 'eval']
    out = []
    if not train:
        return "No training rows found."
    # Overall win rate
    wins = [int(r['win']) for r in train if r.get('win') not in ('', None)]
    win_rate = sum(wins) / len(wins) if wins else 0.0
    # Mine reveal rate
    rmr = [to_float(r.get('reveal_mine_rate')) for r in train]
    rmr = [x for x in rmr if x is not None]
    # Predicted probabilities
    p_all = [to_float(r.get('avg_pred_p_mine')) for r in train]
    p_all = [x for x in p_all if x is not None]
    p_safe = [to_float(r.get('avg_pred_p_mine_safe')) for r in train if r.get('avg_pred_p_mine_safe') not in ('', None)]
    p_safe = [x for x in p_safe if x is not None]
    p_mine = [to_float(r.get('avg_pred_p_mine_mine')) for r in train if r.get('avg_pred_p_mine_mine') not in ('', None)]
    p_mine = [x for x in p_mine if x is not None]
    # Weight stats
    wnorm = [to_float(r.get('w_norm')) for r in train]
    wnorm = [x for x in wnorm if x is not None]

    out.append(f"Overall train episodes: {len(train)}")
    out.append(f"Overall train win rate: {win_rate:.3f}")
    if rmr:
        out.append(f"Avg reveal mine rate: {mean(rmr):.3f}")
    if p_all:
        out.append(f"Avg predicted p(mine) on reveals: {mean(p_all):.3f}")
    if p_safe:
        out.append(f"Avg predicted p(mine) on safe reveals: {mean(p_safe):.3f}")
    if p_mine:
        out.append(f"Avg predicted p(mine) on mine reveals: {mean(p_mine):.3f}")
    if wnorm:
        out.append(f"Avg weight norm: {mean(wnorm):.3f}")

    if evals:
        last_eval = evals[-1]
        out.append("")
        out.append("Last evaluation:")
        out.append(f"  episode: {last_eval.get('episode')}")
        out.append(f"  recent_train_win_rate: {last_eval.get('recent_train_win_rate')}")
        out.append(f"  eval_win_rate: {last_eval.get('eval_win_rate')}")

    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_csv', type=str, default='logs/train_log.csv')
    parser.add_argument('--out', type=str, default='REPORT.md')
    args = parser.parse_args()

    rows = parse_csv(Path(args.log_csv))
    text = summarize(rows)
    Path(args.out).write_text('# Training Report\n\n' + text + '\n')
    print(text)


if __name__ == '__main__':
    main()


