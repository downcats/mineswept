
# Mineswept

Headless Minesweeper environment and a learning agent that improves over time via online logistic regression plus deterministic rules.

## Setup

1) Create a virtual environment and install requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Components

- `minesweeper/engine.py`: Game logic and ASCII renderer
- `minesweeper/features.py`: Feature extraction for candidate cells
- `minesweeper/agents/rule_agent.py`: Deterministic safe/mine inference rules
- `minesweeper/agents/online_lr_agent.py`: Online logistic regression agent with save/load
- `train.py`: Self-play training loop with CSV logging and checkpoints
- `play.py`: Play in CLI with live learning and autosave

## Quick start

Train with autosave and logging:

```bash
python train.py --episodes 2000 --eval_every 200 --log_every 1 --log_csv logs/train_log.csv
```

Play a game using the latest checkpoint (auto-resume):

```bash
python play.py --delay 0.05
```

Resume from a specific checkpoint:

```bash
python play.py --resume checkpoints/agent_ep20.npz
```

## How learning works

- After each reveal, the agent gets a label: mine=1, safe=0
- It updates its weights to reduce predicted mine probability for safe cells and increase it for mines
- Deterministic rules are tried first; the model scores remaining hidden cells by predicted safety

## Logging

Training writes `logs/train_log.csv` with per-episode metrics and periodic eval rows:

- `win`, `reveal_actions`, `flag_actions`, `total_actions`, `final_revealed`, `avg_pred_p_mine`
- `w_norm`, `w_bias` show weight evolution
- Eval rows include `recent_train_win_rate` and `eval_win_rate`

Tail the log:

```bash
tail -n 20 logs/train_log.csv
```

## Checkpoints

- Autosave to `checkpoints/latest.npz` after each episode in training and after each game in play
- Periodic archival checkpoints at `checkpoints/agent_ep{N}.npz` via `--ckpt_every`
- Resume automatically from `latest.npz` or specify `--resume` explicitly

## Configuration

Common flags:

- `--width`, `--height`, `--mines`: board size and mine count (default 9x9, 10 mines)
- `--lr`, `--l2`: learning rate and L2 regularization strength
- `--episodes`: number of episodes to run in training
- `--eval_every`: how often to run evaluation
- `--log_csv`, `--log_every`: CSV file and frequency
- `--ckpt_dir`, `--ckpt_every`, `--resume`, `--auto_resume_latest`

## Notes

- First click is guaranteed safe and expands zeroes recursively
- The model is intentionally simple and fast; you can extend features or swap in a more expressive learner later

