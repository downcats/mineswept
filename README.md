
# Mineswept 🚩💣

Headless Minesweeper environment and learning agents that improve over time via online updates + deterministic rules.

## 🛠️ Setup

1) Create a virtual environment and install requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🧩 Components

- `minesweeper/engine.py` — Game logic and ASCII renderer
- `minesweeper/features.py` — Rich feature extraction for candidate cells
- `minesweeper/agents/rule_agent.py` — Deterministic safe/mine inference rules
- `minesweeper/agents/online_lr_agent.py` — Online logistic regression agent with save/load
- `minesweeper/agents/mlp_agent.py` — Two-layer MLP agent trained online with backprop
- `train.py` — Self-play training loop with CSV logging, autosave, auto-resume, and checkpoints
- `play.py` — Play in CLI with live learning and autosave
- `report.py` — Summarize logs into a Markdown report

## 🚀 Quick start

Hard-mode defaults (16×16, 40 mines) are enabled for both train and play.

Train with autosave and logging (MLP agent):

```bash
python train.py --agent mlp --episodes 2000 --eval_every 200 --log_every 1 --log_csv logs/train_mlp.csv
```

Play a game using the latest checkpoint (auto-resume):

```bash
python play.py --agent mlp --delay 0.05
```

Resume from a specific checkpoint:

```bash
python play.py --resume checkpoints/agent_ep20.npz
```

Generate a report from a log:

```bash
python report.py --log_csv logs/train_mlp.csv --out REPORT.md
```

## 🧠 How learning works

- After each reveal, the agent gets a label: mine=1, safe=0
- It updates its model to reduce predicted mine probability for safe cells and increase it for mines
- Deterministic rules are tried first; the model scores remaining hidden cells by predicted safety
- Logistic agent uses a numerically stable sigmoid and gradient clipping
- MLP agent uses 2 ReLU layers with a sigmoid output, trained online via backprop

## 📈 Logging

Training writes `logs/*.csv` with per-episode metrics and periodic eval rows, including:

- Episode: `win`, `reveal_actions`, `flag_actions`, `total_actions`, `final_revealed`
- Predictions: `avg_pred_p_mine`, `avg_pred_p_mine_safe`, `avg_pred_p_mine_mine`, `reveal_mines`, `reveal_mine_rate`
- Model: `w_norm`, `w_bias` (for MLP these reflect total norm and output bias)
- Eval rows: `recent_train_win_rate`, `eval_win_rate`

Tail the log:

```bash
tail -n 20 logs/train_mlp.csv
```

## 💾 Checkpoints

- Autosave to `checkpoints/latest.npz` after each episode in training and after each game in play
- Periodic archival checkpoints at `checkpoints/agent_ep{N}.npz` via `--ckpt_every`
- Auto-resume respects `--agent` type; if mismatch, a fresh agent is initialized
- You can still explicitly `--resume` any checkpoint

## ⚙️ Configuration

Common flags:

- `--width`, `--height`, `--mines`: board size and mine count (defaults: 16x16, 40 mines)
- `--agent {lr,mlp}`: learner type (online logistic or 2-layer MLP)
- `--lr`, `--l2`: learning rate and L2 regularization strength
- `--episodes`: number of episodes to run in training
- `--eval_every`: how often to run evaluation
- `--log_csv`, `--log_every`: CSV file and frequency
- `--ckpt_dir`, `--ckpt_every`, `--resume`, `--auto_resume_latest`

## 📝 Notes

- First click is guaranteed safe and expands zeroes recursively
- The model is intentionally simple and fast; you can extend features or swap in a more expressive learner later

---

Made with ❤️ for classic puzzle fans.

