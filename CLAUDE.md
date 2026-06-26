# CLAUDE.md

This file gives quick project guidance for coding agents working in this repository.

## Running Experiments

```bash
# Full run: 5 seeds, report-quality plots and CSVs
python main.py

# Required quick grading/debug path: one seed, same experiment structure
python main.py --seed 1

# Short development smoke run
python main.py --fast

# Optional overrides
python main.py --episodes 500 --sweep-episodes 100
python main.py --seed 1 --no-plots
python main.py --seed 1 --visualize --visualize-reward base
python main.py --visualize-only
python main.py --seed 1 --visualize-only
```

All non-visual outputs are written under `results/` by default. The normal run does not open a pygame animation window; use `--visualize` when an animation is needed. Use `--visualize-only` to train and render the full framework for 400 episodes per seed without rerunning every experiment.

## Dependencies

```bash
pip install -r requirements.txt
```

Packages: `numpy`, `gymnasium[toy-text]`, `pandas`, `matplotlib`, `seaborn`, `pygame`.

## Architecture

`agent.py` contains `GeneralizedSweepingSARSAAgent`, a tabular SARSA agent with two optional components:

- `use_planning=True` enables SARSA-style Prioritized Sweeping.
- `use_kernel=True` enables logistic-kernel generalization to nearby Taxi states that share passenger status/location and destination.

`main.py` contains:

1. Component ablation across pure SARSA, planning only, kernel only, and full framework.
2. Reward comparison across base, sparse, reward 3, reward 4, and reward 5.
3. Parameter sweep over `alpha` and `planning_steps` for base and a designed reward.
4. CSV/plot output with rolling smoothing and 95% CI where seeds are available.

`visualize.py` contains the optional pygame policy animation helper.

## Key Design Decisions

- Rewards 3 and 4 are replacement rewards, not base-plus-bonus shaping.
- Reward 5 is the optional potential-based shaping comparison.
- Kernel updates are kept separate from priority-queue insertion.
- Planning updates use epsilon-greedy SARSA-style next actions.
- Kernel neighbors are precomputed and capped to keep runtime practical.
- Single-seed execution is supported with `python main.py --seed 1`.
