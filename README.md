# Reward Design with Generalized Prioritized Sweeping

This project trains Taxi-v4 agents with SARSA, optional Prioritized Sweeping, and optional kernel-based generalization.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Full reproducible run with 5 seeds:

```bash
python main.py
```

Single-seed grading/debug run:

```bash
python main.py --seed 1
```

Short development smoke run:

```bash
python main.py --fast
```

Optional policy animation after training:

```bash
python main.py --seed 1 --visualize --visualize-reward base
```

Only train and show an animation, without rerunning all experiment plots:

```bash
python main.py --visualize-only
```

Show one seed only:

```bash
python main.py --seed 1 --visualize-only
```

`--visualize-only` trains the full framework for 400 episodes per seed by default.

## Outputs

By default, results are written to `results/`:

- `component_ablation.csv` plus smoothed line plots with 95% CI.
- `reward_comparison.csv` plus smoothed line plots with 95% CI.
- `parameter_sweep_seed_results.csv` and `parameter_sweep_summary.csv`.
- Parameter sweep line plots and heatmaps.
- `run_manifest.txt` with the exact run settings.

## Implemented Requirements

- Taxi-v4 with `is_rainy=False` when supported by the installed Gymnasium version.
- One `GeneralizedSweepingSARSAAgent` class with flags for planning and kernel generalization.
- SARSA real updates and SARSA-style Prioritized Sweeping planning updates.
- Symmetric logistic kernel over taxi Manhattan distance, restricted to states with the same passenger status/location and destination.
- Four required reward designs: base, sparse, reward 3, reward 4.
- Additional potential-based shaping comparison as reward 5.
- Component ablations with and without planning and kernel.
- Parameter sweep over two parameters: learning rate `alpha` and planning steps `n`.
- Metrics: environment reward, training reward, success, episode length, illegal actions, wall bumps, unique states, mean absolute TD error.
- Reproducible seeds and a single-seed command for faster checking.
