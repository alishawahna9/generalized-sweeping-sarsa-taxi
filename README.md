# Reward Design with Generalized Prioritized Sweeping

This project trains Taxi-v4 agents with SARSA, optional Prioritized Sweeping, and optional kernel-based generalization.

## Quick Start

Install experiment dependencies:

```bash
python -m pip install -r requirements.txt
```

Optional report/animation tooling (GIF export, PDF/DOCX report building):

```bash
python -m pip install -r requirements-dev.txt
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

Saved-policy visualization first scores 100 candidate start seeds per saved/trained
model seed and renders the best episode found. To change that search size:

```bash
python main.py --visualize-saved --visualize-candidates 300
```

Use `--visualize-candidates 1` to show only the original start seed.

For the clearest live demo of the saved models across all seeds and reward designs:

```bash
python run_trained_model.py
```

Saved successful policy GIFs for each reward are included under:

```bash
reports/policy_animations/
```

To regenerate the GIFs from the saved Q-tables in `results/models` (requires `requirements-dev.txt`):

```bash
python make_policy_gifs.py
```

Only train and show an animation, without rerunning all experiment plots:

```bash
python main.py --visualize-only
```

Show one seed only:

```bash
python main.py --seed 1 --visualize-only
```

`--visualize-only` trains the full framework for 500 episodes per seed by default.

## Outputs

By default, results are written to `results/`:

- `component_ablation.csv` plus smoothed line plots with 95% CI.
- `reward_comparison.csv` plus smoothed line plots with 95% CI.
- `parameter_sweep_seed_results.csv` and `parameter_sweep_summary.csv`.
- `models/*.npz` saved Q-tables for visualization and GIF generation.
- Parameter sweep heatmaps (mean ± 95% CI annotated in each cell).
- `run_manifest.txt` with the exact run settings.

## Implemented Requirements

- Taxi-v4 with `is_rainy=False` when supported by the installed Gymnasium version.
- One `GeneralizedSweepingSARSAAgent` class with flags for planning and kernel generalization.
- SARSA real updates and SARSA-style Prioritized Sweeping planning updates.
- Symmetric logistic kernel over taxi Manhattan distance, restricted to states with the same passenger status/location and destination.
- Four required reward designs: base, sparse, reward 3, reward 4.
- Additional custom Composite Remaining Path reward as reward 5.
- Component ablations with and without planning and kernel.
- Parameter sweep over two parameters: learning rate `alpha` and planning steps `n`.
- Metrics saved in the main CSVs: environment reward, normalized environment reward, success, episode length, and illegal pickup/dropoff actions.
- Reproducible seeds and a single-seed command for faster checking.
