import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from agent import GeneralizedSweepingSARSAAgent
from visualize import run_and_visualize_policy


sns.set_theme(style="whitegrid")

TAXI_LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]
MOVEMENT_ACTIONS = {0, 1, 2, 3}

REWARD_DISPLAY_NAMES = {
    "base": "Base Environment Reward",
    "sparse": "Sparse Reward",
    "reward_3": "Dense Target Progress Reward",
    "reward_4": "Subgoal Safety Reward",
    "reward_5": "Composite Remaining Path Reward",
}

SMOOTHED_METRICS = [
    "Env_Reward",
    "Success",
    "Illegal_Actions",
]


def make_taxi_env(render_mode=None):
    kwargs = {"is_rainy": False}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    try:
        return gym.make("Taxi-v4", **kwargs)
    except TypeError:
        kwargs.pop("is_rainy", None)
        return gym.make("Taxi-v4", **kwargs)


def get_current_target_position(state, agent):
    _, _, passenger_loc, dest_idx = agent.decode_state(state)
    if passenger_loc < 4:
        return TAXI_LOCS[passenger_loc]
    return TAXI_LOCS[dest_idx]


def manhattan_to_current_target(state, agent):
    taxi_row, taxi_col, _, _ = agent.decode_state(state)
    target_row, target_col = get_current_target_position(state, agent)
    return abs(taxi_row - target_row) + abs(taxi_col - target_col)


def remaining_task_distance(state, agent):
    taxi_row, taxi_col, passenger_loc, dest_idx = agent.decode_state(state)
    dest_row, dest_col = TAXI_LOCS[dest_idx]

    if passenger_loc < 4:
        passenger_row, passenger_col = TAXI_LOCS[passenger_loc]
        return (
            abs(taxi_row - passenger_row)
            + abs(taxi_col - passenger_col)
            + abs(passenger_row - dest_row)
            + abs(passenger_col - dest_col)
        )

    return abs(taxi_row - dest_row) + abs(taxi_col - dest_col)


def was_successful_dropoff(env_reward):
    return env_reward == 20


def was_illegal_pickup_or_dropoff(env_reward):
    return env_reward == -10


def was_wall_bump(state, action, next_state):
    return action in MOVEMENT_ACTIONS and state == next_state


def was_successful_pickup(state, next_state, agent):
    _, _, passenger_before, _ = agent.decode_state(state)
    _, _, passenger_after, _ = agent.decode_state(next_state)
    return passenger_before < 4 and passenger_after == 4


def calculate_custom_reward(reward_type, env_reward, state, action, next_state, agent):
    """Return the training reward used by the agent for the selected design."""
    if reward_type == "base":
        return float(env_reward)

    success = was_successful_dropoff(env_reward)
    illegal = was_illegal_pickup_or_dropoff(env_reward)
    pickup = was_successful_pickup(state, next_state, agent)
    wall_bump = was_wall_bump(state, action, next_state)

    old_distance = manhattan_to_current_target(state, agent)
    new_distance = manhattan_to_current_target(next_state, agent)
    progress = old_distance - new_distance

    if reward_type == "sparse":
        return 1.0 if success else 0.0

    if reward_type == "reward_3":
        reward = 0.75 * progress - 0.05
        if pickup:
            reward += 2.0
        if success:
            reward += 5.0
        if illegal:
            reward -= 2.0
        if wall_bump:
            reward -= 0.5
        return float(reward)

    if reward_type == "reward_4":
        reward = -0.10
        if pickup:
            reward += 3.0
        if success:
            reward += 10.0
        if illegal:
            reward -= 4.0
        if wall_bump:
            reward -= 1.0
        if progress < 0:
            reward -= 0.25
        return float(reward)

    if reward_type == "reward_5":
        distance_before = remaining_task_distance(state, agent)
        distance_after = remaining_task_distance(next_state, agent)
        max_path = 16.0

        reward = (distance_before - distance_after) / max_path
        reward -= 0.02

        if success:
            reward += 1.0
        if illegal:
            reward -= 0.3

        return float(reward)

    raise ValueError(f"Unknown reward type: {reward_type}")


def train_agent(env, agent, num_episodes=1000, reward_type="base", seed=None):
    episode_records = []

    for episode in range(1, num_episodes + 1):
        reset_seed = None if seed is None else seed * 100_000 + episode
        state, _ = env.reset(seed=reset_seed)
        action = agent.choose_action(state)

        done = False
        env_reward_total = 0.0
        illegal_actions = 0
        success = False

        while not done:
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            train_reward = calculate_custom_reward(
                reward_type,
                env_reward,
                state,
                action,
                next_state,
                agent,
            )
            next_action = agent.choose_action(next_state)
            agent.learn(
                state,
                action,
                train_reward,
                next_state,
                next_action,
                done,
            )

            success = success or was_successful_dropoff(env_reward)
            illegal_actions += int(was_illegal_pickup_or_dropoff(env_reward))
            env_reward_total += env_reward

            state = next_state
            action = next_action

        agent.decay_epsilon()
        episode_records.append(
            {
                "Episode": episode,
                "Reward_Type": reward_type,
                "Env_Reward": env_reward_total,
                "Success": int(success),
                "Illegal_Actions": illegal_actions,
                "Epsilon": agent.epsilon,
            }
        )

    return episode_records


def slugify(value):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def models_dir(output_dir):
    output_dir = Path(output_dir)
    return output_dir / "models"


def smooth_metrics(df, smoothing_window):
    window = max(1, int(smoothing_window))
    group_cols = ["Config", "Seed"]

    for metric in SMOOTHED_METRICS:
        df[f"Smoothed_{metric}"] = df.groupby(group_cols)[metric].transform(
            lambda values: values.rolling(window=window, min_periods=1).mean()
        )

    return df


def lineplot_with_ci(data, x, y, hue, style=None, markers=False):
    kwargs = {
        "data": data,
        "x": x,
        "y": y,
        "hue": hue,
        "markers": markers,
    }
    if style is not None:
        kwargs["style"] = style

    try:
        return sns.lineplot(**kwargs, errorbar=("ci", 95))
    except TypeError:
        return sns.lineplot(**kwargs, ci=95)


def save_ablation_plots(df, output_dir):
    plot_specs = [
        ("base", "Env_Reward", "Environment reward"),
        ("base", "Success", "Success rate"),
        ("reward_4", "Env_Reward", "Environment reward"),
        ("reward_4", "Success", "Success rate"),
    ]

    for reward_type, metric, ylabel in plot_specs:
        plot_df = df[df["Reward_Type"] == reward_type]
        plt.figure(figsize=(11, 6))
        lineplot_with_ci(
            data=plot_df,
            x="Episode",
            y=f"Smoothed_{metric}",
            hue="Setup",
        )
        plt.title(f"Ablation Study - {REWARD_DISPLAY_NAMES[reward_type]} - {ylabel}")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        if metric == "Success":
            plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(
            output_dir / f"ablation_{reward_type}_{slugify(metric)}.png",
            dpi=180,
        )
        plt.close()


def save_reward_comparison_plots(df, output_dir):
    plot_specs = [
        ("Env_Reward", "Environment reward"),
        ("Success", "Success rate"),
    ]

    for metric, ylabel in plot_specs:
        plt.figure(figsize=(11, 6))
        lineplot_with_ci(
            data=df,
            x="Episode",
            y=f"Smoothed_{metric}",
            hue="Config",
        )
        plt.title(f"Reward Comparison - {ylabel}")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        if metric == "Success":
            plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(output_dir / f"reward_comparison_{slugify(metric)}.png", dpi=180)
        plt.close()


def agent_params_from_config(config):
    meta_keys = {
        "reward_type",
        "setup_name",
        "agent_name",
        "save_family",
    }
    return {key: value for key, value in config.items() if key not in meta_keys}


def save_model_artifact(agent, output_dir, family, reward_type, agent_name, seed):
    model_dir = models_dir(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    file_path = model_dir / f"{family}__{reward_type}__{agent_name}__seed_{seed}.npz"
    np.savez_compressed(
        file_path,
        q_table=agent.q_table,
        reward_type=reward_type,
        agent_name=agent_name,
        seed=seed,
        family=family,
    )


def infer_saved_model_family(agent_name):
    if agent_name == "full":
        return "reward_comparison"
    return "ablation"


def load_saved_model(output_dir, agent_name, reward_type, seed, planning_steps):
    if agent_name != "full" and reward_type not in {"base", "reward_4"}:
        raise ValueError(
            "Saved non-full models are available only for rewards: base, reward_4."
        )

    family = infer_saved_model_family(agent_name)
    file_path = models_dir(output_dir) / f"{family}__{reward_type}__{agent_name}__seed_{seed}.npz"
    if not file_path.exists():
        raise FileNotFoundError(f"Saved model not found: {file_path}")

    agent = GeneralizedSweepingSARSAAgent(
        **visualization_agent_configs(planning_steps)[agent_name]
    )
    with np.load(file_path) as data:
        agent.q_table = data["q_table"]
    return agent, file_path


def run_experiment(
    configs,
    num_episodes,
    seeds,
    exp_name,
    output_dir,
    smoothing_window,
    save_models=True,
    make_plots=True,
):
    print("\n========================================================")
    print(f"Running {exp_name}: {len(configs)} configs, {len(seeds)} seed(s)")
    print("========================================================")

    started = time.perf_counter()
    all_records = []

    for config_name, config in configs.items():
        reward_type = config.get("reward_type", "base")
        setup_name = config.get("setup_name", config_name)
        agent_name = config.get("agent_name")
        save_family = config.get("save_family")
        print(f">>> {config_name}")

        for seed in seeds:
            np.random.seed(seed)
            env = make_taxi_env()
            agent = GeneralizedSweepingSARSAAgent(**agent_params_from_config(config))

            try:
                records = train_agent(
                    env,
                    agent,
                    num_episodes=num_episodes,
                    reward_type=reward_type,
                    seed=seed,
                )
            finally:
                env.close()

            if save_models and agent_name and save_family:
                save_model_artifact(
                    agent,
                    output_dir,
                    save_family,
                    reward_type,
                    agent_name,
                    seed,
                )

            for record in records:
                record["Config"] = config_name
                record["Seed"] = seed
                record["Setup"] = setup_name
                all_records.append(record)

    df = pd.DataFrame(all_records)
    df = smooth_metrics(df, smoothing_window)

    csv_path = output_dir / f"{slugify(exp_name)}.csv"
    df.to_csv(csv_path, index=False)

    if make_plots:
        if exp_name == "Component Ablation":
            save_ablation_plots(df, output_dir)
        elif exp_name == "Reward Comparison":
            save_reward_comparison_plots(df, output_dir)

    elapsed = time.perf_counter() - started
    print(f"[+] Saved {csv_path.name} in {elapsed / 60:.2f} minutes")
    return df


def calculate_epsilon_decay(num_episodes):
    return 0.05 ** (1.0 / (0.6 * num_episodes))


def component_ablation_configs(planning_steps, epsilon_decay):
    configs = {}
    setup_specs = [
        (
            "Pure SARSA",
            "pure_sarsa",
            {
                "epsilon_decay": epsilon_decay,
                "use_planning": False,
                "use_kernel": False,
            },
        ),
        (
            "SARSA + Planning",
            "planning",
            {
                "alpha": 0.3,
                "epsilon_decay": epsilon_decay,
                "use_planning": True,
                "use_kernel": False,
                "planning_steps": planning_steps,
            },
        ),
        (
            "SARSA + Kernel",
            "kernel",
            {
                "alpha": 0.5,
                "beta": 4.0,
                "c": 0.5,
                "epsilon_decay": epsilon_decay,
                "max_kernel_neighbors": 4,
                "kernel_scale": 0.5,
                "use_planning": False,
                "use_kernel": True,
            },
        ),
        (
            "Full Framework",
            "full",
            {
                "alpha": 0.5,
                "epsilon_decay": epsilon_decay,
                "beta": 4.0,
                "c": 0.5,
                "max_kernel_neighbors": 4,
                "kernel_scale": 0.5,
                "use_planning": True,
                "use_kernel": True,
                "planning_steps": planning_steps,
            },
        ),
    ]

    for reward_type in ("base", "reward_4"):
        reward_name = REWARD_DISPLAY_NAMES[reward_type]
        for setup_name, agent_name, params in setup_specs:
            configs[f"{reward_name} | {setup_name}"] = {
                "reward_type": reward_type,
                "setup_name": setup_name,
                "agent_name": agent_name,
                "save_family": "ablation",
                **params,
            }

    return configs


def reward_comparison_configs(planning_steps, epsilon_decay):
    return {
        display_name: {
            "reward_type": reward_type,
            "setup_name": display_name,
            "agent_name": "full",
            "save_family": "reward_comparison",
            "alpha": 0.5,
            "epsilon_decay": epsilon_decay,
            "beta": 4.0,
            "c": 0.5,
            "max_kernel_neighbors": 4,
            "kernel_scale": 0.5,
            "use_planning": True,
            "use_kernel": True,
            "planning_steps": planning_steps,
        }
        for reward_type, display_name in REWARD_DISPLAY_NAMES.items()
    }


def visualization_agent_configs(planning_steps):
    return {
        "pure_sarsa": {
            "use_planning": False,
            "use_kernel": False,
            "planning_steps": planning_steps,
        },
        "planning": {
            "alpha": 0.3,
            "use_planning": True,
            "use_kernel": False,
            "planning_steps": planning_steps,
        },
        "kernel": {
            "alpha": 0.5,
            "beta": 4.0,
            "c": 0.5,
            "max_kernel_neighbors": 4,
            "kernel_scale": 0.5,
            "use_planning": False,
            "use_kernel": True,
            "planning_steps": planning_steps,
        },
        "full": {
            "alpha": 0.5,
            "beta": 4.0,
            "c": 0.5,
            "max_kernel_neighbors": 4,
            "kernel_scale": 0.5,
            "use_planning": True,
            "use_kernel": True,
            "planning_steps": planning_steps,
        },
    }


def mean_tail(records, metric, tail_window=50):
    tail = records[-min(tail_window, len(records)) :]
    return float(np.mean([row[metric] for row in tail]))


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    return float(1.96 * np.std(values, ddof=1) / np.sqrt(values.size))


def run_parameter_sweep(
    seeds,
    sweep_episodes,
    epsilon_decay,
    output_dir,
    make_plots=True,
):
    print("\n========================================================")
    print("Running Parameter Sweep: alpha x planning_steps")
    print("========================================================")

    rows = []
    alpha_values = [0.2, 0.8]
    planning_step_values = [2, 10]
    reward_types = ["base", "reward_4"]

    started = time.perf_counter()

    for reward_type in reward_types:
        for alpha in alpha_values:
            for planning_steps in planning_step_values:
                for seed in seeds:
                    np.random.seed(seed)
                    env = make_taxi_env()
                    agent = GeneralizedSweepingSARSAAgent(
                        alpha=alpha,
                        beta=4.0,
                        c=0.5,
                        max_kernel_neighbors=4,
                        kernel_scale=0.5,
                        epsilon_decay=epsilon_decay,
                        planning_steps=planning_steps,
                        use_planning=True,
                        use_kernel=True,
                    )

                    try:
                        records = train_agent(
                            env,
                            agent,
                            num_episodes=sweep_episodes,
                            reward_type=reward_type,
                            seed=seed,
                        )
                    finally:
                        env.close()

                    rows.append(
                        {
                            "Reward_Type": reward_type,
                            "Reward_Name": REWARD_DISPLAY_NAMES[reward_type],
                            "Alpha": alpha,
                            "Planning_Steps": planning_steps,
                            "Seed": seed,
                            "Final_Env_Reward": mean_tail(records, "Env_Reward"),
                            "Final_Success_Rate": mean_tail(records, "Success"),
                            "Final_Illegal_Actions": mean_tail(records, "Illegal_Actions"),
                        }
                    )

                print(
                    "  "
                    f"{reward_type}, alpha={alpha}, planning_steps={planning_steps}"
                )

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "parameter_sweep_seed_results.csv", index=False)

    group_cols = ["Reward_Type", "Reward_Name", "Alpha", "Planning_Steps"]
    summary = (
        df.groupby(group_cols)
        .agg(
            Final_Env_Reward_Mean=("Final_Env_Reward", "mean"),
            Final_Env_Reward_CI95=("Final_Env_Reward", ci95),
            Final_Success_Rate_Mean=("Final_Success_Rate", "mean"),
            Final_Success_Rate_CI95=("Final_Success_Rate", ci95),
            Final_Illegal_Actions_Mean=("Final_Illegal_Actions", "mean"),
            Final_Illegal_Actions_CI95=("Final_Illegal_Actions", ci95),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "parameter_sweep_summary.csv", index=False)

    if make_plots:
        save_sweep_plots(df, summary, output_dir)

    elapsed = time.perf_counter() - started
    print(f"[+] Saved parameter sweep in {elapsed / 60:.2f} minutes")
    return df, summary


def save_sweep_plots(df, summary, output_dir):
    for reward_type, reward_df in summary.groupby("Reward_Type"):
        heatmap_data = reward_df.pivot(
            index="Alpha",
            columns="Planning_Steps",
            values="Final_Env_Reward_Mean",
        )
        plt.figure(figsize=(7, 5))
        sns.heatmap(heatmap_data, annot=True, cmap="viridis", fmt=".2f")
        plt.title(f"{REWARD_DISPLAY_NAMES[reward_type]}: Mean final environment reward")
        plt.tight_layout()
        plt.savefig(
            output_dir / f"parameter_sweep_{reward_type}_env_reward_heatmap.png",
            dpi=180,
        )
        plt.close()


def write_run_manifest(output_dir, seeds, episodes, sweep_episodes, planning_steps):
    lines = [
        "Reward Design with Generalized Prioritized Sweeping",
        "",
        f"Seeds: {seeds}",
        f"Training episodes per experiment: {episodes}",
        f"Sweep episodes per setting: {sweep_episodes}",
        f"Default planning steps: {planning_steps}",
        "Environment: Taxi-v4, is_rainy=False when supported by Gymnasium.",
        "Smoothing: rolling mean over the configured window, grouped by config and seed.",
    ]
    (output_dir / "run_manifest.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Taxi-v4 reward-design experiments with SARSA, prioritized sweeping, and kernel generalization."
    )
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--fast", action="store_true", help="Short seed-1 development run.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--sweep-episodes", type=int, default=None)
    parser.add_argument("--planning-steps", type=int, default=5)
    parser.add_argument("--smoothing-window", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Train and render a policy without rerunning all experiments.",
    )
    parser.add_argument(
        "--visualize-saved",
        action="store_true",
        help="Load a saved trained policy from results/models and render it.",
    )
    parser.add_argument(
        "--visualize-agent",
        choices=["pure_sarsa", "planning", "kernel", "full"],
        default="full",
        help="Agent setup to train for the pygame animation.",
    )
    parser.add_argument(
        "--visualize-reward",
        choices=list(REWARD_DISPLAY_NAMES),
        default="base",
    )
    parser.add_argument("--visualize-episodes", type=int, default=1)
    return parser.parse_args()


def resolve_run_settings(args):
    if args.fast:
        seeds = args.seeds or [1]
        episodes = args.episodes or 300
        sweep_episodes = args.sweep_episodes or 200
        smoothing_window = args.smoothing_window or 25
    elif args.visualize_only or args.visualize_saved:
        seeds = args.seeds or [1, 2, 3, 4, 5]
        episodes = args.episodes or 600
        sweep_episodes = args.sweep_episodes or 150
        smoothing_window = args.smoothing_window or 50
    else:
        seeds = args.seeds or [1, 2, 3, 4, 5]
        episodes = args.episodes or 600
        sweep_episodes = args.sweep_episodes or 400
        smoothing_window = args.smoothing_window or 50

    return seeds, episodes, sweep_episodes, smoothing_window


def visualize_saved_policies(args, seeds):
    print("\n[+] Loading saved policies for visualization...")
    print(
        "[+] Visualization setup: "
        f"agent={args.visualize_agent}, reward={args.visualize_reward}, seeds={seeds}"
    )

    loaded_any = False
    for seed in seeds:
        try:
            agent, model_path = load_saved_model(
                args.output_dir,
                args.visualize_agent,
                args.visualize_reward,
                seed,
                args.planning_steps,
            )
        except FileNotFoundError:
            print(f"\n[-] No saved model found for seed {seed}; skipping.")
            continue

        loaded_any = True
        print(f"\n[+] Loaded saved model for seed {seed}: {model_path.name}")
        run_and_visualize_policy(
            agent,
            num_episodes=args.visualize_episodes,
            max_steps=150,
            delay=0.10,
            start_seed=seed,
        )

    if not loaded_any:
        raise FileNotFoundError(
            "No saved models matched the requested agent/reward/seed selection."
        )


def visualize_trained_policies(args, seeds, episodes):
    print("\n[+] Training policies for visualization...")
    print(
        "[+] Visualization setup: "
        f"agent={args.visualize_agent}, reward={args.visualize_reward}, "
        f"episodes={episodes}, seeds={seeds}"
    )

    agent_config = visualization_agent_configs(args.planning_steps)[args.visualize_agent]
    agent_config["epsilon_decay"] = calculate_epsilon_decay(episodes)

    for seed in seeds:
        print(f"\n[+] Training visualization policy for seed {seed}...")
        np.random.seed(seed)
        env = make_taxi_env()
        agent = GeneralizedSweepingSARSAAgent(**agent_config)

        try:
            records = train_agent(
                env,
                agent,
                num_episodes=episodes,
                reward_type=args.visualize_reward,
                seed=seed,
            )
        finally:
            env.close()

        tail_success = mean_tail(records, "Success")
        tail_reward = mean_tail(records, "Env_Reward")
        tail_illegal = mean_tail(records, "Illegal_Actions")
        print(
            "[+] Seed "
            f"{seed} training tail: success={tail_success:.2f}, "
            f"reward={tail_reward:.2f}, illegal={tail_illegal:.2f}"
        )

        run_and_visualize_policy(
            agent,
            num_episodes=args.visualize_episodes,
            max_steps=150,
            delay=0.10,
            start_seed=seed,
        )


def main():
    args = parse_args()
    if args.visualize_only or args.visualize_saved:
        args.visualize = True

    seeds, episodes, sweep_episodes, smoothing_window = resolve_run_settings(args)
    epsilon_decay = calculate_epsilon_decay(episodes)
    sweep_epsilon_decay = calculate_epsilon_decay(sweep_episodes)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    write_run_manifest(
        args.output_dir,
        seeds,
        episodes,
        sweep_episodes,
        args.planning_steps,
    )

    make_plots = not args.no_plots
    started = time.perf_counter()

    ablation_df = pd.DataFrame()
    reward_df = pd.DataFrame()

    if not args.visualize_only and not args.visualize_saved:
        ablation_df = run_experiment(
            component_ablation_configs(args.planning_steps, epsilon_decay),
            episodes,
            seeds,
            "Component Ablation",
            args.output_dir,
            smoothing_window,
            make_plots=make_plots,
        )

        reward_df = run_experiment(
            reward_comparison_configs(args.planning_steps, epsilon_decay),
            episodes,
            seeds,
            "Reward Comparison",
            args.output_dir,
            smoothing_window,
            make_plots=make_plots,
        )

        if not args.skip_sweep:
            run_parameter_sweep(
                seeds,
                sweep_episodes,
                sweep_epsilon_decay,
                args.output_dir,
                make_plots=make_plots,
            )

    if args.visualize:
        if args.visualize_saved:
            visualize_saved_policies(args, seeds)
        else:
            visualize_trained_policies(args, seeds, episodes)

    elapsed = time.perf_counter() - started
    print("\n[+] All requested experiments completed.")
    print(f"[+] Results directory: {args.output_dir.resolve()}")
    print(f"[+] Total runtime: {elapsed / 60:.2f} minutes")
    print(
        "[+] Rows saved: "
        f"ablation={len(ablation_df)}, reward_comparison={len(reward_df)}"
    )


if __name__ == "__main__":
    main()
