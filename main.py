import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent import GeneralizedSweepingSARSAAgent
from visualize import run_and_visualize_policy

REWARD_DISPLAY_NAMES = {
    "base":     "Baseline (Sparse Gym Reward)",
    "sparse":   "Strict Sparse Reward",
    "reward_3": "Manhattan Distance Potential",
    "reward_4": "Subgoal Milestone Reward",
    "reward_5": "Heuristic Reward Shaping (Bonus)",
}

def get_designed_reward(env_reward, state, agent, action, reward_type="base"):
    """מחשב ומחליף את הפרס המקורי של הסביבה לפי מערכת הפרס הנבחרת"""
    if reward_type == "base":
        return env_reward

    elif reward_type == "sparse":
        # פרס דליל: +1 על הורדה מוצלח, 0 בשאר הזמן
        return 1.0 if env_reward == 20 else 0.0

    elif reward_type == "reward_3":
        # עיצוב אישי א': מבוסס מרחק מנהטן מהיעד הנוכחי
        taxi_row, taxi_col, pass_loc, dest_idx = agent.decode_state(state)
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)] # R, G, Y, B
        target_pos = locs[pass_loc] if pass_loc < 4 else locs[dest_idx]
        manhattan_dist = abs(taxi_row - target_pos[0]) + abs(taxi_col - target_pos[1])
        if env_reward == 20:
            return 20.0
        return -float(manhattan_dist)

    elif reward_type == "reward_4":
        # עיצוב אישי ב': נהג זהיר - החמרה בעונש על פעולות לא חוקיות
        if env_reward == -10:
            return -25.0
        elif env_reward == 20:
            return 20.0
        return -1.0

    elif reward_type == "reward_5":
        # Potential-Based Reward Shaping (bonus experiment beyond the 4 required designs).
        # This is NOT a full reward replacement: it returns env_reward + shaping_bonus,
        # preserving the original signal while adding a potential-difference term.
        # Included for comparison only — it does not count as one of the 4 required reward designs.
        taxi_row, taxi_col, pass_loc, dest_idx = agent.decode_state(state)
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)] # R, G, Y, B
        target_pos = locs[pass_loc] if pass_loc < 4 else locs[dest_idx]
        current_dist = abs(taxi_row - target_pos[0]) + abs(taxi_col - target_pos[1])
        prev_row, prev_col = taxi_row, taxi_col
        if action == 0:   prev_row -= 1
        elif action == 1: prev_row += 1
        elif action == 2: prev_col -= 1
        elif action == 3: prev_col += 1
        prev_row = max(0, min(4, prev_row))
        prev_col = max(0, min(4, prev_col))
        prev_dist = abs(prev_row - target_pos[0]) + abs(prev_col - target_pos[1])
        shaping = float(prev_dist - current_dist)
        return env_reward + shaping

    return env_reward

def run_ablation_experiment(num_episodes=1000, seeds=[1]):
    """מריץ מחקר אבלציה על 4 קונפיגורציות, על פני כל ה-seeds, ואוסף 3 מטריקות"""
    configurations = {
        "1. Pure SARSA":          {"planning": False, "kernel": False},
        "2. SARSA + Planning":    {"planning": True,  "kernel": False},
        "3. SARSA + Kernel":      {"planning": False, "kernel": True},
        "4. Full Framework":      {"planning": True,  "kernel": True}
    }

    ablation_master_data = []
    print("\n========================================================")
    print(f"Running Ablation Study over {len(seeds)} seed(s)")
    print("========================================================")

    for seed in seeds:
        # Seed numpy RNG before creating env/agent so all random choices are reproducible
        print(f"\n--- Seed {seed} ---")

        for config_name, flags in configurations.items():
            np.random.seed(seed)
            # Seed numpy RNG before creating env/agent so all random choices are reproducible
            print(f">>> Running configuration: {config_name}...")
            env = gym.make("Taxi-v4", is_rainy=False)
            agent = GeneralizedSweepingSARSAAgent(
                alpha=0.1, gamma=0.95, epsilon=0.1,
                use_planning=flags["planning"], planning_steps=10,
                use_kernel=flags["kernel"], beta=4.0, c=0.5
            )

            for episode in range(num_episodes):
                state, _ = env.reset(seed=seed * 100 + episode)
                action = agent.choose_action(state)
                done = False
                total_env_reward = 0
                illegal_actions_count = 0
                td_errors_in_episode = []

                while not done:
                    next_state, env_reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    custom_reward = get_designed_reward(env_reward, state, agent, action, reward_type="base")
                    if env_reward == -10:
                        illegal_actions_count += 1
                    next_action = agent.choose_action(next_state)
                    td_error = agent.update(state, action, custom_reward, next_state, next_action, done)
                    td_errors_in_episode.append(abs(td_error))
                    state = next_state
                    action = next_action
                    total_env_reward += env_reward

                mean_td_error = np.mean(td_errors_in_episode) if td_errors_in_episode else 0.0
                ablation_master_data.append({
                    "Episode": episode + 1,
                    "Config_Name": config_name,
                    "Seed": seed,
                    "Original_Reward": total_env_reward,
                    "Illegal_Actions": illegal_actions_count,
                    "Mean_Absolute_TD_Error": mean_td_error
                })

            env.close()
            print(f"  - Configuration complete.")

    return pd.DataFrame(ablation_master_data)

def plot_ablation_metrics(df_ablation, n_seeds=1):
    """מייצר גרף משולש (3 מטריקות) עבור מחקר האבלציה עם CI 95% על פני seeds"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.set_theme(style="darkgrid")
    df_smoothed = df_ablation.copy()
    df_smoothed["Original_Reward"] = df_smoothed["Original_Reward"].astype(float)
    df_smoothed["Illegal_Actions"] = df_smoothed["Illegal_Actions"].astype(float)
    df_smoothed["Mean_Absolute_TD_Error"] = df_smoothed["Mean_Absolute_TD_Error"].astype(float)

    # Smooth per (Config_Name, Seed) to avoid rolling across seed boundaries
    for config in df_smoothed["Config_Name"].unique():
        for seed in df_smoothed["Seed"].unique():
            mask = (df_smoothed["Config_Name"] == config) & (df_smoothed["Seed"] == seed)
            for col in ["Original_Reward", "Illegal_Actions", "Mean_Absolute_TD_Error"]:
                df_smoothed.loc[mask, col] = (
                    df_smoothed.loc[mask, col].rolling(window=25, min_periods=1).mean()
                )

    # seaborn aggregates over duplicate (Episode, Config_Name) rows (i.e., across seeds)
    errorbar_arg = ("ci", 95) if n_seeds > 1 else None
    ci_label = f", 95% CI over {n_seeds} seeds" if n_seeds > 1 else f" (seed={df_ablation['Seed'].iloc[0]})"

    sns.lineplot(data=df_smoothed, x="Episode", y="Original_Reward",
                 hue="Config_Name", ax=axes[0], linewidth=2, errorbar=errorbar_arg)
    axes[0].set_title("1. Cumulative Original Reward", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Total Reward (Smoothed)")

    sns.lineplot(data=df_smoothed, x="Episode", y="Illegal_Actions",
                 hue="Config_Name", ax=axes[1], linewidth=2, errorbar=errorbar_arg)
    axes[1].set_title("2. Number of Illegal Actions", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Illegal Actions Per Episode")

    sns.lineplot(data=df_smoothed, x="Episode", y="Mean_Absolute_TD_Error",
                 hue="Config_Name", ax=axes[2], linewidth=2, errorbar=errorbar_arg)
    axes[2].set_title("3. Mean Absolute TD Error", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Absolute TD Error")

    plt.suptitle(
        f"Ablation Study: Analysis of 3 Complementary Metrics{ci_label}",
        fontsize=16, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig("ablation_study_3_metrics.png", dpi=300, bbox_inches='tight')
    print("\n[+] Ablation study graph saved: ablation_study_3_metrics.png")
    plt.show()

def run_reward_experiment(reward_type, num_episodes=1000, seeds=[1]):
    """מריץ את המודל המלא על פני סידים עבור מערכת פרס ספציפית"""
    all_runs_rewards = []
    best_agent = None
    best_score = -np.inf
    print(f"\n>>> Running reward experiment: {REWARD_DISPLAY_NAMES.get(reward_type, reward_type.upper())} over {len(seeds)} seed(s)...")

    for seed in seeds:
        # Seed numpy RNG before creating env/agent for full reproducibility
        np.random.seed(seed)
        env = gym.make("Taxi-v4", is_rainy=False)
        agent = GeneralizedSweepingSARSAAgent(
            alpha=0.1, gamma=0.95, epsilon=0.1,
            use_planning=True, planning_steps=10, use_kernel=True, beta=4.0, c=0.5
        )

        rewards_per_episode = []
        for episode in range(num_episodes):
            state, info = env.reset(seed=seed * 100 + episode)
            action = agent.choose_action(state)
            done = False
            total_env_reward = 0

            while not done:
                next_state, env_reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                custom_reward = get_designed_reward(env_reward, state, agent, action, reward_type=reward_type)
                next_action = agent.choose_action(next_state)
                agent.update(state, action, custom_reward, next_state, next_action, done)
                state = next_state
                action = next_action
                total_env_reward += env_reward

            rewards_per_episode.append(total_env_reward)

        all_runs_rewards.append(rewards_per_episode)
        env.close()

        seed_score = np.mean(rewards_per_episode[-100:])
        print(f"  - Seed {seed} done. Avg reward (last 100 eps): {seed_score:.1f}")

        if seed_score > best_score:
            best_score = seed_score
            best_agent = agent

    return all_runs_rewards, best_agent

def plot_reward_results(df_rewards, n_seeds=1):
    """מייצר את הגרף המשווה בין מערכות הפרס השונות (עם רווח סמך 95% על פני הסידים)"""
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    # CI bands are only meaningful and safe to compute with more than one seed
    errorbar_arg = ("ci", 95) if n_seeds > 1 else None
    sns.lineplot(data=df_rewards, x="Episode", y="Original_Reward", hue="Reward_Type",
                 linewidth=2.5, errorbar=errorbar_arg)
    ci_label = ", 95% CI" if n_seeds > 1 else ""
    plt.title(
        f"Reward Comparison: Original Reward Convergence "
        f"({n_seeds} seed{'s' if n_seeds != 1 else ''}{ci_label}, rolling mean=25)",
        fontsize=15, fontweight='bold'
    )
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Total Original Env Reward (Smoothed)", fontsize=12)
    plt.legend(title="Reward Types", fontsize=10)
    plt.savefig("reward_shaping_results.png", dpi=300, bbox_inches='tight')
    print("\n[+] Reward comparison graph saved: reward_shaping_results.png")
    plt.show()

def run_parameter_sweep(alphas, planning_steps_list, reward_types, seeds, num_episodes=500):
    """Runs a 2D grid search over alpha x planning_steps for each reward_type."""
    results = {}
    total = len(alphas) * len(planning_steps_list) * len(reward_types)
    count = 0

    print(f"\n{'='*60}")
    print(f"2D Parameter Sweep ({total} configs x {len(seeds)} seed(s))")
    print(f"{'='*60}")

    for r_type in reward_types:
        matrix = np.zeros((len(alphas), len(planning_steps_list)))

        for i, alpha in enumerate(alphas):
            for j, n_steps in enumerate(planning_steps_list):
                count += 1
                seed_scores = []

                for seed in seeds:
                    # Seed numpy RNG before creating env/agent for full reproducibility
                    np.random.seed(seed)
                    env = gym.make("Taxi-v4", is_rainy=False)
                    agent = GeneralizedSweepingSARSAAgent(
                        alpha=alpha, gamma=0.95, epsilon=0.1,
                        use_planning=True, planning_steps=n_steps,
                        use_kernel=True, beta=4.0, c=0.5
                    )

                    episode_rewards = []
                    for episode in range(num_episodes):
                        state, _ = env.reset(seed=seed * 100 + episode)
                        action = agent.choose_action(state)
                        done = False
                        total_reward = 0

                        while not done:
                            next_state, env_reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                            custom_reward = get_designed_reward(
                                env_reward, state, agent, action, reward_type=r_type
                            )
                            next_action = agent.choose_action(next_state)
                            agent.update(state, action, custom_reward, next_state, next_action, done)
                            state = next_state
                            action = next_action
                            total_reward += env_reward

                        episode_rewards.append(total_reward)

                    env.close()
                    seed_scores.append(np.mean(episode_rewards[-100:]))

                mean_score = np.mean(seed_scores)
                matrix[i, j] = mean_score
                print(f"  [{count}/{total}] {REWARD_DISPLAY_NAMES.get(r_type, r_type.upper())}: alpha={alpha}, n={n_steps} -> score={mean_score:.2f}")

        results[r_type] = matrix

    return results

def plot_sweep_heatmaps(sweep_results, alphas, planning_steps_list, n_seeds=1):
    """Displays one heatmap per reward type from the 2D parameter sweep."""
    n_plots = len(sweep_results)
    _, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    if n_plots == 1:
        axes = [axes]

    cbar_label = f"Mean Avg Reward — last 100 eps, mean over {n_seeds} seed{'s' if n_seeds != 1 else ''}"

    for ax, (r_type, matrix) in zip(axes, sweep_results.items()):
        df_heat = pd.DataFrame(
            matrix,
            index=[f"a={a}" for a in alphas],
            columns=[f"n={n}" for n in planning_steps_list]
        )
        sns.heatmap(df_heat, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": cbar_label})
        ax.set_title(REWARD_DISPLAY_NAMES.get(r_type, r_type.upper()), fontsize=13, fontweight='bold')
        ax.set_xlabel("Planning Steps (n)", fontsize=11)
        ax.set_ylabel("Learning Rate (alpha)", fontsize=11)

    plt.suptitle(
        f"2D Parameter Sweep: Learning Rate (alpha) x Planning Steps (n)"
        f"  [{n_seeds} seed{'s' if n_seeds != 1 else ''}, mean performance]",
        fontsize=15, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig("parameter_sweep_heatmap.png", dpi=300, bbox_inches='tight')
    print("\n[+] Parameter sweep heatmap saved: parameter_sweep_heatmap.png")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run Taxi RL Experiments")
    parser.add_argument('--fast', action='store_true',
                        help="Quick single-seed run; disables CI bands and uses fewer episodes")
    args = parser.parse_args()

    SEEDS = [1] if args.fast else [1, 2, 3, 4, 5]
    num_episodes   = 500 if args.fast else 1000
    sweep_episodes = 200 if args.fast else 500

    print(f"Running experiments over {len(SEEDS)} seed(s) "
          f"({'fast mode' if args.fast else 'full mode'})")

    # 1. Ablation study over all seeds (CI shown when len(SEEDS) > 1)
    df_ablation = run_ablation_experiment(num_episodes=num_episodes, seeds=SEEDS)
    df_ablation.to_csv("ablation_metrics_results.csv", index=False)
    plot_ablation_metrics(df_ablation, n_seeds=len(SEEDS))

    print("\n[+] Ablation study complete. Moving to reward shaping experiment...")
    print("========================================================\n")

    # 2. Reward shaping comparison across 5 reward systems
    reward_types = ["base", "sparse", "reward_3", "reward_4", "reward_5"]
    reward_master_data = []
    best_base_agent = None

    for r_type in reward_types:
        matrix, agent = run_reward_experiment(
            reward_type=r_type, num_episodes=num_episodes, seeds=SEEDS
        )
        if r_type == "base":
            best_base_agent = agent

        for seed_idx, run_rewards in enumerate(matrix):
            smoothed = pd.Series(run_rewards).rolling(window=25, min_periods=1).mean().values
            for ep_idx, r in enumerate(smoothed):
                reward_master_data.append({
                    "Episode": ep_idx + 1,
                    "Original_Reward": r,
                    "Reward_Type": REWARD_DISPLAY_NAMES[r_type],
                    "Seed": SEEDS[seed_idx]
                })

    df_rewards = pd.DataFrame(reward_master_data)
    df_rewards.to_csv("reward_shaping_metrics.csv", index=False)
    print("\n[+] Reward experiment data saved: reward_shaping_metrics.csv")
    plot_reward_results(df_rewards, n_seeds=len(SEEDS))

    # 3. 2D Parameter Sweep: alpha x planning_steps on base and reward_3
    sweep_results = run_parameter_sweep(
        alphas=[0.01, 0.1, 0.5],
        planning_steps_list=[5, 10, 25],
        reward_types=["base", "reward_3"],
        seeds=SEEDS,
        num_episodes=sweep_episodes
    )
    plot_sweep_heatmaps(sweep_results, [0.01, 0.1, 0.5], [5, 10, 25], n_seeds=len(SEEDS))

    # 4. Policy visualization
    print("\n[+] Visualizing best BASE agent policy...")
    run_and_visualize_policy(best_base_agent, num_episodes=3)

    print("\n[+] All experiments completed successfully!")

if __name__ == "__main__":
    main()
