import os
import gymnasium as gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from agent import GeneralizedSweepingSARSAAgent
from visualize import run_and_visualize_policy

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
        
        # הגדרת המטרה הנוכחית של המונית (הנוסע או תחנת ההורדה)
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
        # עיצוב אישי ג': Potential-Based Shaping (פתרון ללולאות המרחק)
        taxi_row, taxi_col, pass_loc, dest_idx = agent.decode_state(state)
        locs = [(0, 0), (0, 4), (4, 0), (4, 3)] # R, G, Y, B
        target_pos = locs[pass_loc] if pass_loc < 4 else locs[dest_idx]
        
        # 1. מרחק מנהטן הנוכחי (אחרי הצעד)
        current_dist = abs(taxi_row - target_pos[0]) + abs(taxi_col - target_pos[1])
        
        # 2. שחזור המיקום הקודם של המונית לפי הפעולה שנלקחה
        prev_row, prev_col = taxi_row, taxi_col
        if action == 0:   prev_row -= 1  # נסעה דרומה, אז קודם הייתה שורה אחת פחות (למעלה)
        elif action == 1: prev_row += 1  # נסעה צפונה, אז קודם הייתה שורה אחת יותר (למטה)
        elif action == 2: prev_col -= 1  # נסעה מזרחה, אז קודם הייתה עמודה אחת פחות
        elif action == 3: prev_col += 1  # נסעה מערבה, אז קודם הייתה עמודה אחת יותר
        
        # מגבילים את גבולות הגריד (0-4) ליתר ביטחון
        prev_row = max(0, min(4, prev_row))
        prev_col = max(0, min(4, prev_col))
        
        # מרחק מנהטן הקודם (לפני הצעד)
        prev_dist = abs(prev_row - target_pos[0]) + abs(prev_col - target_pos[1])
        
        # חישוב הבונוס/קנס על פי השינוי במרחק (Potential Difference)
        shaping = float(prev_dist - current_dist) # חיובי אם התקרבנו, שלילי אם התרחקנו
        
        # החזרת הפרס המקורי בתוספת רכיב הפוטנציאל
        return env_reward + shaping
        
    return env_reward

def run_ablation_experiment(num_episodes=1000, seed=1):
    """מריץ מחקר אבלציה על 4 קומבינציות ואוסף 3 מטריקות שונות"""
    configurations = {
        "1. Pure SARSA":          {"planning": False, "kernel": False},
        "2. SARSA + Planning":    {"planning": True,  "kernel": False},
        "3. SARSA + Kernel":      {"planning": False, "kernel": True},
        "4. Full Framework":      {"planning": True,  "kernel": True}
    }
    
    ablation_master_data = []
    print("\n========================================================")
    print(f"🧬 מתחיל מחקר אבלציה (Ablation Study) - סיד {seed}")
    print("========================================================")

    for config_name, flags in configurations.items():
        print(f">>> מריץ קומבינציה: {config_name}...")
        env = gym.make("Taxi-v4", is_rainy=False)
        
        agent = GeneralizedSweepingSARSAAgent(
            alpha=0.1, gamma=0.95, epsilon=0.1,
            use_planning=flags["planning"], planning_steps=10, 
            use_kernel=flags["kernel"], beta=4.0, c=0.5
        )

        for episode in range(num_episodes):
            state, info = env.reset(seed=seed * 100 + episode)
            action = agent.choose_action(state)
            done = False
            
            total_env_reward = 0
            illegal_actions_count = 0
            td_errors_in_episode = []

            while not done:
                next_state, env_reward, terminated, truncated, info = env.step(action)
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
                "Original_Reward": total_env_reward,
                "Illegal_Actions": illegal_actions_count,
                "Mean_Absolute_TD_Error": mean_td_error
            })
            
        env.close()
        print(f"  - קומבינציה זו הסתיימה בהצלחה.")

    return pd.DataFrame(ablation_master_data)

def plot_ablation_metrics(df_ablation):
    """מייצר גרף משולש (3 מטריקות) עבור מחקר האבלציה"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.set_theme(style="darkgrid")
    
    df_smoothed = df_ablation.copy()
    
    # 🔥 התיקון: המרת סוגי הנתונים ל-float מראש כדי למנוע את שגיאת ה-LossySetitemError
    df_smoothed["Original_Reward"] = df_smoothed["Original_Reward"].astype(float)
    df_smoothed["Illegal_Actions"] = df_smoothed["Illegal_Actions"].astype(float)
    df_smoothed["Mean_Absolute_TD_Error"] = df_smoothed["Mean_Absolute_TD_Error"].astype(float)

    for config in df_smoothed["Config_Name"].unique():
        mask = df_smoothed["Config_Name"] == config
        df_smoothed.loc[mask, "Original_Reward"] = df_smoothed.loc[mask, "Original_Reward"].rolling(window=25, min_periods=1).mean()
        df_smoothed.loc[mask, "Illegal_Actions"] = df_smoothed.loc[mask, "Illegal_Actions"].rolling(window=25, min_periods=1).mean()
        df_smoothed.loc[mask, "Mean_Absolute_TD_Error"] = df_smoothed.loc[mask, "Mean_Absolute_TD_Error"].rolling(window=25, min_periods=1).mean()

    # ציור מטריקה 1
    sns.lineplot(data=df_smoothed, x="Episode", y="Original_Reward", hue="Config_Name", ax=axes[0], linewidth=2)
    axes[0].set_title("1. Cumulative Original Reward", fontsize=12, fontweight='bold')
    axes[0].set_ylabel("Total Reward (Smoothed)")
    
    # ציור מטריקה 2
    sns.lineplot(data=df_smoothed, x="Episode", y="Illegal_Actions", hue="Config_Name", ax=axes[1], linewidth=2)
    axes[1].set_title("2. Number of Illegal Actions", fontsize=12, fontweight='bold')
    axes[1].set_ylabel("Illegal Actions Per Episode")
    
    # ציור מטריקה 3
    sns.lineplot(data=df_smoothed, x="Episode", y="Mean_Absolute_TD_Error", hue="Config_Name", ax=axes[2], linewidth=2)
    axes[2].set_title("3. Mean Absolute TD Error", fontsize=12, fontweight='bold')
    axes[2].set_ylabel("Absolute TD Error")

    plt.suptitle("Ablation Study Deep Dive: Analysis of 3 Complementary Metrics", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("ablation_study_3_metrics.png", dpi=300, bbox_inches='tight')
    print("\n[+] הגרף המשולש של האבלציה נשמר בהצלחה: ablation_study_3_metrics.png")
    plt.show()

def run_reward_experiment(reward_type, num_episodes=1000, seeds=[1]):
    """מריץ את המודל המלא על פני סידים עבור מערכת פרס ספציפית"""
    all_runs_rewards = []
    best_agent = None
    best_score = -np.inf
    print(f"\n>>> מריץ ניסוי מערכת פרס: {reward_type.upper()} על פני {len(seeds)} סידים...")

    for seed in seeds:
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
        print(f"  - סיד {seed} הסתיים. פרס מקורי ממוצע ב-100 הפרקים האחרונים: {seed_score:.1f}")

        if seed_score > best_score:
            best_score = seed_score
            best_agent = agent

    return all_runs_rewards, best_agent

def plot_reward_results(df_rewards, n_seeds=1):
    """מייצר את הגרף המשווה בין מערכות הפרס השונות (עם רווח סמך 95% על פני הסידים)"""
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="darkgrid")
    
    # errorbar=("ci", 95): seaborn מפיק רווח סמך 95% אוטומטית כאשר יש כמה סידים לאותו Episode
    sns.lineplot(data=df_rewards, x="Episode", y="Original_Reward", hue="Reward_Type",
                 linewidth=2.5, errorbar=("ci", 95))
    plt.title(f"Reward Comparison: Original Reward Convergence "
              f"({n_seeds} seed{'s' if n_seeds != 1 else ''}, 95% CI, rolling mean=25)",
              fontsize=15, fontweight='bold')
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel("Total Original Env Reward (Smoothed)", fontsize=12)
    plt.legend(title="Reward Types", fontsize=10)
    
    plt.savefig("reward_shaping_results.png", dpi=300, bbox_inches='tight')
    print("\n[+] גרף השוואת הפרסים נשמר בהצלחה בשם: reward_shaping_results.png")
    plt.show()

def main():
    # 1. הרצת מחקר האבלציה המקיף (גרף שלשה)
    df_ablation = run_ablation_experiment(num_episodes=1000, seed=1)
    df_ablation.to_csv("ablation_metrics_results.csv", index=False)
    plot_ablation_metrics(df_ablation)
    
    print("\n[+] מחקר האבלציה הסתיים בהצלחה. ממשיך לניסוי השני של עיצוב הפרסים...")
    print("========================================================\n")

    # 2. הרצת הניסוי של ה-Reward Shaping (5 מערכות פרס)
    reward_types = ["base", "sparse", "reward_3", "reward_4", "reward_5"]
    num_episodes = 1000

    # ⚠️י.
    # seeds = [1, 2, 3, 4, 5]
    seeds = [1]
    
    reward_master_data = []
    best_base_agent = None

    for r_type in reward_types:
        matrix, agent = run_reward_experiment(reward_type=r_type, num_episodes=num_episodes, seeds=seeds)

        if r_type == "base":
            best_base_agent = agent

        for seed_idx, run_rewards in enumerate(matrix):
            smoothed = pd.Series(run_rewards).rolling(window=25, min_periods=1).mean().values
            for ep_idx, r in enumerate(smoothed):
                reward_master_data.append({
                    "Episode": ep_idx + 1,
                    "Original_Reward": r,
                    "Reward_Type": r_type.upper(),
                    "Seed": seeds[seed_idx]
                })
                
    df_rewards = pd.DataFrame(reward_master_data)
    df_rewards.to_csv("reward_shaping_metrics.csv", index=False)
    print("\n[+] כל נתוני הניסוי נשמרו בהצלחה ל-reward_shaping_metrics.csv")
    
    plot_reward_results(df_rewards, n_seeds=len(seeds))

    # 3. הצגה ויזואלית
    print("\n[+] מפעיל תצוגה ויזואלית של הסוכן הטוב ביותר (BASE)...")
    run_and_visualize_policy(best_base_agent, num_episodes=3)

    print("\n[+] כל הקוד והניסויים של הפרויקט הסתיימו בהצלחה מלאה!")

if __name__ == "__main__":
    main()