import time

import gymnasium as gym
import numpy as np

from main import REWARD_DISPLAY_NAMES, load_saved_model


SEEDS = [1, 2, 3, 4, 5]
REWARD_TYPES = ["base", "sparse", "reward_3", "reward_4", "reward_5"]

MAX_STEPS = 60
STEP_DELAY_SECONDS = 0.35
LOOP_REPEAT_LIMIT = 4
PLANNING_STEPS = 5


def make_env():
    try:
        return gym.make("Taxi-v4", is_rainy=False, render_mode="human")
    except TypeError:
        return gym.make("Taxi-v4", render_mode="human")


def run_one_episode(agent, seed, reward_type):
    env = make_env()
    observation, _ = env.reset(seed=seed)

    total_reward = 0
    repeated_state_actions = 0
    visited_state_actions = set()

    try:
        for step in range(1, MAX_STEPS + 1):
            action = agent.choose_action(observation)
            state_action = (int(observation), int(action))

            if state_action in visited_state_actions:
                repeated_state_actions += 1
            else:
                visited_state_actions.add(state_action)

            if repeated_state_actions >= LOOP_REPEAT_LIMIT:
                print(
                    f"[STOP] Loop detected | seed={seed} | reward={reward_type} | "
                    f"step={step} | state={observation} | action={action}"
                )
                return "FAILED_LOOP", step - 1, total_reward

            next_observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            print(
                f"  step={step:02d} | state={observation:3d} | action={action} | "
                f"reward={reward:3d} | next={next_observation:3d}"
            )

            observation = next_observation
            time.sleep(STEP_DELAY_SECONDS)

            if terminated or truncated:
                status = "SUCCESS" if reward == 20 else "FINISHED"
                return status, step, total_reward

        return "MAX_STEPS", MAX_STEPS, total_reward
    finally:
        env.close()


def main():
    total_runs = len(SEEDS) * len(REWARD_TYPES)
    run_number = 0

    for seed in SEEDS:
        for reward_type in REWARD_TYPES:
            run_number += 1
            reward_name = REWARD_DISPLAY_NAMES[reward_type]

            print("\n" + "=" * 72)
            print(
                f"[RUN {run_number}/{total_runs}] "
                f"seed={seed} | reward={reward_type} ({reward_name})"
            )

            agent, model_path = load_saved_model(
                output_dir="results",
                agent_name="full",
                reward_type=reward_type,
                seed=seed,
                planning_steps=PLANNING_STEPS,
            )

            agent.epsilon = 0.0
            np.random.seed(seed)

            print(f"[MODEL] {model_path}")
            print(
                f"[CONFIG] max_steps={MAX_STEPS}, "
                f"delay={STEP_DELAY_SECONDS}s, loop_limit={LOOP_REPEAT_LIMIT}"
            )

            status, steps, total_reward = run_one_episode(agent, seed, reward_type)

            print(
                f"[RESULT] seed={seed} | reward={reward_type} | "
                f"steps={steps} | total_reward={total_reward} | {status}"
            )


if __name__ == "__main__":
    main()
