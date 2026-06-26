import time

import gymnasium as gym
import pygame


def make_visual_taxi_env():
    for render_mode in ("human", "ansi", None):
        env = None
        try:
            kwargs = {}
            if render_mode is not None:
                kwargs["render_mode"] = render_mode

            try:
                env = gym.make("Taxi-v4", is_rainy=False, **kwargs)
            except TypeError:
                env = gym.make("Taxi-v4", **kwargs)

            if render_mode is not None:
                env.reset(seed=0)
                env.render()

            return env, render_mode
        except Exception:
            if env is not None:
                env.close()

    try:
        return gym.make("Taxi-v4", is_rainy=False), None
    except TypeError:
        return gym.make("Taxi-v4"), None


def run_and_visualize_policy(
    agent,
    num_episodes=3,
    max_steps=150,
    delay=0.15,
    start_seed=100,
):
    """Render a greedy policy in Taxi-v4 using Gymnasium's native renderer."""
    env, render_mode = make_visual_taxi_env()
    print(
        f"\n[+] Visualizing policy for {num_episodes} episode(s) "
        f"from seed {start_seed}..."
    )

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        for episode in range(num_episodes):
            state, _ = env.reset(seed=start_seed + episode)
            done = False
            total_reward = 0
            steps = 0
            success = False
            repeated_state_actions = 0
            visited_state_actions = set()

            while not done and steps < max_steps:
                if render_mode == "human":
                    pygame.event.get()
                action = agent.choose_action(state)

                state_action = (state, action)
                if state_action in visited_state_actions:
                    repeated_state_actions += 1
                else:
                    visited_state_actions.add(state_action)

                if repeated_state_actions >= 20:
                    print(f"[-] Policy loop detected at step {steps}.")
                    break

                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                success = success or reward == 20
                state = next_state
                total_reward += reward
                steps += 1

                if render_mode == "human":
                    time.sleep(delay)
                elif render_mode == "ansi":
                    print(env.render())

            status = "SUCCESS" if success else ("STOPPED" if steps >= max_steps else "FAILED_LOOP")
            print(
                f"Visualization episode {episode + 1} | "
                f"steps={steps} | reward={total_reward} | {status}"
            )
            time.sleep(1)

    finally:
        env.close()
        agent.epsilon = original_epsilon
        print("[+] Visualization finished.")
