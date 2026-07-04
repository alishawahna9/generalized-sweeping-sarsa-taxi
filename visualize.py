import time

import gymnasium as gym
import numpy as np


TAXI_SUCCESS_REWARD = 20
LOOP_REPEAT_LIMIT = 20


def make_visualization_env(render_mode=None):
    kwargs = {"is_rainy": False}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    try:
        return gym.make("Taxi-v4", **kwargs)
    except TypeError:
        kwargs.pop("is_rainy", None)
        return gym.make("Taxi-v4", **kwargs)


def make_render_env():
    last_error = None
    for render_mode in ("human", "ansi", None):
        env = None
        try:
            env = make_visualization_env(render_mode=render_mode)
            env.reset(seed=0)
            if render_mode is not None:
                env.render()
            return env, render_mode
        except Exception as exc:
            last_error = exc
            if env is not None:
                env.close()

    raise RuntimeError("Could not create a Taxi-v4 render environment.") from last_error


def _greedy_action(agent, state):
    action_values = agent.q_table[int(state)]
    best_value = np.max(action_values)
    best_actions = np.flatnonzero(np.isclose(action_values, best_value))
    return int(np.random.choice(best_actions))


def _rollout(env, agent, seed, max_steps):
    original_random_state = np.random.get_state()
    np.random.seed(seed)
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    success = False
    repeated_state_actions = 0
    visited_state_actions = set()
    steps = 0

    try:
        for step in range(1, max_steps + 1):
            action = _greedy_action(agent, state)
            state_action = (int(state), int(action))

            if state_action in visited_state_actions:
                repeated_state_actions += 1
            else:
                visited_state_actions.add(state_action)

            if repeated_state_actions >= LOOP_REPEAT_LIMIT:
                return total_reward, success, step - 1, "FAILED_LOOP"

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps = step

            if terminated:
                success = reward == TAXI_SUCCESS_REWARD
                status = "SUCCESS" if success else "FINISHED"
                return total_reward, success, steps, status
            if truncated:
                return total_reward, success, steps, "TRUNCATED"

        return total_reward, success, steps, "MAX_STEPS"
    finally:
        np.random.set_state(original_random_state)


def _score_candidates(agent, start_seed, candidate_count, max_steps):
    candidate_count = max(candidate_count, 1)
    env = make_visualization_env()
    try:
        scored = []
        for offset in range(candidate_count):
            seed = start_seed + offset
            total_reward, success, steps, status = _rollout(env, agent, seed, max_steps)
            scored.append(
                (success, total_reward, -steps, status != "FAILED_LOOP", seed, status)
            )
    finally:
        env.close()

    scored.sort(key=lambda item: item[:5], reverse=True)
    return scored


def run_and_visualize_policy(
    agent,
    num_episodes=1,
    max_steps=150,
    delay=0.10,
    start_seed=0,
    candidate_count=100,
):
    """Score candidate seeds greedily, then render the best episodes in a pygame window."""
    num_episodes = max(num_episodes, 1)
    scored = _score_candidates(agent, start_seed, candidate_count, max_steps)
    best = scored[:num_episodes]

    print(
        f"[+] Scored {max(candidate_count, 1)} candidate start seed(s); "
        f"rendering the best {len(best)} episode(s)."
    )

    env, render_mode = make_render_env()
    original_random_state = np.random.get_state()
    try:
        for rank, (success, total_reward, neg_steps, _, seed, status) in enumerate(
            best,
            start=1,
        ):
            steps = -neg_steps
            print(
                f"[+] Rendering episode {rank}/{len(best)} "
                f"(seed={seed}, status={status}, steps={steps}, "
                f"reward={total_reward:.1f}, render={render_mode})"
            )
            np.random.seed(seed)
            state, _ = env.reset(seed=seed)
            if render_mode == "ansi":
                print(env.render())
            elif render_mode == "human":
                env.render()
            time.sleep(delay)

            repeated_state_actions = 0
            visited_state_actions = set()

            for step in range(1, max_steps + 1):
                action = _greedy_action(agent, state)
                state_action = (int(state), int(action))

                if state_action in visited_state_actions:
                    repeated_state_actions += 1
                else:
                    visited_state_actions.add(state_action)

                if repeated_state_actions >= LOOP_REPEAT_LIMIT:
                    print(f"[-] Policy loop detected at step {step - 1}.")
                    break

                state, reward, terminated, truncated, _ = env.step(action)
                if render_mode == "ansi":
                    print(env.render())
                elif render_mode == "human":
                    env.render()
                time.sleep(delay)

                if terminated or truncated:
                    break
    finally:
        np.random.set_state(original_random_state)
        env.close()
