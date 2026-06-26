import argparse
from pathlib import Path

import numpy as np

from agent import GeneralizedSweepingSARSAAgent
from main import (
    REWARD_DISPLAY_NAMES,
    TAXI_LOCS,
    calculate_custom_reward,
    calculate_epsilon_decay,
    load_saved_model,
    make_taxi_env,
    train_agent,
    visualization_agent_configs,
    was_wall_bump,
)

ACTIONS = {
    0: "south",
    1: "north",
    2: "east",
    3: "west",
    4: "pickup",
    5: "dropoff",
}


def format_state(agent, state):
    row, col, passenger_loc, dest_idx = agent.decode_state(state)
    passenger_name = "in_taxi" if passenger_loc == 4 else "RGYB"[passenger_loc]
    destination_name = "RGYB"[dest_idx]
    return (
        f"state={state} | taxi=({row},{col}) | "
        f"passenger={passenger_name} | destination={destination_name}"
    )


def current_target_name(agent, state):
    _, _, passenger_loc, dest_idx = agent.decode_state(state)
    if passenger_loc < 4:
        return f"pickup at {'RGYB'[passenger_loc]} {TAXI_LOCS[passenger_loc]}"
    return f"dropoff at {'RGYB'[dest_idx]} {TAXI_LOCS[dest_idx]}"


def explain_kernel(agent, state, action):
    if not agent.use_kernel or action not in (0, 1, 2, 3):
        return "Kernel: off for this step."

    state_ids, weights = agent.kernel_neighbors[state]
    if state_ids.size == 0:
        return "Kernel: no neighbors available."

    parts = []
    for neighbor_state, weight in zip(state_ids[:3], weights[:3]):
        row, col, passenger_loc, dest_idx = agent.decode_state(int(neighbor_state))
        parts.append(
            f"{int(neighbor_state)}->({row},{col},p={passenger_loc},d={dest_idx}) w={weight:.3f}"
        )
    return "Kernel neighbors: " + ", ".join(parts)


def build_agent(args):
    if args.load_saved:
        agent, source = load_saved_model(
            args.output_dir,
            args.agent,
            args.reward,
            args.seed,
            args.planning_steps,
        )
        print(f"Loaded saved model: {source}")
        return agent

    config = visualization_agent_configs(args.planning_steps)[args.agent].copy()
    config["epsilon_decay"] = calculate_epsilon_decay(args.train_episodes)
    agent = GeneralizedSweepingSARSAAgent(**config)
    env = make_taxi_env()
    try:
        train_agent(
            env,
            agent,
            num_episodes=args.train_episodes,
            reward_type=args.reward,
            seed=args.seed,
        )
    finally:
        env.close()
    print(
        "Trained fresh model for explanation: "
        f"agent={args.agent}, reward={args.reward}, episodes={args.train_episodes}, seed={args.seed}"
    )
    return agent


def explain_episode(agent, args):
    env = make_taxi_env(render_mode="ansi")
    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    try:
        state, _ = env.reset(seed=args.seed)
        action = agent.choose_action(state)
        done = False
        step_idx = 0

        print("\n=== Explained Episode Start ===")
        print(format_state(agent, state))
        print(f"Target now: {current_target_name(agent, state)}")
        print(env.render())

        while not done and step_idx < args.max_steps:
            step_idx += 1
            q_current = agent.q_table[state, action]
            model_size_before = len(agent.model)
            pq_before = len(agent.pq)

            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            train_reward = calculate_custom_reward(
                args.reward,
                env_reward,
                state,
                action,
                next_state,
                agent,
            )
            next_action = agent.choose_action(next_state)
            q_next = 0.0 if done else agent.q_table[next_state, next_action]
            td_error = train_reward + agent.gamma * q_next - q_current

            print(f"\nStep {step_idx}")
            print(f"Action: {ACTIONS[action]} ({action})")
            print(f"Before update: Q(s,a)={q_current:.3f}")
            print(f"Environment reward={env_reward:.3f} | Training reward={train_reward:.3f}")
            print(f"Next action: {ACTIONS[next_action]} ({next_action}) | Q(s',a')={q_next:.3f}")
            print(f"SARSA td_error = r + gamma*Q(s',a') - Q(s,a) = {td_error:.3f}")
            print(explain_kernel(agent, state, action))

            agent.learn(state, action, train_reward, next_state, next_action, done)

            q_after = agent.q_table[state, action]
            print(f"After update: Q(s,a)={q_after:.3f}")
            print(
                "Planning/model status: "
                f"model {model_size_before}->{len(agent.model)}, "
                f"priority queue {pq_before}->{len(agent.pq)}"
            )
            if was_wall_bump(state, action, next_state):
                print("Movement note: the taxi tried to move into a wall and stayed put.")

            print(format_state(agent, next_state))
            print(f"Target now: {current_target_name(agent, next_state)}")
            print(env.render())

            state = next_state
            action = next_action

        status = "success" if env_reward == 20 else "finished_without_dropoff"
        print(f"\n=== Episode End: {status} after {step_idx} step(s) ===")
    finally:
        agent.epsilon = original_epsilon
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Explain one Taxi-v4 episode step by step, including SARSA/planning/kernel details."
    )
    parser.add_argument(
        "--agent",
        choices=["pure_sarsa", "planning", "kernel", "full"],
        default="full",
    )
    parser.add_argument(
        "--reward",
        choices=list(REWARD_DISPLAY_NAMES),
        default="base",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--planning-steps", type=int, default=5)
    parser.add_argument("--train-episodes", type=int, default=150)
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--load-saved", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    return parser.parse_args()


def main():
    args = parse_args()
    agent = build_agent(args)
    explain_episode(agent, args)


if __name__ == "__main__":
    main()
