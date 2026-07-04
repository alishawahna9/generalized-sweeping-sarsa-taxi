"""Generate policy animation GIFs from saved Q-tables.

Loads the trained "full" framework models saved under results/models by
main.py, runs one greedy episode per reward design with
render_mode="rgb_array", and saves the frames as a GIF under
reports/policy_animations/. By default it renders a fixed regular episode
instead of searching for the easiest successful start state.

Usage:
    python make_policy_gifs.py
    python make_policy_gifs.py --reward base --seed 1
    python make_policy_gifs.py --search-best --start-candidates 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def find_project_root(start_path):
    """Find the repository root whether this script is in root/ or tools/."""
    candidates = [start_path.parent, *start_path.parents]
    for candidate in candidates:
        if (candidate / "main.py").exists() and (candidate / "agent.py").exists():
            return candidate
    raise RuntimeError(
        "Could not find the project root. Run this script from inside the "
        "generalized-sweeping-sarsa-taxi project."
    )


try:
    from PIL import Image
except ImportError as exc:
    raise SystemExit(
        "Missing GIF dependency: Pillow. Install development dependencies with "
        "`pip install -r requirements-dev.txt`."
    ) from exc


try:
    ROOT = find_project_root(Path(__file__).resolve())
except RuntimeError as exc:
    raise SystemExit(str(exc)) from exc
sys.path.insert(0, str(ROOT))

from main import (  # noqa: E402
    REWARD_DISPLAY_NAMES,
    TAXI_SUCCESS_REWARD,
    load_saved_model,
    make_taxi_env,
)

OUTPUT_DIR = ROOT / "reports" / "policy_animations"
MODEL_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_START_SEED = 1
DEFAULT_START_CANDIDATES = 1
MAX_STEPS = 150
LOOP_REPEAT_LIMIT = 20
FRAME_DURATION_MS = 350


def run_greedy_episode(env, agent, start_seed, max_steps=MAX_STEPS, capture_frames=True):
    """Run one greedy episode and return episode metadata plus frames."""
    original_epsilon = agent.epsilon
    original_random_state = np.random.get_state()
    agent.epsilon = 0.0
    np.random.seed(start_seed)
    frames = []

    try:
        state, _ = env.reset(seed=start_seed)
        if capture_frames:
            frames.append(env.render())
        done = False
        steps = 0
        success = False
        total_reward = 0.0
        repeated_state_actions = 0
        visited_state_actions = set()

        while not done and steps < max_steps:
            action = agent.choose_action(state)

            state_action = (int(state), int(action))
            if state_action in visited_state_actions:
                repeated_state_actions += 1
            else:
                visited_state_actions.add(state_action)

            if repeated_state_actions >= LOOP_REPEAT_LIMIT:
                break

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            success = success or reward == TAXI_SUCCESS_REWARD
            if capture_frames:
                frames.append(env.render())
            steps += 1
    finally:
        agent.epsilon = original_epsilon
        np.random.set_state(original_random_state)

    status = "SUCCESS" if success else (
        "MAX_STEPS" if steps >= max_steps else "FAILED_LOOP"
    )
    return {
        "frames": frames,
        "success": success,
        "steps": steps,
        "status": status,
        "start_seed": start_seed,
        "total_reward": total_reward,
    }


def save_gif(frames, file_path):
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        file_path,
        save_all=True,
        append_images=images[1:],
        duration=FRAME_DURATION_MS,
        loop=0,
    )


def rank_episode(result):
    return (
        int(result["success"]),
        result["total_reward"],
        -result["steps"],
        int(result["status"] != "FAILED_LOOP"),
    )


def candidate_start_seeds(start_seed, count):
    count = max(1, count)
    return range(start_seed, start_seed + count)


def make_gif_for_reward(
    reward_type,
    planning_steps,
    model_seeds,
    start_seed,
    start_candidates,
    save_best_attempt,
    search_best,
):
    env = make_taxi_env(render_mode="rgb_array")
    best_attempt = None
    loaded_models = 0

    try:
        for model_seed in model_seeds:
            try:
                agent, model_path = load_saved_model(
                    ROOT / "results",
                    "full",
                    reward_type,
                    model_seed,
                    planning_steps,
                )
            except FileNotFoundError as exc:
                print(f"[-] {reward_type}: missing model for seed {model_seed}: {exc}")
                continue

            loaded_models += 1

            if not search_best:
                result = run_greedy_episode(
                    env,
                    agent,
                    start_seed,
                    capture_frames=True,
                )
                file_name = (
                    f"{reward_type}_policy.gif"
                    if result["success"]
                    else f"{reward_type}_regular_attempt.gif"
                )
                file_path = OUTPUT_DIR / file_name

                if result["success"] or save_best_attempt:
                    save_gif(result["frames"], file_path)
                    prefix = "[+]" if result["success"] else "[!]"
                    print(
                        f"{prefix} {reward_type}: saved regular episode "
                        f"{file_path.name} "
                        f"(model={model_path.name}, model_seed={model_seed}, "
                        f"start_seed={result['start_seed']}, "
                        f"steps={result['steps']}, "
                        f"reward={result['total_reward']:.1f}, "
                        f"status={result['status']})"
                    )
                    return True

                print(
                    f"[-] {reward_type}: regular episode failed "
                    f"(model={model_path.name}, model_seed={model_seed}, "
                    f"start_seed={result['start_seed']}, "
                    f"status={result['status']})."
                )
                return False

            for candidate_seed in candidate_start_seeds(start_seed, start_candidates):
                result = run_greedy_episode(
                    env,
                    agent,
                    candidate_seed,
                    capture_frames=False,
                )
                result["model_path"] = model_path
                result["model_seed"] = model_seed
                result["agent"] = agent

                if best_attempt is None or rank_episode(result) > rank_episode(best_attempt):
                    best_attempt = result

        if loaded_models == 0:
            print(f"[-] {reward_type}: no saved models found under {ROOT / 'results' / 'models'}.")
            return False

        if best_attempt is not None and (best_attempt["success"] or save_best_attempt):
            file_name = (
                f"{reward_type}_policy.gif"
                if best_attempt["success"]
                else f"{reward_type}_best_attempt.gif"
            )
            file_path = OUTPUT_DIR / file_name
            rendered = run_greedy_episode(
                env,
                best_attempt["agent"],
                best_attempt["start_seed"],
                capture_frames=True,
            )
            save_gif(rendered["frames"], file_path)

            prefix = "[+]" if best_attempt["success"] else "[!]"
            summary = "saved" if best_attempt["success"] else "no successful episode found; saved"
            print(
                f"{prefix} {reward_type}: {summary} {file_path.name} "
                f"(model={best_attempt['model_path'].name}, "
                f"model_seed={best_attempt['model_seed']}, "
                f"start_seed={best_attempt['start_seed']}, "
                f"steps={best_attempt['steps']}, "
                f"reward={best_attempt['total_reward']:.1f}, "
                f"status={best_attempt['status']})"
            )
            return True

        print(
            f"[-] {reward_type}: no successful greedy episode found after "
            f"{loaded_models * max(1, start_candidates)} candidate run(s)."
        )
        return False
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate policy GIFs from saved models in results/models."
    )
    parser.add_argument(
        "--reward",
        choices=list(REWARD_DISPLAY_NAMES),
        default=None,
        help="Generate a GIF only for this reward design (default: all).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Use only this saved model seed (default: try seeds 1-5).",
    )
    parser.add_argument("--planning-steps", type=int, default=5)
    parser.add_argument("--start-seed", type=int, default=DEFAULT_START_SEED)
    parser.add_argument(
        "--start-candidates",
        type=int,
        default=DEFAULT_START_CANDIDATES,
        help="Number of start seeds to try with --search-best.",
    )
    parser.add_argument(
        "--search-best",
        action="store_true",
        help="Search model/start seed candidates and render the best episode.",
    )
    parser.add_argument(
        "--no-best-attempt",
        action="store_true",
        help="Do not save an attempt GIF when no successful episode is found.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    reward_types = [args.reward] if args.reward else list(REWARD_DISPLAY_NAMES)
    if args.seed:
        model_seeds = [args.seed]
    elif args.search_best:
        model_seeds = MODEL_SEEDS
    else:
        model_seeds = [MODEL_SEEDS[0]]

    print(f"[+] Project root: {ROOT}")
    print(f"[+] Models directory: {ROOT / 'results' / 'models'}")
    print(f"[+] Output directory: {OUTPUT_DIR}")
    print(
        "[+] Selection mode: "
        f"{'search best' if args.search_best else 'regular fixed episode'}"
    )

    results = {
        reward_type: make_gif_for_reward(
            reward_type,
            args.planning_steps,
            model_seeds,
            args.start_seed,
            args.start_candidates,
            not args.no_best_attempt,
            args.search_best,
        )
        for reward_type in reward_types
    }

    failed = [reward for reward, ok in results.items() if not ok]
    if failed:
        print(f"\n[-] Failed rewards: {failed}")
        sys.exit(1)
    print("\n[+] All requested GIFs generated.")


if __name__ == "__main__":
    main()
