"""Evaluate a trained Deep Q-Network on CartPole-v1."""

import csv
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import torch

from model import DQN


@dataclass(frozen=True)
class EvaluationResult:
    """Store summary statistics from model evaluation."""

    average_reward: float
    minimum_reward: float
    maximum_reward: float


def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_model(
    env: gym.Env,
    device: torch.device,
    model_path: str,
) -> DQN:
    """Load a trained DQN checkpoint for the environment."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    model = DQN(
        state_size=state_size,
        action_size=action_size,
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
        )
    )

    model.eval()

    return model


def select_greedy_action(
    model: DQN,
    state,
    device: torch.device,
) -> int:
    """Select the action with the highest predicted Q-value."""
    state_tensor = torch.as_tensor(
        state,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        q_values = model(state_tensor)

    return int(torch.argmax(q_values, dim=1).item())


def run_episode(
    env: gym.Env,
    model: DQN,
    device: torch.device,
) -> float:
    """Run one greedy evaluation episode and return its reward."""
    state, _ = env.reset()

    total_reward = 0.0
    done = False

    while not done:
        action = select_greedy_action(
            model=model,
            state=state,
            device=device,
        )

        state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated
        total_reward += reward

    return total_reward


def save_evaluation_log(
    rewards: list[float],
    result: EvaluationResult,
) -> None:
    """Write episode rewards and summary statistics to CSV."""
    log_directory = Path("logs")
    log_directory.mkdir(exist_ok=True)

    log_path = log_directory / "evaluation_log.csv"

    with log_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as log_file:
        writer = csv.writer(log_file)

        writer.writerow([
            "episode",
            "reward",
        ])

        for episode, reward in enumerate(rewards, start=1):
            writer.writerow([
                episode,
                reward,
            ])

        writer.writerow([])
        writer.writerow([
            "metric",
            "value",
        ])
        writer.writerow([
            "average_reward",
            result.average_reward,
        ])
        writer.writerow([
            "minimum_reward",
            result.minimum_reward,
        ])
        writer.writerow([
            "maximum_reward",
            result.maximum_reward,
        ])


def summarize_rewards(
    rewards: list[float],
) -> EvaluationResult:
    """Calculate summary statistics for evaluation rewards."""
    return EvaluationResult(
        average_reward=sum(rewards) / len(rewards),
        minimum_reward=min(rewards),
        maximum_reward=max(rewards),
    )


def print_summary(result: EvaluationResult) -> None:
    """Print final evaluation statistics."""
    print("\nEvaluation complete.")
    print(f"Average reward: {result.average_reward:.2f}")
    print(f"Minimum reward: {result.minimum_reward:.1f}")
    print(f"Maximum reward: {result.maximum_reward:.1f}")

    if result.average_reward >= 400:
        print(
            "Success: average evaluation reward "
            "is at least 400."
        )
    else:
        print("Average evaluation reward is below 400.")

    print(
        "Evaluation log saved to "
        "logs/evaluation_log.csv"
    )


def evaluate(
    model_path: str = "dqn_cartpole.pt",
    num_episodes: int = 100,
) -> None:
    """Evaluate a saved DQN using a purely greedy policy."""
    device = get_device()
    env = gym.make("CartPole-v1")

    print(f"Using device: {device}")

    model = load_model(
        env=env,
        device=device,
        model_path=model_path,
    )

    rewards = []

    for episode in range(1, num_episodes + 1):
        reward = run_episode(
            env=env,
            model=model,
            device=device,
        )

        rewards.append(reward)

        print(
            f"Evaluation Episode "
            f"{episode:3d}/{num_episodes} | "
            f"Reward: {reward:.1f}"
        )

    result = summarize_rewards(rewards)

    save_evaluation_log(
        rewards=rewards,
        result=result,
    )

    print_summary(result)

    env.close()


if __name__ == "__main__":
    evaluate()
