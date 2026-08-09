"""Evaluate a trained DQN agent on CartPole-v1."""

import gymnasium as gym
import torch

from model import DQN
import csv
import os

def evaluate(
    model_path: str = "dqn_cartpole.pt",
    num_episodes: int = 100,
) -> None:
    """Evaluate a trained DQN using a greedy policy.

    Args:
        model_path: Path to the saved PyTorch model.
        num_episodes: Number of evaluation episodes.
    """
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

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

    rewards = []

    os.makedirs("logs", exist_ok=True)

    evaluation_file = open(
        "logs/evaluation_log.csv",
        "w",
        newline="",
        encoding="utf-8",
    )

    evaluation_writer = csv.writer(evaluation_file)

    evaluation_writer.writerow([
        "episode",
        "reward",
    ])

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()

        state, _ = env.reset()
        print(f"Episode {episode} initial state: {state}")

        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(
                state,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)

            with torch.no_grad():
                q_values = model(state_tensor)

            action = int(
                torch.argmax(q_values, dim=1).item()
            )

            next_state, reward, terminated, truncated, _ = env.step(action)

            done = terminated or truncated

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

        evaluation_writer.writerow([
            episode,
            total_reward,
        ])

        print(
            f"Evaluation Episode {episode:2d}/{num_episodes} | "
            f"Reward: {total_reward:.1f}"
        )

    average_reward = sum(rewards) / len(rewards)

    evaluation_writer.writerow([])
    evaluation_writer.writerow([
        "average_reward",
        average_reward,
    ])
    evaluation_writer.writerow([
        "minimum_reward",
        min(rewards),
    ])
    evaluation_writer.writerow([
        "maximum_reward",
        max(rewards),
    ])

    evaluation_file.close()

    print("\nEvaluation complete.")
    print(f"Average reward: {average_reward:.2f}")
    print(f"Minimum reward: {min(rewards):.1f}")
    print(f"Maximum reward: {max(rewards):.1f}")

    if average_reward >= 400:
        print("Success: average evaluation reward is at least 400.")
    else:
        print("Average evaluation reward is below 400.")

    env.close()


if __name__ == "__main__":
    evaluate()