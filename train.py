"""Train a Deep Q-Network on CartPole-v1."""

import csv
import os
from collections import deque

import gymnasium as gym
import torch

from agent import DQNAgent


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int = 10,
) -> float:
    """Evaluate the agent using a purely greedy policy."""
    eval_env = gym.make("CartPole-v1")

    rewards = []

    for _ in range(num_episodes):
        state, _ = eval_env.reset()

        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(
                state=state,
                epsilon=0.0,
            )

            next_state, reward, terminated, truncated, _ = eval_env.step(action)

            done = terminated or truncated

            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    eval_env.close()

    return sum(rewards) / len(rewards)


def train() -> DQNAgent:
    """Train and return a DQN agent."""
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        learning_rate=1e-3,
        gamma=0.99,
        batch_size=64,
        buffer_size=100_000,
    )

    num_episodes = 500

    warmup_steps = 1000

    target_update_frequency = 3

    evaluation_frequency = 25
    evaluation_episodes = 10

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    recent_rewards = deque(maxlen=100)

    total_steps = 0

    best_eval_reward = float("-inf")
    best_eval_episode = 0

    os.makedirs("logs", exist_ok=True)

    log_file = open(
        "logs/training_log.csv",
        "w",
        newline="",
        encoding="utf-8",
    )

    log_writer = csv.writer(log_file)

    log_writer.writerow([
        "episode",
        "reward",
        "average_reward",
        "average_loss",
        "epsilon",
        "buffer_size",
        "evaluation_reward",
    ])

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()

        done = False
        episode_reward = 0.0
        episode_losses = []

        while not done:
            action = agent.select_action(
                state=state,
                epsilon=epsilon,
            )

            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                info=info,
            )

            state = next_state
            episode_reward += reward
            total_steps += 1

            if (
                total_steps >= warmup_steps
                and len(agent.replay_buffer) >= agent.batch_size
            ):
                loss = agent.learn()
                episode_losses.append(loss)

        recent_rewards.append(episode_reward)

        average_reward = sum(recent_rewards) / len(recent_rewards)

        if episode_losses:
            average_loss = sum(episode_losses) / len(episode_losses)
        else:
            average_loss = 0.0

        if episode % target_update_frequency == 0:
            agent.update_target_network()

        epsilon = max(
            epsilon_min,
            epsilon * epsilon_decay,
        )

        evaluation_reward = ""

        if episode % evaluation_frequency == 0:
            evaluation_reward = evaluate_agent(
                agent=agent,
                num_episodes=evaluation_episodes,
            )

            print(
                f"*** Greedy Evaluation at Episode {episode}: "
                f"Average Reward = {evaluation_reward:.2f} ***"
            )

            if evaluation_reward > best_eval_reward:
                best_eval_reward = evaluation_reward
                best_eval_episode = episode

                torch.save(
                    agent.q_network.state_dict(),
                    "dqn_cartpole.pt",
                )

                print(
                    f"*** New best model saved at Episode {episode} "
                    f"with Evaluation Reward = "
                    f"{evaluation_reward:.2f} ***"
                )

        print(
            f"Episode {episode:3d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Avg Reward: {average_reward:6.1f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Buffer: {len(agent.replay_buffer)}"
        )

        log_writer.writerow([
            episode,
            episode_reward,
            average_reward,
            average_loss,
            epsilon,
            len(agent.replay_buffer),
            evaluation_reward,
        ])

        log_file.flush()

    print(
        f"\nTraining complete. "
        f"Best evaluation reward: {best_eval_reward:.2f} "
        f"at episode {best_eval_episode}"
    )

    print("Best model saved to dqn_cartpole.pt")

    env.close()
    log_file.close()

    return agent


if __name__ == "__main__":
    train()
