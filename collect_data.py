"""Collect CartPole experiences using an epsilon-greedy DQN agent."""

import gymnasium as gym
import numpy as np
import torch

from agent import DQNAgent
from replay_buffer import Experience


NUM_EPISODES = 10
EPSILON = 1.0


def step_and_store(
    env: gym.Env,
    agent: DQNAgent,
    state: np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, float, bool]:
    """Take one action, store the transition, and return the result."""
    action = agent.select_action(
        state=state,
        epsilon=epsilon,
    )

    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    experience = Experience(
        state,
        action,
        float(reward),
        next_state,
        done,
        info,
    )

    agent.store_transition(experience)

    return next_state, float(reward), done


def collect_episodes(
    env: gym.Env,
    agent: DQNAgent,
    num_episodes: int,
    epsilon: float,
) -> list[float]:
    """Collect transitions across multiple CartPole episodes."""
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0

        while True:
            state, reward, done = step_and_store(
                env=env,
                agent=agent,
                state=state,
                epsilon=epsilon,
            )

            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode}/{num_episodes} | "
            f"Reward: {total_reward:.0f} | "
            f"Replay Buffer: {len(agent.replay_buffer)}"
        )

    return episode_rewards


def main() -> None:
    """Create the DQN agent and collect initial experience."""
    env = gym.make("CartPole-v1")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        device=device,
    )

    rewards = collect_episodes(
        env=env,
        agent=agent,
        num_episodes=NUM_EPISODES,
        epsilon=EPSILON,
    )

    print("\nData collection complete.")
    print(f"Episodes collected: {len(rewards)}")
    print(f"Transitions collected: {len(agent.replay_buffer)}")
    print(
        f"Average reward: "
        f"{sum(rewards) / len(rewards):.2f}"
    )

    env.close()


if __name__ == "__main__":
    main()
