"""Train a Deep Q-Network on CartPole-v1."""

from collections import deque

import gymnasium as gym
import torch

from agent import DQNAgent


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

    epsilon = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995

    recent_rewards = deque(maxlen=100)

    total_steps = 0

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

        print(
            f"Episode {episode:3d} | "
            f"Reward: {episode_reward:6.1f} | "
            f"Avg Reward: {average_reward:6.1f} | "
            f"Loss: {average_loss:.4f} | "
            f"Epsilon: {epsilon:.3f} | "
            f"Buffer: {len(agent.replay_buffer)}"
        )

    torch.save(
        agent.q_network.state_dict(),
        "dqn_cartpole.pt",
    )

    print("Model saved to dqn_cartpole.pt")

    env.close()

    return agent


if __name__ == "__main__":
    train()