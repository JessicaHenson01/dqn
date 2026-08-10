import gymnasium as gym
import torch

from agent import DQNAgent


def collect_episodes(
    env: gym.Env,
    agent: DQNAgent,
    num_episodes: int,
    epsilon: float,
) -> list[float]:

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()

        done = False
        total_reward = 0.0

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
            total_reward += reward

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode + 1}/{num_episodes} | "
            f"Reward: {total_reward:.0f} | "
            f"Replay Buffer: {len(agent.replay_buffer)}"
        )

    return episode_rewards


def main() -> None:
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
    )

    num_episodes = 10
    epsilon = 1.0

    rewards = collect_episodes(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        epsilon=epsilon,
    )

    print("\nData collection complete.")
    print(f"Episodes collected: {len(rewards)}")
    print(f"Transitions collected: {len(agent.replay_buffer)}")
    print(f"Average reward: {sum(rewards) / len(rewards):.2f}")

    env.close()


if __name__ == "__main__":
    main()