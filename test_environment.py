"""Test the Gymnasium CartPole-v1 environment with random actions."""

import gymnasium as gym


def main() -> None:
    """Run one CartPole episode using randomly sampled actions."""
    env = gym.make(
        "CartPole-v1",
        render_mode="human",
    )

    state, _ = env.reset()

    print("Initial state:", state)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    total_reward = 0.0

    for step in range(200):
        action = env.action_space.sample()

        _, reward, terminated, truncated, _ = env.step(action)

        total_reward += reward

        print(
            f"Step {step + 1}: "
            f"action={action}, reward={reward}, "
            f"terminated={terminated}, "
            f"truncated={truncated}"
        )

        if terminated or truncated:
            print(
                f"Episode finished after "
                f"{step + 1} steps."
            )
            break

    print(f"Total reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
