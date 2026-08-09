"""Verify the CartPole-v1 environment setup."""

import gymnasium as gym


def main():
    """Create CartPole, step through it, and verify rendering."""
    env = gym.make("CartPole-v1", render_mode="human")

    state, info = env.reset()

    print("Initial state:", state)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    total_reward = 0.0

    for step in range(200):
        # Random action for environment verification only.
        action = env.action_space.sample()

        next_state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        print(
            f"Step {step + 1}: "
            f"action={action}, reward={reward}, "
            f"terminated={terminated}, truncated={truncated}"
        )

        state = next_state

        if terminated or truncated:
            print(f"Episode finished after {step + 1} steps.")
            break

    print(f"Total reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    main()