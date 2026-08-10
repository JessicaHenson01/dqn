"""Basic tests for the DQN components."""

import gymnasium as gym
import torch

from agent import DQNAgent
from replay_buffer import Experience


def main():
    """Create and verify the DQN agent."""
    env = gym.make("CartPole-v1")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("State size:", state_size)
    print("Action size:", action_size)

    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
    )

    print(agent.q_network)

    state, info = env.reset()

    action = agent.select_action(
        state=state,
        epsilon=0.0,
    )

    print("Greedy action:", action)

    next_state, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

    experience = Experience(
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        done=done,
        info=info,
    )

    agent.store_transition(experience)

    print("Replay buffer size:", len(agent.replay_buffer))

    agent.update_target_network()

    print("Target network update successful.")

    env.close()


if __name__ == "__main__":
    main()
