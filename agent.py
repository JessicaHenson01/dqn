"""Deep Q-Network agent implementation."""

import random

import numpy as np
import torch
from torch import nn, optim

from model import DQN
from replay_buffer import ReplayBuffer


class DQNAgent:
    """Agent implementing the core components of Deep Q-Learning."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: torch.device,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 100_000,
    ) -> None:
        """Initialize the DQN agent.

        Args:
            state_size: Dimension of the environment observation.
            action_size: Number of possible actions.
            device: Device on which PyTorch computations are performed.
            learning_rate: Adam optimizer learning rate.
            gamma: Discount factor for future rewards.
            batch_size: Number of transitions used per training step.
            buffer_size: Maximum number of transitions in replay memory.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        self.gamma = gamma
        self.batch_size = batch_size

        self.q_network = DQN(
            state_size=state_size,
            action_size=action_size,
        ).to(device)

        self.target_network = DQN(
            state_size=state_size,
            action_size=action_size,
        ).to(device)

        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

        self.target_network.eval()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        self.loss_function = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            batch_size=batch_size,
            device=device,
        )

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
    ) -> int:
        """Select an action using an epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        self.q_network.eval()

        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        self.q_network.train()

        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, object],
    ) -> None:
        """Store a transition in replay memory."""
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            info,
        )

    def update_target_network(self) -> None:
        """Copy Q-network weights into the target network."""
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    def learn(self) -> float:
        """Update the Q-network using a batch from replay memory."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()

        # Q-values predicted by the online network for the actions taken.
        current_q_values = self.q_network(states).gather(1, actions)

        # Compute target Q-values without tracking gradients.
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(
                dim=1,
                keepdim=True,
            )[0]

            target_q_values = rewards + (
                self.gamma * next_q_values * (1 - dones)
            )

        loss = self.loss_function(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())