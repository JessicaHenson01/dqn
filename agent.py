"""Deep Q-Network agent implementation for CartPole-v1."""

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, optim

from model import DQN
from replay_buffer import Experience, ReplayBuffer


@dataclass(frozen=True)
class AgentConfig:
    """Store DQN agent hyperparameters."""

    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    buffer_size: int = 100_000


class DQNAgent:
    """Deep Q-Network agent with replay memory and target network."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        device: torch.device,
        config: AgentConfig | None = None,
    ) -> None:
        """Initialize the DQN agent.

        Args:
            state_size: Number of values in an environment state.
            action_size: Number of discrete environment actions.
            device: Device used for PyTorch operations.
            config: Optional DQN hyperparameter configuration.
        """
        settings = config or AgentConfig()

        self.action_size = action_size
        self.device = device
        self.gamma = settings.gamma

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
            lr=settings.learning_rate,
        )

        self.replay_buffer = ReplayBuffer(
            capacity=settings.buffer_size,
            batch_size=settings.batch_size,
            device=device,
        )

    @property
    def batch_size(self) -> int:
        """Return the replay-buffer minibatch size."""
        return self.replay_buffer.batch_size

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float,
    ) -> int:
        """Select an action using an epsilon-greedy policy."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.as_tensor(
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
        experience: Experience,
    ) -> None:
        """Store one environment transition in replay memory."""
        self.replay_buffer.add(experience)

    def update_target_network(self) -> None:
        """Copy the online network weights to the target network."""
        self.target_network.load_state_dict(
            self.q_network.state_dict()
        )

    def learn(self) -> float:
        """Perform one DQN optimization step using replay memory."""
        states, actions, rewards, next_states, dones = (
            self.replay_buffer.sample()
        )

        current_q_values = self.q_network(
            states
        ).gather(
            1,
            actions,
        )

        with torch.no_grad():
            next_q_values = self.target_network(
                next_states
            ).max(
                dim=1,
                keepdim=True,
            )[0]

            target_q_values = rewards + (
                self.gamma
                * next_q_values
                * (1 - dones)
            )

        loss = nn.functional.mse_loss(
            current_q_values,
            target_q_values,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
