"""Neural network architecture for the Deep Q-Network agent."""

import torch
from torch import nn


class DQN(nn.Module):
    """Fully connected network for estimating action Q-values."""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
    ) -> None:
        """Initialize the DQN architecture.

        Args:
            state_size: Number of values in the environment state.
            action_size: Number of discrete actions.
            hidden_size: Number of neurons in each hidden layer.
        """
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each possible action."""
        return self.network(state)
