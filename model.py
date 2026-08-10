import torch
from torch import nn


class DQN(nn.Module):

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 128,
    ) -> None:

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