"""Experience replay buffer for Deep Q-Network training."""

import random
from collections import deque
from typing import Any, NamedTuple

import numpy as np
import torch


class Experience(NamedTuple):
    """Store one reinforcement-learning environment transition."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    info: dict[str, Any]


class ReplayBuffer:
    """Store and randomly sample experiences for DQN training."""

    def __init__(
        self,
        capacity: int,
        batch_size: int,
        device: torch.device,
    ) -> None:
        """Initialize the replay buffer.

        Args:
            capacity: Maximum number of experiences to store.
            batch_size: Number of experiences sampled per batch.
            device: PyTorch device used for returned tensors.
        """
        self.memory: deque[Experience] = deque(
            maxlen=capacity
        )
        self.batch_size = batch_size
        self.device = device

    def add(
        self,
        experience: Experience,
    ) -> None:
        """Add one experience to replay memory."""
        self.memory.append(experience)

    def sample(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Randomly sample a minibatch and return PyTorch tensors."""
        experiences = random.sample(
            self.memory,
            k=self.batch_size,
        )

        states = torch.tensor(
            np.array([
                experience.state
                for experience in experiences
            ]),
            dtype=torch.float32,
            device=self.device,
        )

        actions = torch.tensor(
            [
                experience.action
                for experience in experiences
            ],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)

        rewards = torch.tensor(
            [
                experience.reward
                for experience in experiences
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        next_states = torch.tensor(
            np.array([
                experience.next_state
                for experience in experiences
            ]),
            dtype=torch.float32,
            device=self.device,
        )

        dones = torch.tensor(
            [
                experience.done
                for experience in experiences
            ],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        return (
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

    def __len__(self) -> int:
        """Return the current number of stored experiences."""
        return len(self.memory)
