"""Train and periodically evaluate a Deep Q-Network on CartPole-v1."""

import csv
import os
from collections import deque
from dataclasses import dataclass, field
from typing import TextIO

import gymnasium as gym
import torch

from agent import AgentConfig, DQNAgent
from replay_buffer import Experience


@dataclass(frozen=True)
class RunConfig:
    """Store training and evaluation schedule settings."""

    num_episodes: int = 500
    warmup_steps: int = 1000
    target_update_frequency: int = 3
    evaluation_frequency: int = 25
    evaluation_episodes: int = 10


@dataclass(frozen=True)
class ExplorationConfig:
    """Store epsilon-greedy exploration settings."""

    start: float = 1.0
    minimum: float = 0.01
    decay: float = 0.995


@dataclass(frozen=True)
class TrainingConfig:
    """Group all DQN configuration settings."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)
    exploration: ExplorationConfig = field(
        default_factory=ExplorationConfig
    )


@dataclass
class TrainingState:
    """Store values that change during training."""

    epsilon: float
    total_steps: int = 0
    best_eval_reward: float = float("-inf")
    best_eval_episode: int = 0


@dataclass
class EpisodeMetrics:
    """Store statistics produced by one training episode."""

    reward: float
    average_reward: float
    average_loss: float
    epsilon: float
    buffer_size: int
    evaluation_reward: float | str = ""


@dataclass
class TrainingContext:
    """Store objects shared throughout the training process."""

    env: gym.Env
    agent: DQNAgent
    config: TrainingConfig
    state: TrainingState


def get_device() -> torch.device:
    """Return the best available PyTorch device."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def create_agent(
    env: gym.Env,
    device: torch.device,
    config: AgentConfig,
) -> DQNAgent:
    """Create a DQN agent using environment dimensions."""
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    return DQNAgent(
        state_size=state_size,
        action_size=action_size,
        device=device,
        config=config,
    )


def evaluate_agent(
    agent: DQNAgent,
    num_episodes: int,
) -> float:
    """Evaluate the current agent using a greedy policy."""
    eval_env = gym.make("CartPole-v1")
    rewards = []

    for _ in range(num_episodes):
        state, _ = eval_env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(
                state=state,
                epsilon=0.0,
            )

            next_state, reward, terminated, truncated, _ = (
                eval_env.step(action)
            )

            done = terminated or truncated
            state = next_state
            total_reward += reward

        rewards.append(total_reward)

    eval_env.close()

    return sum(rewards) / len(rewards)


def should_learn(
    agent: DQNAgent,
    total_steps: int,
    warmup_steps: int,
) -> bool:
    """Return whether the agent has enough experience to learn."""
    return (
        total_steps >= warmup_steps
        and len(agent.replay_buffer) >= agent.batch_size
    )


def run_training_episode(
    context: TrainingContext,
) -> tuple[float, float]:
    """Run one training episode and return reward and average loss."""
    state, _ = context.env.reset()
    episode_reward = 0.0
    episode_losses = []
    done = False

    while not done:
        action = context.agent.select_action(
            state=state,
            epsilon=context.state.epsilon,
        )

        next_state, reward, terminated, truncated, info = (
            context.env.step(action)
        )

        done = terminated or truncated

        context.agent.store_transition(
            Experience(
                state,
                action,
                float(reward),
                next_state,
                done,
                info,
            )
        )

        state = next_state
        episode_reward += reward
        context.state.total_steps += 1

        if should_learn(
            context.agent,
            context.state.total_steps,
            context.config.run.warmup_steps,
        ):
            episode_losses.append(context.agent.learn())

    if not episode_losses:
        return episode_reward, 0.0

    average_loss = sum(episode_losses) / len(episode_losses)

    return episode_reward, average_loss


def save_best_model(
    agent: DQNAgent,
    episode: int,
    evaluation_reward: float,
    state: TrainingState,
) -> None:
    """Save the model when greedy evaluation improves."""
    if evaluation_reward <= state.best_eval_reward:
        return

    state.best_eval_reward = evaluation_reward
    state.best_eval_episode = episode

    torch.save(
        agent.q_network.state_dict(),
        "dqn_cartpole.pt",
    )

    print(
        f"*** New best model saved at Episode {episode} "
        f"with Evaluation Reward = {evaluation_reward:.2f} ***"
    )


def evaluate_if_needed(
    episode: int,
    context: TrainingContext,
) -> float | str:
    """Run periodic greedy evaluation when required."""
    run_config = context.config.run

    if episode % run_config.evaluation_frequency != 0:
        return ""

    evaluation_reward = evaluate_agent(
        agent=context.agent,
        num_episodes=run_config.evaluation_episodes,
    )

    print(
        f"*** Greedy Evaluation at Episode {episode}: "
        f"Average Reward = {evaluation_reward:.2f} ***"
    )

    save_best_model(
        agent=context.agent,
        episode=episode,
        evaluation_reward=evaluation_reward,
        state=context.state,
    )

    return evaluation_reward


def update_training_state(
    episode: int,
    context: TrainingContext,
) -> float | str:
    """Update target network, epsilon, and periodic evaluation."""
    if (
        episode
        % context.config.run.target_update_frequency
        == 0
    ):
        context.agent.update_target_network()

    exploration = context.config.exploration

    context.state.epsilon = max(
        exploration.minimum,
        context.state.epsilon * exploration.decay,
    )

    return evaluate_if_needed(
        episode=episode,
        context=context,
    )


def collect_episode_metrics(
    episode: int,
    context: TrainingContext,
    recent_rewards: deque,
) -> EpisodeMetrics:
    """Run one episode and collect its training statistics."""
    episode_reward, average_loss = run_training_episode(
        context=context,
    )

    recent_rewards.append(episode_reward)

    average_reward = (
        sum(recent_rewards) / len(recent_rewards)
    )

    evaluation_reward = update_training_state(
        episode=episode,
        context=context,
    )

    return EpisodeMetrics(
        reward=episode_reward,
        average_reward=average_reward,
        average_loss=average_loss,
        epsilon=context.state.epsilon,
        buffer_size=len(context.agent.replay_buffer),
        evaluation_reward=evaluation_reward,
    )


def create_log_writer(
    log_file: TextIO,
) -> csv.writer:
    """Create the training CSV writer and write its header."""
    writer = csv.writer(log_file)

    writer.writerow([
        "episode",
        "reward",
        "average_reward",
        "average_loss",
        "epsilon",
        "buffer_size",
        "evaluation_reward",
    ])

    return writer


def write_log_row(
    writer: csv.writer,
    episode: int,
    metrics: EpisodeMetrics,
) -> None:
    """Write one episode of statistics to the training log."""
    writer.writerow([
        episode,
        metrics.reward,
        metrics.average_reward,
        metrics.average_loss,
        metrics.epsilon,
        metrics.buffer_size,
        metrics.evaluation_reward,
    ])


def print_episode_status(
    episode: int,
    metrics: EpisodeMetrics,
) -> None:
    """Print one formatted training-progress line."""
    print(
        f"Episode {episode:3d} | "
        f"Reward: {metrics.reward:6.1f} | "
        f"Avg Reward: {metrics.average_reward:6.1f} | "
        f"Loss: {metrics.average_loss:.4f} | "
        f"Epsilon: {metrics.epsilon:.3f} | "
        f"Buffer: {metrics.buffer_size}"
    )


def run_training_loop(
    context: TrainingContext,
    log_file: TextIO,
) -> None:
    """Run all configured DQN training episodes."""
    recent_rewards = deque(maxlen=100)
    log_writer = create_log_writer(log_file)

    for episode in range(
        1,
        context.config.run.num_episodes + 1,
    ):
        metrics = collect_episode_metrics(
            episode=episode,
            context=context,
            recent_rewards=recent_rewards,
        )

        print_episode_status(
            episode=episode,
            metrics=metrics,
        )

        write_log_row(
            writer=log_writer,
            episode=episode,
            metrics=metrics,
        )

        log_file.flush()


def train() -> DQNAgent:
    """Train a DQN agent on CartPole-v1."""
    config = TrainingConfig()
    env = gym.make("CartPole-v1")
    device = get_device()

    print(f"Using device: {device}")

    agent = create_agent(
        env=env,
        device=device,
        config=config.agent,
    )

    state = TrainingState(
        epsilon=config.exploration.start,
    )

    context = TrainingContext(
        env=env,
        agent=agent,
        config=config,
        state=state,
    )

    os.makedirs("logs", exist_ok=True)

    with open(
        "logs/training_log.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as log_file:
        run_training_loop(
            context=context,
            log_file=log_file,
        )

    print(
        "\nTraining complete. "
        f"Best evaluation reward: "
        f"{state.best_eval_reward:.2f} "
        f"at episode {state.best_eval_episode}"
    )

    print("Best model saved to dqn_cartpole.pt")

    env.close()

    return agent


if __name__ == "__main__":
    train()
