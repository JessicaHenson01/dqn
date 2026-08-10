# Deep Q-Network for CartPole-v1

This project implements a Deep Q-Network (DQN) from scratch using PyTorch and the Gymnasium `CartPole-v1` environment. The agent learns to balance a pole on a moving cart by estimating Q-values for the available actions and selecting actions using an epsilon-greedy policy during training.

The implementation includes a Q-network, target network, experience replay buffer, epsilon-greedy exploration, periodic target-network updates, model checkpointing, and greedy policy evaluation.

## Environment Setup

This project uses the Gymnasium `CartPole-v1` environment. The environment has a four-dimensional observation space with:

1. Cart position
2. Cart velocity
3. Pole angle
4. Pole angular velocity

The action space has two discrete actions: applying force to the cart in different directions.

The environment can be tested independently using `test_environment.py`.

### Requirements

* Python 3.10+
* PyTorch
* Gymnasium
* NumPy

Install the required dependencies with:

```bash
python -m pip install -r requirements.txt
```

To verify the CartPole environment:

```bash
python test_environment.py
```

## Project Structure

```text
dqn/
├── logs/
│   ├── training_log.csv
│   └── evaluation_log.csv
├── agent.py
├── collect_data.py
├── model.py
├── replay_buffer.py
├── train.py
├── evaluate.py
├── test_agent.py
├── test_environment.py
├── requirements.txt
├── dqn_cartpole.pt
└── README.md
```

### File Descriptions

* `model.py` — Defines the neural network used to estimate Q-values
* `agent.py` — Implements the DQN agent, epsilon-greedy action selection, online and target networks, and learning updates
* `replay_buffer.py` — Implements the experience replay buffer and minibatch sampling
* `collect_data.py` — Demonstrates epsilon-greedy environment interaction and transition collection
* `train.py` — Runs the complete DQN training procedure and periodic greedy evaluation
* `evaluate.py` — Loads the saved model and evaluates it without exploration
* `test_environment.py` — Verifies that `CartPole-v1` can be created, stepped through, and rendered
* `test_agent.py` — Performs basic checks of the DQN architecture, replay buffer, action selection, and target-network update
* `logs/` — Contains CSV logs generated during training and evaluation.
* `dqn_cartpole.pt` — Saved weights for the best-performing DQN checkpoint.

## DQN Architecture

The DQN is implemented as a fully connected multilayer perceptron:

```text
State (4)
   ↓
Linear (4 → 128)
   ↓
ReLU
   ↓
Linear (128 → 128)
   ↓
ReLU
   ↓
Linear (128 → 2)
   ↓
Q-values for each action
```

The four input values matching the CartPole observation. The two outputs match the estimated Q-values of the 2 available actions.

The output values are Q-value estimates rather than probabilities. During greedy action selection, the action with the highest predicted Q-value is selected.

## Online and Target Networks

The agent has 2 copies of the DQN:

* **Online Q-network:** Used to select actions and updated through gradient descent
* **Target Q-network:** Used to calculate more stable target Q-values during learning

The target network starts with the same parameters as the online network. During training, its parameters are replaced with the current online-network parameters using a hard update.

## Experience Replay

Each environment transition is stored in a replay buffer as:

```text
(state, action, reward, next_state, done, info)
```

Once the initial warm-up period is complete, random minibatches of 64 transitions are sampled from the buffer for training.

Random replay sampling reduces the correlation between consecutive environment observations and allows experiences to be reused for multiple learning updates.

## Hyperparameters

The primary training configuration is:

| Hyperparameter            |       Value |
| ------------------------- | ----------: |
| Learning rate             |       0.001 |
| Discount factor (`gamma`) |        0.99 |
| Batch size                |          64 |
| Replay buffer capacity    |     100,000 |
| Hidden layer size         |         128 |
| Training episodes         |         500 |
| Replay warm-up            | 1,000 steps |
| Target update frequency   |  3 episodes |
| Initial epsilon           |         1.0 |
| Minimum epsilon           |        0.01 |
| Epsilon decay             |       0.995 |
| Evaluation frequency      | 25 episodes |
| Evaluation episodes       |          10 |

A discount factor of 0.99 places substantial value on future rewards, which is important for learning to keep the pole balanced over long episodes. A replay batch size of 64 provides a balance between stable gradient estimates and inexpensive updates.

Epsilon begins at 1.0 to encourage exploration. It gradually decreases during training so that the agent increasingly relies on its learned Q-values.

## DQN Training

Training uses an epsilon-greedy policy. At each step, the agent chooses a random action with probability epsilon or the action with the highest Q-value predicted by the online network.

Training begins after 1,000 environment transitions. A minibatch is sampled from replay memory and the online network is optimized using the DQN target:

```text
target = reward + gamma * max(Q_target(next_state))
```

For terminal states, the future Q-value contribution is removed.

The online network is trained to minimize the mean squared error between its predicted Q-value for the selected action and the target Q-value.

### Start Training

Run:

```bash
python train.py
```

Training runs for 500 episodes.

Every 25 episodes, the current policy is evaluated over 10 episodes with `epsilon = 0.0`. This provides a measurement of learned policy performance without random exploratory actions.

The model with the highest periodic greedy evaluation reward is saved as:

```text
dqn_cartpole.pt
```

Saving the best evaluation checkpoint rather than simply the final training state helps account for instability in DQN training.

## Training Logs

Training statistics are written to:

```text
logs/training_log.csv
```

The log records:

* Episode number
* Episode reward
* Rolling average reward
* Average training loss
* Epsilon
* Replay-buffer size
* Periodic greedy evaluation reward

The rolling training reward may differ substantially from greedy evaluation performance because training continues to use epsilon-greedy exploration. Random exploratory actions can cause an otherwise strong policy to terminate an episode early.

## Evaluation

The saved checkpoint is evaluated using a purely greedy policy

Run:

```bash
python evaluate.py
```

The evaluation script loads `dqn_cartpole.pt`, runs multiple CartPole episodes, and reports the reward for each episode along with the average, minimum, and maximum reward.

A reward of 500 indicates that the agent kept the pole balanced for the full `CartPole-v1` episode.

## Results

During development, the DQN learned to successfully balance the CartPole and reached the maximum reward of 500 during greedy evaluation.

Periodic evaluation also demonstrated why evaluating seprately from training reward is important. Training rewards remained noisy because epsilon-greedy exploration introduced random actions, while greedy evaluations provided a clearer measurement of the learned policy.

The best-performing model is retained in `dqn_cartpole.pt` and can be reproduced or evaluated using the scripts described above.

## Reproducing the Experiment

From a clean environment:

```bash
python -m pip install -r requirements.txt
python test_environment.py
python test_agent.py
python train.py
python evaluate.py
```

The training procedure will generate a new `training_log.csv` and save the best model checkpoint to `dqn_cartpole.pt`.
