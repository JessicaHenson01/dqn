## Environment Setup

This project uses the Gymnasium `CartPole-v1` environment for training and
evaluating the Deep Q-Network. CartPole is a discrete-action reinforcement
learning environment in which an agent controls a cart by applying a force
to either the left or right while attempting to keep an attached pole
balanced.

The environment has a four-dimensional observation space containing the
cart position, cart velocity, pole angle, and pole angular velocity. The
action space contains two discrete actions corresponding to moving the cart
left or right.

Install the required dependencies with:

```bash
python -m pip install -r requirements.txt

```
### DQN Architecture and Hyperparameters

The DQN uses a fully connected neural network with an input dimension of four, corresponding to the four values in the CartPole observation space. The network contains two hidden layers with 128 neurons each and ReLU activation functions. The output layer contains two neurons representing the estimated Q-values of the two available actions.

The agent maintains both an online Q-network and a target network. The online network is used for action selection and learning, while the target network provides stable target Q-values during training. The target network is initialized with the same weights as the online network and will be periodically updated using a hard copy of the online network parameters.

The replay buffer stores up to 100,000 transitions and randomly samples minibatches of 64 experiences during training. A discount factor of 0.99 is used so that the agent places substantial value on future rewards while still prioritizing immediate rewards. The Adam optimizer uses a learning rate of 0.001.
