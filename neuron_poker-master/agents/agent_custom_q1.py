"""manual keypress agent"""
# pylint: disable=import-error
#from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from rl.memory import SequentialMemory
#
# from agents.agent_keras_rl_dqn import TrumpPolicy, memory_limit, window_length
# from gym_env import env

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import torch.optim as optim
import time

log = logging.getLogger(__name__)

class DQN(nn.Module):
    def __init__(self,input_shape,nb_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape,512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512,512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512,nb_actions)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class Player:
    """Mandatory class with the player methods"""

    def __init__(self, name='DQN', load_model=None, env=None, window_length=1, nb_max_start_steps=1,
                 train_interval=100, nb_steps_warmup=50, nb_steps=100000, memory_limit=None, batch_size=500,
                 enable_double_dqn=False, lr=1e-3, input_shape=4, nb_actions=18):
        """Initiaization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        self.window_length = window_length
        self.nb_max_start_steps = nb_max_start_steps
        self.train_interval = train_interval
        self.nb_steps_warmup = nb_steps_warmup
        self.nb_steps = nb_steps
        self.memory_limit = memory_limit if memory_limit is not None else int(nb_steps / 2)
        self.batch_size = batch_size
        self.enable_double_dqn = enable_double_dqn
        self.lr = lr

        # self.dqn = DQN # Probably should not be None?
        self.dqn = DQN(input_shape, nb_actions)
        self.model = DQN(input_shape, nb_actions)

        # Might not need this in here, I think initaite_agent is used for setting up env?
        self.env = env

        # Also set up in the function already?
        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        """initiate a deep Q agent"""

        self.env = env
        nb_actions = self.env.action_space.n

        self.dqn = DQN(env.observation_space.shape[0], nb_actions)
        self.optimizer = optim.Adam(self.dqn.parameters(),lr=self.lr)

        self.memory = ReplayBuffer(self.memory_limit)
        self.policy = TrumpPolicy()

        if self.load_model:
            self.load(self.load_model)
    
    def start_step_policy(self,observation):
        log.info('Random action')
        _= observation
        action = self.env.action.space.sample()
        return action
    
    def train(self, env_name):
        """Train a model"""
        # initiate training loop
        gamma = 0.99
        for step in range(self.nb_steps):
            # Sample a batch of transitions from the memory
            state, action, reward, next_state, done = self.memory.sample(self.batch_size)
            state = torch.tensor(state, dtype=torch.float32)
            action = torch.tensor(action, dtype=torch.int64)
            reward = torch.tensor(reward, dtype=torch.float32)
            next_state = torch.tensor(next_state, dtype=torch.float32)
            done = torch.tensor(done, dtype=torch.float32)

            # Calculate Q-values for the current and next states
            q_values = self.dqn(state)
            next_q_values = self.dqn(next_state)

            # Select the Q-value for the taken action
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Calculate the expected Q-value for the next state
            next_q_value = next_q_values.max(1)[0]

            # Calculate the target Q-value
            expected_q_value = reward + gamma * next_q_value * (1 - done)

            # Calculate the loss
            loss = (q_value - expected_q_value.detach()).pow(2).mean()

            # Backpropagate the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Save the model
        torch.save(self.dqn.state_dict(), f'dqn_{env_name}_weights.pth')

        # Finally, evaluate our algorithm for 5 episodes.
        # TODO: Implement your evaluation logic here
        self.evaluate(self.env)
    
    def evaluate(self, env, nb_episodes=5):
        """Evaluate the agent"""
        total_rewards = []
        for episode in range(nb_episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                state = torch.tensor([state], dtype=torch.float32)
                with torch.no_grad():
                    action = self.dqn(state).max(1)[1].item()
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
        avg_reward = sum(total_rewards) / len(total_rewards)
        print(f"Average reward over {nb_episodes} episodes: {avg_reward}")

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        total_rewards = []
        for episode in range(nb_episodes):
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                if render:
                    self.env.render()
                state = torch.tensor([state], dtype=torch.float32)
                with torch.no_grad():
                    action = self.dqn(state).max(1)[1].item()
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
            total_rewards.append(total_reward)
        avg_reward = sum(total_rewards) / len(total_rewards)
        # print(f"Average reward over {nb_episodes} episodes: {avg_reward}")



    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = info  # not using the info for decision

        # Convert the observation to PyTorch tensor and add an extra dimension
        observation = torch.tensor([observation], dtype=torch.float32)

        # Decide if explore or exploit
        if random.random() < self.epsilon:  # Exploration: choose a random action
            action = random.choice(action_space)
        else:  # Exploitation: choose the action with the highest Q-value
            with torch.no_grad():
                action = self.dqn(observation).max(1)[1].item()

        return action


class TrumpPolicy:
    def __init__(self, tau, clip):
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        assert q_values.dim() == 1
        nb_actions = q_values.shape[0]

        exp_values = torch.exp(torch.clamp(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / torch.sum(exp_values)
        action = torch.multinomial(probs, 1).item()
        print(f"Chosen action {action} - probabilities: {probs}")
        return action
    
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)
    
'''
Your PyTorch implementation seems to be a good start, but there are a few things you might want to consider:

1. **Replay Buffer**: In your Keras code, you used `SequentialMemory` from Keras RL as your replay buffer. In your PyTorch code, you implemented your own `ReplayBuffer`. Make sure your custom `ReplayBuffer` in PyTorch has the same functionality as the `SequentialMemory` in Keras.

2. **Policy**: You used `BoltzmannQPolicy` in Keras and implemented your own `TrumpPolicy` in PyTorch. Ensure that your `TrumpPolicy` in PyTorch behaves the same way as the `BoltzmannQPolicy` in Keras.

3. **Training Loop**: In your Keras code, you used the `fit` method provided by Keras RL. In your PyTorch code, you implemented your own training loop. Make sure your training loop in PyTorch is correctly implementing the Q-learning algorithm.

4. **Model Saving and Loading**: In Keras, you used `model.save_weights` and `model.load_weights` to save and load model weights. In PyTorch, you used `torch.save` and `torch.load`. These are not equivalent. In PyTorch, `torch.save` and `torch.load` can save and load the entire model, including its architecture, not just the weights. If you want to save and load only the weights in PyTorch, you should use `state_dict` and `load_state_dict`.

5. **Missing Parts**: Your PyTorch code seems to be missing the implementation of the `CustomProcessor` class that is present in your Keras code. If this class is necessary for preprocessing your data, you should implement it in PyTorch as well.

Please revise your PyTorch code considering these points. If you have any specific issues or errors, feel free to ask! I'm here to help.
'''