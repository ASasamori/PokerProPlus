import torch
import torch.nn as nn
import os
from datetime import datetime
import numpy as np
import random
from collections import deque
import logging
from gym_env.enums import Action

log = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    def __init__(self, input_shape, nb_actions):
        super(DQNNetwork, self).__init__()
        self.input_shape = input_shape
        self.nb_actions = nb_actions

        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_shape, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, nb_actions)

        # Define the activation functions and dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Define the forward pass of the neural network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)

        return x


class Player:
    """Mandatory class with the player methods"""
    def __init__(self, name='DQN', load_model=None, env=None, window_length=1, nb_max_start_steps=1,
                 train_interval=100, nb_steps_warmup=50, nb_steps=100000, memory_limit=None, batch_size=500,
                 enable_double_dqn=False, lr=1e-3):
        """Initialization of an agent"""
        self.equity_alive = 0
        self.actions = []
        self.last_action_in_stage = ''
        self.temp_stack = []
        self.name = name
        self.autoplay = True

        # Hyperparameters we can adjust
        self.window_length = window_length
        self.nb_max_start_steps = nb_max_start_steps
        self.train_interval = train_interval
        self.nb_steps_warmup = nb_steps_warmup
        self.nb_steps = nb_steps
        self.memory_limit = memory_limit if memory_limit is not None else int(nb_steps / 2)
        self.batch_size = batch_size
        self.enable_double_dqn = enable_double_dqn
        self.lr = lr

        self.dqn = None
        self.model = None
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        """Initiate a deep Q agent"""
        self.env = env
        nb_actions = self.env.action_space.n
        input_shape = self.env.observation_space[0]

        self.model = DQNNetwork(input_shape, nb_actions).to(self.device)
        self.memory = deque(maxlen=self.memory_limit)
        self.policy = TrumpPolicy()  # You'll need to translate this class
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.target_model = DQNNetwork(input_shape, nb_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model_update_interval = 1e-2

    def train(self, env_name):
        """Train a model"""
        # Create a directory for saving the model
        timestr = datetime.now().strftime("%Y%m%d-%H%M%S") + "_" + str(env_name)
        model_dir = os.path.join("./models", timestr)
        os.makedirs(model_dir, exist_ok=True)

        # Set the number of episodes and other training parameters
        num_episodes = self.nb_steps // 500  # We set a fixed number of steps for training
        max_steps = 500
        epsilon_start = 1.0
        epsilon_final = 0.01
        epsilon_decay = 500

        # Training loop
        for episode in range(num_episodes):
            state = self.env.reset()
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * np.exp(-episode / epsilon_decay)
            episode_reward = 0
            for step in range(max_steps):
                # Select an action based on the current policy
                if random.random() > epsilon:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                        q_values = self.model(state_tensor).squeeze(0)
                    action = q_values.max(0)[1].item()
                else:
                    action = self.env.action_space.sample()

                # Take the action and observe the next state and reward
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward

                # Store the transition in the replay memory
                self.memory.append((state, action, reward, next_state, done))

                # Sample a batch from the replay memory and update the model
                if len(self.memory) >= self.batch_size:
                    self.update_model()

                state = next_state

                if done:
                    break

            # Save the model after each episode
            torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_{episode}.pt"))

        # Save the final model
        torch.save(self.model.state_dict(), os.path.join(model_dir, "final_model.pt"))

    def update_model(self):
        """Update the model weights using a batch of transitions"""
        # Sample a batch from the replay memory
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute the Q-values for the current states
        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute the target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute the loss and update the model
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update the target model
        if self.steps % self.target_model_update_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps += 1

    def load(self, env_name):
        """Load a model"""
        model_dir = os.path.join("./models", env_name)
        model_path = os.path.join(model_dir, "final_model.pt")

        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(f"Model not found at {model_path}")

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        self.memory = deque(maxlen=self.memory_limit)
        self.policy = TrumpPolicy()  # You'll need to translate this class

        class CustomProcessor:
            """The agent and the environment"""

            def process_state_batch(self, batch):
                """
                Given a state batch, remove the second dimension to feed into the network
                """
                return np.squeeze(batch, axis=1)

            def process_info(self, info):
                processed_info = info['player_data']
                if 'stack' in processed_info:
                    processed_info = {'x': 1}
                return processed_info

        self.processor = CustomProcessor()

        for episode in range(nb_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False

            while not done:
                if render:
                    self.env.render()

                state = self.processor.process_state_batch([state])
                state = torch.tensor(state, dtype=torch.float32, device=self.device)

                with torch.no_grad():
                    q_values = self.model(state)
                action = self.policy.select_action(q_values)

                next_state, reward, done, info = self.env.step(action)
                episode_reward += reward

                processed_info = self.processor.process_info(info)
                self.memory.append((state, action, reward, next_state, done, processed_info))

                state = next_state

            print(f"Episode {episode}: Reward = {episode_reward}")

    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action


class TrumpPolicy:
    """Custom policy when making decision based on neural network."""

    def __init__(self, tau=1.0, clip=(-500, 500)):
        self.tau = tau
        self.clip = clip

    def select_action(self, q_values):
        """Return the selected action

        Args:
            q_values (torch.Tensor): Tensor of Q-values for each action

        Returns:
            int: Selected action
        """
        assert q_values.ndim == 1
        q_values = q_values.float()
        nb_actions = q_values.size(0)

        exp_values = torch.exp(torch.clamp(q_values / self.tau, min=self.clip[0], max=self.clip[1]))
        probs = exp_values / torch.sum(exp_values)
        action = torch.multinomial(probs, 1).item()
        log.info(f"Chosen action by PyTorch {action} - probabilities: {probs}")
        return action

class CustomProcessor:
    """The agent and the environment"""

    def __init__(self):
        """Initialize properties"""
        self.legal_moves_limit = None

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into CNN"""
        batch = np.array(batch)
        return np.squeeze(batch, axis=1)

    def process_info(self, info):
        if 'legal_moves' in info.keys():
            self.legal_moves_limit = info['legal_moves']
        else:
            self.legal_moves_limit = None
        return {'x': 1}  # Only arrays allowed it seems

    def process_action(self, action):
        """Find nearest legal action"""
        if hasattr(self, 'legal_moves_limit') and self.legal_moves_limit is not None:
            self.legal_moves_limit = [move.value for move in self.legal_moves_limit]
            if action not in self.legal_moves_limit:
                for i in range(5):
                    action += i
                    if action in self.legal_moves_limit:
                        break
                    action -= i * 2
                    if action in self.legal_moves_limit:
                        break
                    action += i

        return action