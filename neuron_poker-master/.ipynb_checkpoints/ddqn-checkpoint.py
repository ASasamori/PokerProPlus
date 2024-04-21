import logging
import time

import numpy as np

from gym_env.enums import Action

import tensorflow as tf
import json

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# from rl.policy import BoltzmannQPolicy
# from rl.memory import SequentialMemory
# from rl.agents import DQNAgent
# from rl.core import Processor

class QNetwork(tf.keras.Model):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.fc2 = tf.keras.layers.Dense(action_size)

    def call(self, state):
        state = tf.nn.relu(self.fc1(state))
        return self.fc2(state)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    ## MIGHT HAVE PROBLEM !!!!!!!!!!! ######
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

def generate_nn(state_size, action_size, hidden_size=64):
    keras

class DQNAgentPolicy:

    def __init__(self, self, state_size, action_size, hidden_size, 
                 buffer_size, batch_size, update_every,
                 learning_rate, gamma, tau, device, use_double_dqn=False)
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.update_every = update_every
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.update_steps = 0
        self.use_double_dqn = use_double_dqn

        self.q = QNetwork(state_size, action_size, hidden_size)
        self.q_targ = QNetwork(state_size, action_size, hidden_size)
        self.optimizer = Adam(lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)


    def train(self):
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        
        if (self.update_steps + 1) % self.update_every == 0:
            self.soft_update()


    def act(self, state, eps=.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.q.eval()
        with torch.no_grad():
            action_values = self.q(state)
        self.q.train()
 
        if random.random() > eps:
            return np.argmax(action_values.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        with tf.GradientTape() as tape:
            # Get expected Q values from local model
            Q_expected = self.q(states)
            Q_expected = tf.gather(Q_expected, actions, axis=1, batch_dims=0)
    
            # Get max predicted Q values (for next states) from target model
            if not self.use_double_dqn:
                Q_targets_next = tf.reduce_max(self.q_targ(next_states), axis=1, keepdims=True)
            else:
                Q_local_max_actions = tf.argmax(self.q(next_states), axis=1, output_type=tf.int32)
                Q_targets_next = tf.gather(self.q_targ(next_states), Q_local_max_actions, axis=1, batch_dims=0)
    
            # Compute Q targets for current states 
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
    
            # Compute loss
            loss = tf.reduce_mean(tf.square(Q_expected - Q_targets))
    
        # Apply gradients
        gradients = tape.gradient(loss, self.q.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q.trainable_variables))
    
        self.update_steps += 1

    def soft_update(self):
        target_weights = self.q_targ.get_weights()
        local_weights = self.q.get_weights()
        new_weights = [self.tau * local_w + (1 - self.tau) * target_w for target_w, local_w in zip(target_weights, local_weights)]
        target_model.set_weights(new_weights)
        
        

    





    