import logging
import time
import numpy as np
import tensorflow as tf
import json
import keras
from keras import layers
from keras.optimizers import Adam
from keras.models import Sequential

from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.agents.dqn import DQNAgent
from rl.core import Processor
from gym_env.enums import Action

autoplay = True  # play automatically if played against keras-rl

log = logging.getLogger(__name__)


def create_q_model(env):
    print(env.observation_space)
    return Sequential(
        [
            layers.Dense(512,activation='relu', input_shape=(1,1,328)),
            layers.Dropout(0.2),
            layers.Dense(256,activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(128,activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(64,activation='relu'),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(32, activation='relu'),
            layers.Dense(8, activation='linear')
        ]
    )

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

        # hyperparameters we can adjust
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

        if load_model:
            self.load(load_model)

    def initiate_agent(self, env):
        """Initiate a deep Q agent"""
        self.env = env

        nb_actions = self.env.action_space.n
        self.model = create_q_model(env)

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=self.memory_limit, window_length=self.window_length)
        policy = BoltzmannQPolicy()

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory,
                            nb_steps_warmup=self.nb_steps_warmup, target_model_update=1e-2, policy=policy,
                            batch_size=self.batch_size, train_interval=self.train_interval)
        self.dqn.compile(optimizer=Adam(learning_rate=self.lr), metrics=['mae'])

    def start_step_policy(self, observation):
        """Custom policy for random decisions for warm-up."""
        log.info("Random action")
        _ = observation
        action = self.env.action_space.sample()
        return action

    def train(self, env_name):
        """Train a model"""
        # initiate training loop
        self.dqn.fit(self.env, nb_steps=self.nb_steps, visualize=False,
                     verbose=2)

        # Save the architecture
        #dqn_json = self.model.to_json()
        #with open("dqn_{}_json.json".format(env_name), "w") as json_file:
        #    json.dump(dqn_json, json_file)

        # After training is done, we save the final weights.
        self.dqn.save_weights('dqn_{}_weights.h5'.format(env_name), overwrite=True)

        # Finally, evaluate our algorithm for 5 episodes.
        self.dqn.test(self.env, nb_episodes=5, visualize=False)

    def load(self, env_name):
        """Load a model"""
        # Load the architecture
        #with open('dqn_{}_json.json'.format(env_name), 'r') as architecture_json:
        #    dqn_json = json.load(architecture_json)
        
        #self.model = model_from_json(dqn_json)
        #self.model.load_weights('dqn_{}_weights.h5'.format(env_name))
        return("wip")

    def play(self, nb_episodes=5, render=False):
        """Let the agent play"""
        memory = SequentialMemory(limit=self.memory_limit, window_length=self.window_length)
        policy = BoltzmannQPolicy()

        nb_actions = self.env.action_space.n

        self.dqn = DQNAgent(model=self.model, nb_actions=nb_actions, memory=memory,
                            nb_steps_warmup=self.nb_steps_warmup, target_model_update=1e-2, policy=policy,
                            processor=CustomProcessor(),
                            batch_size=self.batch_size, train_interval=self.train_interval,
                            enable_double_dqn=self.enable_double_dqn)
        self.dqn.compile(optimizer=Adam(learning_rate=self.lr), metrics=['mae'])

        self.dqn.test(self.env, nb_episodes=nb_episodes, visualize=render)

    def action(self, action_space, observation, info):
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = observation  # not using the observation for random decision
        _ = info

        this_player_action_space = {Action.FOLD, Action.CHECK, Action.CALL, Action.RAISE_POT, Action.RAISE_HALF_POT,
                                    Action.RAISE_2POT}
        _ = this_player_action_space.intersection(set(action_space))

        action = None
        return action



class CustomProcessor(Processor):
    """The agent and the environment"""

    def __init__(self):
        """Initialize properties"""
        self.legal_moves_limit = None

    def process_state_batch(self, batch):
        """Remove second dimension to make it possible to pass it into CNN"""
        return np.squeeze(batch, axis=1)

    def process_info(self, info):
        if 'legal_moves' in info.keys():
            self.legal_moves_limit = info['legal_moves']
        else:
            self.legal_moves_limit = None
        return {'x': 1}  # Only arrays allowed it seems

    def process_action(self, action):
        """Find nearest legal action"""
        if 'legal_moves_limit' in self.__dict__ and self.legal_moves_limit is not None:
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


class TrumpPolicy(BoltzmannQPolicy):
    """Custom policy when making decision based on neural network."""

    def select_action(self, q_values):
        """Return the selected action"""
        assert q_values.ndim == 1
        q_values = q_values.astype('float64')
        nb_actions = q_values.shape[0]

        exp_values = np.exp(np.clip(q_values / self.tau, self.clip[0], self.clip[1]))
        probs = exp_values / np.sum(exp_values)
        action = np.random.choice(range(nb_actions), p=probs)
        log.info(f"Chosen action by keras-rl {action} - probabilities: {probs}")
        return action