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

class DQN(nn.Module):
    def __init__(self,input_shape,nb_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_shape,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512, nb_actions)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.2)
        x = self.fc4(x)
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

        self.dqn = None # Probably should not be None?
        self.model = DQN(input_shape, nb_actions)
        self.env = env

        if load_model:
            self.load(load_model)

    def initiate_agent(self, nb_actions):
        """initiate a deep Q agent"""

        self.model = Sequential()
        self.model.add(Dense(512, activation='relu', input_shape=env.observation_space))  # pylint: disable=no-member
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(nb_actions, activation='linear'))

        # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
        # even the metrics!
        memory = SequentialMemory(limit=memory_limit, window_length=window_length)  # pylint: disable=unused-variable
        policy = TrumpPolicy()  # pylint: disable=unused-variable

    def action(self, action_space, observation, info):  # pylint: disable=no-self-use,unused-argument
        """Mandatory method that calculates the move based on the observation array and the action space."""
        _ = (observation, info)  # not using the observation for random decision
        action = None

        # decide if explore or explot

        # forward

        # save to memory

        # backward
        # decide what to use for training
        # update model
        # save weights

        return action
