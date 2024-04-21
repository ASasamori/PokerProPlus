import logging

import itertools
import gymnasium as gym
import numpy as np
import pandas as pd

from agents.agent_consider_equity import Player as EquityPlayer
from agents.agent_keras_rl_dqn import Player as DQNPlayer
from agents.agent_random import Player as RandomPlayer
from gym_env.env import HoldemTable as GameEnv

def dqn_train_keras_rl(model_name):

    game_stack_size = 100
    
    env = GameEnv(initial_stacks=game_stack_size, small_blind=1, big_blind=2, render=False, 
                  funds_plot=True, max_raises_per_player_round=2, 
                  use_cpp_montecarlo=False, raise_illegal_moves=False, 
                  calculate_equity=False, epochs_max = 10)
    
    env.add_player(EquityPlayer(name='Equity player 1', min_call_equity=.5, min_bet_equity=.7))
    env.add_player(EquityPlayer(name='Equity player 2', min_call_equity=.2, min_bet_equity=.3))
    env.add_player(RandomPlayer())
    env.add_player(RandomPlayer())
    env.add_player(RandomPlayer())
    env.add_player(PlayerShell(name="", stack_size=game_stack_size))

    env.reset()

    ddqn = DDQNPlayer()
    ddqn.initiate_agent(env)
    ddqn.train(env_name=model_name)