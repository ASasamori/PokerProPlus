# PokerProPlus

## Collaborators

|                Name           |Email|
|-------------------------------|-----------------------------|
|     Andrew Sasamori           |     sasamori@bu.edu         |
|     Johnson Yang              |     johnsony@bu.edu         |
|     Mete Gumusayak            |     mgumus@bu.edu           |
|    Wai Yuen Cheng             |     wycheng@bu.edu          |

## Project Summary:
### Background
With the rapid growth and expansion of AI in the last few years, it is becoming a necessity to understand the core fundamentals behind newer models. To this extent, we will be looking to familiarize ourselves with reinforcement learning techniques to create an AI model that can play no-limit Texas hold'em poker at the same level as humans. As our main focus is on the deep learning aspect, we will use an [open-source poker environment](https://github.com/dickreuter/neuron_poker). This environment provides example agents, a simple render of the game, and a foundation for the states/actions. The environment also provides an example of a deep Q-Network (DQN) using Keras that we will familiarize ourselves with before implementing our own DQN and applying other reinforcement learning models and techniques to the environment. 

### Goals
Our goal largely consists of making a successful AI to play poker. This AI will be trained using pure reinforcement learning, meaning we will not use a database, and instead will train through self-play. To do so, we will apply various reinforcement learning techniques and models. We will begin with simpler models such as [DQN](https://www.adaltas.com/en/2019/01/09/applying-deep-reinforcement-learning-poker/) and build our way towards more complicated algorithms such as [Monte Carlo Counterfactual Regret Minimization](https://www.adaltas.com/en/2019/01/09/applying-deep-reinforcement-learning-poker/). Furthermore, we will compare the results from different models to analyze how well each model played poker. 

## Solution Concept
![PokerProPlusSoftwareDiagram](https://github.com/ASasamori/PokerProPlus/assets/76934261/16368a44-0ec5-496b-8980-0a6d29d8338f)
1. Provided
- Within the open-sourced game environment contains the states, actions, and a few agents. 

2. Improved Actions
- The current actions within the environment consist of fold, check, call, raise_pot, raise_half_pot, and raise_2pot.
- We would like to add some actions such as raise_small (raise by big blind amount), raise_x (where x is a constant), or raise_% (where % is a % of the players current equity)

3. Agent_dqn_sb
- Our implementation of DQN using Stable Baselines. Stable Baselines is a reinforcement learning library implemented using TensorFlow and OpenAI Gym, it provides users with a set of RL algorithms that are easily deployable.
- Meanwhile, DQN is a reinforcement learning algorithm used for training agents to make sequential decisions in environments with discrete action spaces. It works using Q-learning which aims to learn the best cumulative reward of taking an action in a given state. Instead of using a table of q-values (which would take too much space to compute as there is a near-infinite permutation of actions in a game of poker), we will be using a deep neural network to learn the best options for the next state and action. 

4. Agent_dqn_pytorch
- Our implementation of DQN using the pytorch library.

5. Other_agents
- Agents with algorithms such as Monte Carlo Counterfactual Regret Minimization(MCCRM), Deep Deterministic Policy Gradient (DDPG), or Advantage Actor-Critic (A2C).
- However, we are unsure if we have the time or resources to devise, create, and train these models so we have left it as "Other_agents" for now.

6. Hyperparameter Tuning
- Each agent will have unique parameters such as in the agent_keras_rl_dqn having parameters such as nb_max_start_steps (max number of random actions at the beginning), nb_steps_warmup (before training starts), nb_steps (number of total steps), memory_limit (limit memory of experience replay), and batch_size (number of items sampled from memory to train).
- We will tune these parameters to find the parameters that will provide the greatest rewards. 


## Acceptance Criteria
### Minimum Viable Product:
 - DQN model with Stable Baselines
 - Custom DQN model
 - Keras model with optimized hyperparameters
 - Expansion of states and actions within the provided game environment
 
 ### Stretch Goal:
 - A custom poker game environment
 - UI for the bot playing
 - A game of poker between bots built with different models


# Structures:
### Agents (Folder):
The agents folder contains different policies for the agent and training logics (if applicable).
##### agent_consider_equity.py:
This file provides a Player class template for agents that solely considers the amount of equity to output an action based on conditional logics.  
##### agent_keras_rl_ddqn.py:
This file provides a Player class template that offers starting code for implementing Double Deep Q Network RL approach. It also contains visualization functions for plotting results. 
##### agent_keras_rl_dqn.py:
This file provides a Player class template that offers starting code for implementing Deep Q Network RL approach. It also contains visualization functions for plotting results. 
##### agent_random.py:
This file provides a Player class template that outputs an action randomly without any computation or learning involved. 
##### agent_pytorch_dqn.py:
This file provides a Player class template that implements a Deep Q Network RL approach using pytorch and follows a similar format to the keras bot. 
#### agent_keypress.py
This file provides a Player class template so a person can play in the game against the bots by providing keyboard inputs


### gym_env:
gym_env is the inherited library from the original neuron_poker repo. It provides the game environment setup for the agent to obtain observations from. It also provides a set of actions that agents can take. 


# Usage
- Install Python 3.11, (can also use PyCharm IDE).
- Install Poetry with ``curl -sSL https://install.python-poetry.org | python3 -``
- Create a virtual environment with ``poetry env use python3.11``
- Activate it with ``poetry shell``
- Install all required packages with ``poetry install --no-root``
- Run 6 random players playing against each other:
  ``poetry run python main.py selfplay random --render`` or
- To manually control the players:``poetry run python main.py selfplay keypress --render``
- Example of genetic algorithm with self improvement: ``poetry run python main.py selfplay equity_improvement --improvement_rounds=20 --episodes=10``
- In order to use the C++ version of the equity calculator, you will also need to install Visual Studio 2019 (or GCC over Cygwin may work as well). To use it, use the -c option when running main.py.
- For more advanced users: ``poetry run python main.py selfplay dqn_train -c`` will start training the deep Q agent with C++ Monte Carlo for faster calculation
