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

### DQN Explanation:
Deep Q Networks, or DQN, is a model-free, online, off-policy RL method that uses a target critic and an experience buffer. It allows agents to learn from high-dimensional inputs, like images, and perform complex tasks. DQN considers future rewards when updating the value function for a given state-action pair. This means that the final reward doesn't need to be waited for until the end of the episode. 
DQN agents are value-based RL agents that train a critic to estimate the expected discounted cumulative long-term reward when following the optimal policy. In Q-learning, the agent learns an ideal Q function that converts a pair of states and actions into an expected cumulative reward.

### DDQN vs DQN
Double Deep Q Networks, or DDQN, builds upon the DQN architecture by introducing the double Q-learning approach, using two Q-networks to provide more accurate Q-value estimates and address the overestimation bias present in standard DQN.

### Goals
Our goal largely consists of making a successful AI to play poker. This AI will be trained using pure reinforcement learning, meaning we will not use a database, and instead will train through self-play. To do so, we will apply various reinforcement learning techniques and models. We will begin with simpler models such as [DQN](https://www.adaltas.com/en/2019/01/09/applying-deep-reinforcement-learning-poker/) and build our way towards more complicated algorithms such as [Monte Carlo Counterfactual Regret Minimization](https://www.adaltas.com/en/2019/01/09/applying-deep-reinforcement-learning-poker/). (Edit: Never had time to implment more complex algorithms for this Poker AI. Implemented DDQN and DQN using 2 different libraries which gave us enough conclusive results for a semester.) Furthermore, we will compare the results from different models to analyze how well each model played poker. 

## Solution Concept
![PokerProPlusSoftwareDiagram](https://github.com/ASasamori/PokerProPlus/assets/76934261/16368a44-0ec5-496b-8980-0a6d29d8338f)
1. Provided
- Within the open-sourced game environment contains the states, actions, and a few agents. 

2. Improved Actions
- The current actions within the environment consist of fold, check, call, raise_pot, raise_half_pot, and raise_2pot.
- We would like to add some actions such as raise_small (raise by big blind amount), raise_x (where x is a constant), or raise_% (where % is a % of the players current equity)

3. agent_keras_rl_dqn.py
- DQN implementation using the KerasRL library.

4. agent_dqn_pytorch.py
- DQN implementation using the PyTorch library.

5. agent_keras_rl_ddqn.py
- DDQN implementation using the KerasRL library. (Very similar to DQN approach)

7. Other_agents
- Agents with algorithms such as Monte Carlo Counterfactual Regret Minimization(MCCRM), Deep Deterministic Policy Gradient (DDPG), or Advantage Actor-Critic (A2C).
- However, we are unsure if we have the time or resources to devise, create, and train these models so we have left it as "Other_agents" for now.

6. Hyperparameter Tuning
- Each agent will have unique parameters such as in the agent_keras_rl_dqn having parameters such as nb_max_start_steps (max number of random actions at the beginning), nb_steps_warmup (before training starts), nb_steps (number of total steps), memory_limit (limit memory of experience replay), and batch_size (number of items sampled from memory to train).
- We will tune these parameters to find the parameters that will provide the greatest rewards. 


## Finished Product
### At time of Due of Date (May 3rd, 2024) :
 - Custom DQN model
   - Keras RL Implementation
   - PyTorch implementation
 - Custom DDQN model (in Keras RL)
 - Expansion of states and actions within the provided game environment
 
 ### Ways to fix in the future:
 - Recreate the game environment
   - env.py, enums.py, main.py, are all buggy code
   - Would like to have a way to have a pretrained model (with preset weights, etc.) to be available to upload
 - Easier UI and display of how each game updates the model
 - Allow for a Universal Arena
   - Test your model to see how well it Performs against DQN (Keras RL + PyTorch) and DDQN (Keras) that we have implemented


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
##### agent_keypress.py
This file provides a Player class template so a person can play in the game against the bots by providing keyboard inputs.


### gym_env (Folder):
gym_env is the inherited library from the original neuron_poker repo. It provides the game environment setup for the agent to obtain observations from. It also provides a set of actions that agents can take. 
#### env.py
This file handles simulating a poker game including taking in the actions, finding legal actions, calculating rewards.
##### cycle.py
This file handles cycling between players, rotating the positions (dealer, Big blind Small blind), whose turn it currently is, how many moves per hand.
##### enums.py 
This files stores the enumerate for the Actions and Stages of the game.
##### rendering.py
This file displays the game so one can see the players actions and the cards more easily.

# Usage 
Building project is not different from Neuron Poker Repository: https://github.com/dickreuter/neuron_poker, and since we inherited this environment, you might have to change directory to build within /neuron_poker-master.
- Install Python 3.11, (can also use PyCharm IDE).
- Install Poetry with ``curl -sSL https://install.python-poetry.org | python3 -``
- Create a virtual environment with ``poetry env use python3.11``
- Activate it with ``poetry shell``
- Install all required packages with ``poetry install --no-root``
- Run 6 random players playing against each other:
  ``poetry run python main.py selfplay random --render`` or
- To manually control the players:``poetry run python main.py selfplay keypress --render``
- In order to use the C++ version of the equity calculator, you will also need to install Visual Studio 2019 (or GCC over Cygwin may work as well). To use it, use the -c option when running main.py.
- For more advanced users: ``poetry run python main.py selfplay dqn_train -c`` will start training the deep Q agent with C++ Monte Carlo for faster calculation
