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
Our goal largely consists of making a successful AI to play poker. This AI will be trained using pure reinforcement learning, meaning we will not use a database, and instead will train through self-play. To do so, we will apply various reinforcement learning techniques and models. We will begin with simpler models such as [DQN](https://www.adaltas.com/en/2019/01/09/applying-deep-reinforcement-learning-poker/) and build our way towards more complicated algorithms such as [Monte Carlo Counterfactual Regret Minimization](https://www.adaltas.com/en/2019/01/09/applying-deep-reinforcement-learning-poker/). Along the way, we will compare the results from different models to analyze how well each model played poker. 

## Solution Concept


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
