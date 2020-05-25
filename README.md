[//]: # (Image References)

[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Project 3: Collaboration and Competition

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

# Solution
## Tennis
It was great fun.

The solution is described in detail in [Report.md](Report.md).

To solve the task I decided to use a DDPG(Deep Deterministic Policy Gradient) [paper](https://arxiv.org/pdf/1509.02971.pdf) based agent, with which I could already gain experience in P2 [Link](https://github.com/SusannaMaria/DRLND_P2_ContinuousControl). The agent was trained in solo play and played the games with himself during the training. 
The main focus was on the determination of hyperparameters. I investigated in 10 variants of an already stable set of parameters. It was important to realize that the success of a training cannot be guaranteed, and that it was necessary to repeat the training with the same hyper parameters, as well as to fix the model weights and a rollback if the training result suddenly collapsed.

All in all I trained a DDPG agent so 10x10x2500 episodes. I recorded a few seconds of "endgame" between two DDPGs with different parameters trained to demonstrate the success of solving the task at hand. 

Two of my trained agents in action

![Trained Agent](static/game.gif)

It shows that two different behaviors of the agents was developed. "put in back" and a more aggressive behavior.  In the video, the sides are changed about halfway through, the behavior has not changed much.

* [Report.md](Report.md) The report for the submission of my solution for Project 3 of the Udacity Deep Reinforcement Learning course [DRLND](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) 
* [train.py](train.py) Training function and solo play test function. Function was executed with a [script](scripts/parameter_optimize.sh) to train with various different Hyperparameter sets
* [agent.py](agent.py) MADDG and DDPG implmentation 
* [models.py](models.py) Actor and Critic Models for DDPG
* [training_analysis.py](training_analysis.py) Analysis of 10x2500 episodic trained agents which I called **runs**
* [tournament.py](tournament.py) Playing a tournament of the bests of each **run** 10x10 games
* [tournament_analysis.py](tournament_analysis.py) Showing the results of the tournament as txt and a heatmap for the report.
* [single_match.py](single_match.py) Playing a single match between two agents, used to record a screencast which was converted later to a animated gif see: [makegif.sh](scripts/makegif.sh)

## Soccer
Based on the implementation for the tennis game, I implemented a system with 4 DDPG agents to handle the soccer game. Currently I'm evaluating the hyperparameters and let play 1000 episodes each for the training, there is still no knowledge from my side and a reasonable acting agent for Striker and Goalie

* [Soccer.py](Soccer.py) Training function.
* [agent_soccer.py](agent_soccer.py) DDPG Agent implementation
* [models_soccer.py](models_soccer.py) Actor and Critic Models for DDPG

It is important to mention that I chose a softmax policy instead of epislon greedy [info](https://ai.stackexchange.com/questions/17603/what-is-the-difference-between-the-epsilon-greedy-and-softmax-policies)
So I choose the random actions with probabilities proportional to their current values. 


# How to install
After reaching the Nanodegree of Udacity for Deep Reinforcement Learning, I decided to remove the installation guide for this project taken from Udacity. If I find the time I will provide a general guide that builds and installs the Unity ML Agents Environment from official sources without using information from Udacity - [https://github.com/Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)


I definitely recommend the Udacity course for Deep Reinforcement Learning. It took me 3 months to complete the course and I found the time and money a wise investment. Not so much for my career but for personal development.

https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893
