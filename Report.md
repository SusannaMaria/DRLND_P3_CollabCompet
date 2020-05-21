# Implementation
After the benchmark description, I decided to use the DDPG (Deep Deterministic Policy Gradient) algorithm (Two agents who played together but trained independently). After a number of failed trainings over 1500 episodes, I decided on an approach where one agent should play both sides simultaneously during training.  https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/ddpg_agent.py and https://github.com/qiaochen/DDPG_MultiAgent -related to the integration of a solo agent into the tennis environment.

# Base Agent: DDPG 
Paper: https://arxiv.org/pdf/1509.02971.pdf 

To solve the problem I used the implementation of the Deep Deterministic Policy Gradient (DDPG) algorithm already available for project P2. [Continuous Control)(https://github.com/SusannaMaria/DRLND_P2_ContinuousControl/blob/master/Report.md)

My Agent implementation: [agent.py](agent.py). 
It was very helpful to study https://towardsdatascience.com/deep-deterministic-policy-gradients-explained-2d94655a9b7b

![](static/ddpg.png)

## Characteristics
**Target network**: Using two deep networks θ- and θ for actor. Using the first one to retrieve Q values while the second one includes all updates in the training. After some updates, we synchronize θ- with θ. The purpose is to fix the Q-value targets temporarily so we don’t have a moving target to chase. In addition, parameter changes do not impact θ- immediately and therefore even the input may not be 100% iid (Independent and identically distributed random variables)

**Experience replay**: Put the transitions into a buffer and take a sample from a mini-batch of 128-size samples from this buffer to train the deep network. This forms an input data set that is more stable for training. As the randomly sample of the playback buffer, the data is more independent of each other and closer to the iid (independent and identically distributed).

**Ornstein-Uhlenbeck process** The Ornstein-Uhlenbeck Process generates noise that is correlated with the previous noise, as to prevent the noise from canceling out or *freezing* the overall dynamics <cite>https://www.quora.com/Why-do-we-use-the-Ornstein-Uhlenbeck-Process-in-the-exploration-of-DDPG/answer/Edouard-Leurent?ch=10&share=4b79f94f&srid=udNQP</cite>

## Multi Agent Approach with DDPG 
Initially I used two independent DDPG agents to solve the tennis problem. After many failures with very low rewards over more than 2000 episodes, I got the advice to use one instance of a DDPG agent for both sides to teach themselves to play tennis on their own. The agent could be trained successfully with this approach but showed great instabilities and basically every training with different hyper parameters led to similar results and the agent collapsed to zero after a very good reward.
### Soloplay DDPG
A DDPG agent plays both sides during the training of the tennis game and thus learns to play both sides. Since the environment rewards cooperative play (keeping the ball in the air for a long time) this seemed like a worthwhile approach.

The chosen approach of the solo play of a DDPG is by far not perfect but I could train the agent to handle the task.

A typical behavior of the training. It shows no increase in the reward over a long period of time and then rises explosively and then collapses again soon. It was interesting that different results were achieved with the same hyperparameter definitions.

![](02Run/rewards.png)

### Solving the problem with stabiliity ... or not


# Models

# Training
|    |   epsilon |   epsilon_decay |   gamma |   lr_decay |   main_n_loop |   num_episodes |   random_seed |   replay_buff_size |   soft_update_tau |   test_n_run |
|---:|----------:|----------------:|--------:|-----------:|--------------:|---------------:|--------------:|-------------------:|------------------:|-------------:|
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|  0 |         1 |           0.995 |    0.95 |      0.995 |            10 |           2500 |           999 |              1e+06 |             0.001 |          100 |
|    |   actor_layer_dim_1 |   actor_layer_dim_2 |   actor_layer_dim_3 |   batch_size |   critic_layer_dim_1 |   critic_layer_dim_2 |   critic_layer_dim_3 |   lr_actor |   lr_critic |   scores_min |   scores_max |   scores_mean |   scores_std |
|---:|--------------------:|--------------------:|--------------------:|-------------:|---------------------:|---------------------:|---------------------:|-----------:|------------:|-------------:|-------------:|--------------:|-------------:|
|  0 |                  64 |                 128 |                   0 |           64 |                   64 |                  128 |                    0 |     1e-05  |      0.0001 |       0      |       1.9529 |       0.34578 |    0.592145  |
|  0 |                  64 |                 128 |                   0 |          128 |                   64 |                  128 |                    0 |     1e-05  |      0.0001 |       0.0403 |       0.8662 |       0.17942 |    0.237737  |
|  0 |                  64 |                 128 |                   0 |           64 |                   64 |                  128 |                    0 |     0.0001 |      0.0001 |       0.0194 |       2.601  |       0.78306 |    0.856658  |
|  0 |                  64 |                 128 |                   0 |          128 |                   64 |                  128 |                    0 |     0.0001 |      0.0001 |       0      |       2.331  |       0.49189 |    0.79874   |
|  0 |                  64 |                 128 |                  64 |           64 |                   64 |                  128 |                    0 |     0.0001 |      0.0001 |       0      |       2.15   |       0.40662 |    0.683908  |
|  0 |                  64 |                 128 |                  64 |          128 |                   64 |                  128 |                    0 |     0.0001 |      0.0001 |       0      |       2.3709 |       0.52985 |    0.702423  |
|  0 |                  64 |                 128 |                  64 |          256 |                   64 |                  128 |                    0 |     0.0001 |      0.0001 |       0      |       2.27   |       0.8188  |    0.886559  |
|  0 |                  64 |                 128 |                 128 |           64 |                   64 |                  128 |                    0 |     0.0001 |      0.0001 |       0      |       1.1311 |       0.20758 |    0.331744  |
|  0 |                  64 |                 128 |                 128 |           64 |                   64 |                  128 |                    0 |     1e-05  |      0.0001 |       0.002  |       0.1561 |       0.07861 |    0.0412998 |


# Analyis
# Tournament
