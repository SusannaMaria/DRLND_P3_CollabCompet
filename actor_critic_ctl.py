from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import os
import pandas as pd
from types import SimpleNamespace


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def actor_critic_train(env, agent, cfg, brain_name, num_agents,
                       print_every=1):
    """Deep Deterministic Policy Gradient (DDPG)

    Params
    ======
        agent                 : Instance of Agent
        cfg                   : Configuration of training 
        print_every (int)     : interval to display results

    """
    n_episodes = int(cfg['N_EPISODES'])
    max_t = int(cfg['MAX_T'])
    save_n_episodes = int(cfg['SAVE_N_EPISODES'])

    # list of mean scores from each episode
    mean_scores = []
    # list of lowest scores from each episode
    min_scores = []
    # list of highest scores from each episode
    max_scores = []
    # mean scores from most recent episodes
    df = pd.DataFrame(columns=['episode', 'duration',
                               'min', 'max', 'std', 'mean'])

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset environment
        # get current state for each agent
        states = env_info.vector_observations
        # initialize score for each agent
        scores = np.zeros(num_agents)
        agent.reset()
        start_time = time.time()
        for t in range(max_t):

            # select an action
            actions = agent.act(states, add_noise=True)
            # send actions to environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations    # get next state
            rewards = env_info.rewards                    # get reward
            # see if episode has finished
            dones = env_info.local_done

            # save experience to replay buffer, perform learning step at
            # defined interval
            for state, action, reward, next_state, done in zip(states, actions,
                                                               rewards,
                                                               next_states,
                                                               dones):
                agent.step(state, action, reward, next_state, done, t)

            states = next_states

            scores += rewards
            if np.any(dones):  # exit loop when episode ends
                break
        # save time needed for episode
        duration = time.time() - start_time
        # save lowest score for a single agent
        min_scores.append(np.min(scores))
        # save highest score for a single agent
        max_scores.append(np.max(scores))
        # save mean score for the episode
        mean_scores.append(np.mean(scores))

        df.loc[i_episode-1] = [i_episode] + list([round(duration),
                                                  np.min(scores),
                                                  np.max(scores),
                                                  np.std(scores),
                                                  np.mean(scores)])

        if i_episode % print_every == 0:
            print('\rEpisode {} ({} sec)  -- \tMin: {:.1f}\tMax: {:.1f}\tMean: {:.1f}'.format(
                  i_episode, round(duration), min_scores[-1], max_scores[-1],
                  mean_scores[-1]))

        if i_episode % save_n_episodes == 0:
            epi_str = "{:03}".format(i_episode)
            torch.save(agent.actor_local.state_dict(),
                       agent.name+"_"+epi_str+"_actor_ckpt.pth")
            torch.save(agent.critic_local.state_dict(),
                       agent.name+"_"+epi_str+"_critic_ckpt.pth")
    
    torch.save(agent.actor_local.state_dict(),
               agent.name+"_final_actor_ckpt.pth")
    torch.save(agent.critic_local.state_dict(),
               agent.name+"_final_critic_ckpt.pth")

    return df


def actor_critic_test(env, agent, cfg, ckpt, n_episodes=100):
    df = pd.DataFrame(columns=['episode', 'duration',
                               'min', 'max', 'std', 'mean'])

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    # print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    # print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    # print('There are {} agents. Each observes a state with length: {}'.format(
    #     states.shape[0], state_size))
    # print('The state for the first agent looks like:', states[0])

    ckpt_path = cfg['CKPT_PATH']
    actor_path = '{}/{}_{}_actor_ckpt.pth'.format(ckpt_path, agent.name, ckpt)
    critic_path = '{}/{}_{}_critic_ckpt.pth'.format(ckpt_path, agent.name, ckpt)
    if not os.path.exists(actor_path) or not os.path.exists(critic_path):
        print("Checkpoint(s) not found: {},{}".format(actor_path, critic_path))
        return df

    if torch.cuda.is_available():
        agent.actor_local.load_state_dict(torch.load(actor_path))
        agent.critic_local.load_state_dict(
            torch.load(critic_path))
    else:
        agent.actor_local.load_state_dict(torch.load(
            actor_path, map_location=lambda storage, loc: storage))
        agent.critic_local.load_state_dict(torch.load(
            critic_path, map_location=lambda storage, loc: storage))

    for i_episode in range(0, n_episodes):
        env_info = env.reset(train_mode=True)[
            brain_name]     # reset the environment
        # get the current state (for each agent)
        states = env_info.vector_observations
        # initialize the score (for each agent)
        scores = np.zeros(num_agents)
        start_time = time.time()
        while True:
            # select an action
            actions = agent.act(states, add_noise=False)
            # send all actions to tne environment
            env_info = env.step(actions)[brain_name]
            # get next state (for each agent)
            next_states = env_info.vector_observations
            # get reward (for each agent)
            rewards = env_info.rewards
            dones = env_info.local_done  # see if episode finished
            # update the score (for each agent)
            scores += rewards
            # roll over states to next time step
            states = next_states
            if np.any(dones):  # exit loop if episode finished
                break
        duration = time.time() - start_time
        
        df.loc[i_episode] = [i_episode+1] + list([round(duration),
                                                  np.min(scores),
                                                  np.max(scores),
                                                  np.std(scores),
                                                  np.mean(scores)])

        print('\rEpisode {} ({} sec)\tMean: {:.1f}'.format(
            i_episode, round(duration), np.mean(scores)))

    return df
