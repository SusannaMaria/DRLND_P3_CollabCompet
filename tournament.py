import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from models import ActorNetwork, CriticNetwork
from collections import deque, namedtuple
import random
import copy
from agent import DDPGAgent
from analysis import readproject
from tqdm import tqdm
from unityagents import UnityEnvironment

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = np.finfo(np.float32).eps.item()


class TourDDPG:
    """
    The Multi-Agent consisting of two DDPG Agents
    """

    def __init__(self, agent_1, agent_2
                 ):
        """
        Initialize constituent agents
        :args - tuple of parameters for DDPG Agent
                 (state_dim,
                 action_dim,
                 lr_actor,
                 lr_critic,
                 lr_decay,
                 replay_buff_size,
                 gamma,
                 batch_size,
                 random_seed, 
                 soft_update_tau)
        """
        super(TourDDPG, self).__init__()

        self.agent_1 = agent_1
        self.agent_2 = agent_2
        # the agent self-plays with itself
        self.adversarial_agents = [self.agent_1, self.agent_2]

    def get_actors(self):
        """
        get actors of all the agents in the MADDPG object
        """
        actors = [ddpg_agent.actor_local for ddpg_agent in self.adversarial_agents]
        return actors

    def get_target_actors(self):
        """
        get target_actors of all the agents in the MADDPG object
        """
        target_actors = [
            ddpg_agent.actor_target for ddpg_agent in self.adversarial_agents]
        return target_actors

    def act(self, states_all_agents, add_noise=False):
        """
        get actions from all agents in the MADDPG object
        """
        actions = [agent.act(state, add_noise) for agent, state in zip(
            self.adversarial_agents, states_all_agents)]
        return np.stack(actions, axis=0)

    def update(self, *experiences):
        """
        update the critics and actors of all the agents
        """
        states, actions, rewards, next_states, dones = experiences
        for agent_idx, agent in enumerate(self.adversarial_agents):
            state = states[agent_idx, :]
            action = actions[agent_idx, :]
            reward = rewards[agent_idx]
            next_state = next_states[agent_idx, :]
            done = dones[agent_idx]
            agent.update_model(state, action, reward, next_state, done)


def play(env, game, duration):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    test_scores = []
    for i_episode in tqdm(range(1, 1+duration)):
        # initialize the scores
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[
            brain_name]   # reset the environment
        states = env_info.vector_observations               # get the current states
        dones = [False]*num_agents
        while not np.any(dones):
            actions = game.act(states)                     # select actions
            # send the actions to the environment
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations      # get the next states
            rewards = env_info.rewards                      # get the rewards
            dones = env_info.local_done                     # see if episode has finished
            scores += rewards                               # update the scores
            # roll over the states to next time step
            states = next_states

        test_scores.append(np.max(scores))

    avg_score = sum(test_scores)/len(test_scores)
    print("Test Score: {}".format(avg_score))

    return avg_score


def loadagent(ckp_name, *args, **kargs):
    agent = DDPGAgent(*args, **kargs)
    agent.path = ckp_name
    actor_state_dict, critic_state_dict = torch.load(ckp_name)
    agent.actor_local.load_state_dict(actor_state_dict)
    agent.actor_target.load_state_dict(actor_state_dict)
    agent.critic_local.load_state_dict(critic_state_dict)
    agent.critic_target.load_state_dict(critic_state_dict)
    agent.lr_actor *= agent.lr_decay
    agent.lr_critic *= agent.lr_decay
    return agent


projects = ["01Run", "02Run", "03Run", "04Run", "05Run", "06Run", "07Run", "08Run", "09Run", "10Run"]

dfs_args = []

agents = []

unity_environment_path = "./Tennis_Linux/Tennis.x86_64"
env = UnityEnvironment(file_name=unity_environment_path)
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]

max_index = []
for project in projects:
    ds = readproject("chkpts/{}/project.json".format(project))    
    max_index.append(ds['scores'].index(max(ds['scores'])))

c = 0
for project in projects:
    ds = readproject("chkpts/{}/project.json".format(project))
    args = ds['args']
    ckp_name = "chkpts/{}/{:02d}_best_model.checkpoint".format(project, max_index[c])
    c+=1
    agent = loadagent(ckp_name,
                      state_dim=state_size,
                      action_dim=action_size,
                      actor_layer_dim_1=args['actor_layer_dim_1'],
                      actor_layer_dim_2=args['actor_layer_dim_2'],
                      actor_layer_dim_3=args['actor_layer_dim_3'],
                      critic_layer_dim_1=args['critic_layer_dim_1'],
                      critic_layer_dim_2=args['critic_layer_dim_2'],
                      critic_layer_dim_3=args['critic_layer_dim_3'])
    agents.append(agent)
match = 1
results = np.zeros((len(agents), len(agents)))

for idx, ag_x in enumerate(agents):
    for idy, ag_y in enumerate(agents):
        print("{}: {} vs. {}".format(match, ag_x.path, ag_y.path))
        game = TourDDPG(ag_x, ag_y)
        result = play(env, game, 100)
        results[idy][idx] = result
        print("{}: {} vs. {} = {}".format(match, ag_x.path, ag_y.path, result))
        f=open("tournament_result.txt", "a")
        f.write("{}: {} vs. {} = {}\r\n".format(match, ag_x.path, ag_y.path, result))
        f.close()
        match += 1

plt.imshow(results, cmap='winter', interpolation='nearest')
plt.show()

