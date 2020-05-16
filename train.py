from actor_critic_ctl import actor_critic_train, actor_critic_test
#from td3_agent import AgentTD3
from ddpg_agent import AgentDDPG
from unityagents import UnityEnvironment
import pandas as pd
import matplotlib.pyplot as plt

def plot_train(df):
    """Print min max plot of DQN Agent analytics

    Params
    ======
        df :    Dataframe with scores
    """
    ax = df.plot(x='episode', y='mean')
    plt.fill_between(x='episode', y1='min', y2='max',
                     color='lightgrey', data=df)
    x_coordinates = [0, 150]
    y_coordinates = [30, 30]
    plt.plot(x_coordinates, y_coordinates, color='red')
    plt.show()

env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64')
# get the default brain

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]

print('There are {} agents. Each observes a state with length: {}'.format(
    states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

# agent = AgentTD3(state_size=state_size, action_size=action_size,
#                   random_seed=1, cfg_path="config.ini")


# # Train
# df = actor_critic_train(env, agent, agent.cfg, brain_name, num_agents)
# metadata = agent.cfg_items
# filename = 'data_{}.hdf5'.format(agent.name)
# store = pd.HDFStore(filename)
# store.put('dataset_01', df)
# store.get_storer('dataset_01').attrs.metadata = metadata
# store.close()
# plot_train(df)
