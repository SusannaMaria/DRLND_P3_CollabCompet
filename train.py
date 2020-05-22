from unityagents import UnityEnvironment
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from agent import MADDPG
import argparse
from tqdm import tqdm
import os

unity_environment_path = "./Tennis_Linux/Tennis.x86_64"


def train(args):
    """main training function

    Instanciate MADDPG Class with hyperparameter set and perform training loop

    Arguments:
        args {argparse} -- parameter set, which can be controlled from outside by script call

    Returns:
        array of scores -- Array of episodic scores during training
    """
    # prepare environment
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

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

    # Crate instance of MADDPG Class, mainly possible to control the model dimensions, learnrates and batch sizes
    agent = MADDPG(state_size,
                   action_size,
                   lr_actor=args.lr_actor,
                   lr_critic=args.lr_critic,
                   lr_decay=args.lr_decay,
                   replay_buff_size=args.replay_buff_size,
                   gamma=args.gamma,
                   batch_size=args.batch_size,
                   random_seed=args.random_seed,
                   soft_update_tau=args.soft_update_tau,
                   actor_layer_dim_1=args.actor_layer_dim_1,
                   actor_layer_dim_2=args.actor_layer_dim_2,
                   actor_layer_dim_3=args.actor_layer_dim_3,
                   critic_layer_dim_1=args.critic_layer_dim_1,
                   critic_layer_dim_2=args.critic_layer_dim_2,
                   critic_layer_dim_3=args.critic_layer_dim_3

                   )

    total_rewards = []
    avg_scores = []
    max_avg_score = -1
    max_score = -1
    threshold_init = 20
    noise_t = args.epsilon
    noise_decay = args.epsilon_decay
    latest_avg_score = -1
    # for early-stopping training if consistently worsen for # episodes
    worsen_tolerance = threshold_init
    for i_episode in range(1, 1+args.num_episodes):

        env_inst = env.reset(train_mode=True)[
            brain_name]    # reset the environment
        states = env_inst.vector_observations                # get the current state
        # initialize score array
        scores = np.zeros(num_agents)
        dones = [False]*num_agents
        while not np.any(dones):
            # select an action
            actions = agent.act(states, noise_t)
            # send the action to the environment
            env_inst = env.step(actions)[brain_name]
            next_states = env_inst.vector_observations       # get the next state
            rewards = env_inst.rewards                       # get the reward
            dones = env_inst.local_done                      # see if episode has finished
            agent.update(states, actions, rewards, next_states, dones)

            noise_t *= noise_decay
            scores += rewards                                # update scores
            states = next_states

        episode_score = np.max(scores)
        total_rewards.append(episode_score)
        print("\rEpisodic {} Score: {:.4f}\t Avg Score: {:.4f}".format(
            i_episode, episode_score, latest_avg_score), end=' ')

        if max_score <= episode_score:
            max_score = episode_score
            # save best model so far
            agent.save(
                "chkpts/{}/{:02d}_best_model.checkpoint".format(args.model_path, args.loop_counter))

        # record avg score for the latest 100 steps
        if len(total_rewards) >= args.test_n_run:
            latest_avg_score = sum(
                total_rewards[(len(total_rewards)-args.test_n_run):]) / args.test_n_run
            avg_scores.append(latest_avg_score)

            if max_avg_score <= latest_avg_score:           # record better results
                worsen_tolerance = threshold_init           # re-count tolerance
                max_avg_score = latest_avg_score
            else:
                if max_avg_score > 0.5:
                    worsen_tolerance -= 1                   # count worsening counts
                    print("Loaded from last best model.")
                    # continue from last best-model
                    agent.load(
                        "chkpts/{}/{:02d}_best_model.checkpoint".format(args.model_path, args.loop_counter))
                if worsen_tolerance <= 0:                   # earliy stop training
                    print("Early Stop Training.")
                    break
    del agent
    return total_rewards


def test(args):
    """Test function for solo play of DDPG Agent

    args.test_n_run determined how many episodes are played (default = 100)

    Arguments:
        args {argparse} -- parameter set, which can be controlled from outside by script call

    Returns:
        average score -- Average score of game
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # dim of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # dim of the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = MADDPG(state_size, action_size, actor_layer_dim_1=args.actor_layer_dim_1,
                   actor_layer_dim_2=args.actor_layer_dim_2,
                   actor_layer_dim_3=args.actor_layer_dim_3,
                   critic_layer_dim_1=args.critic_layer_dim_1,
                   critic_layer_dim_2=args.critic_layer_dim_2,
                   critic_layer_dim_3=args.critic_layer_dim_3)

    agent.load(
        "chkpts/{}/{:02d}_best_model.checkpoint".format(args.model_path, args.loop_counter))

    test_scores = []
    for i_episode in tqdm(range(1, 1+args.test_n_run)):
        # initialize the scores
        scores = np.zeros(num_agents)
        env_info = env.reset(train_mode=True)[
            brain_name]   # reset the environment
        states = env_info.vector_observations               # get the current states
        dones = [False]*num_agents
        while not np.any(dones):
            actions = agent.act(states)                     # select actions
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


class RawTextArgumentDefaultsHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawTextHelpFormatter
):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=RawTextArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_episodes', default=int(2500),
                        type=int, help=''' ''')
    parser.add_argument('--lr_actor', default=float(1e-4),
                        type=float, help=''' ''')
    parser.add_argument('--lr_critic', default=float(1e-4),
                        type=float, help=''' ''')
    parser.add_argument('--lr_decay', default=float(0.995),
                        type=float, help=''' ''')
    parser.add_argument('--replay_buff_size',
                        default=int(1e6), type=int, help=''' ''')
    parser.add_argument('--gamma', default=float(0.95),
                        type=float, help=''' ''')
    parser.add_argument('--batch_size', default=int(64),
                        type=int, help=''' ''')
    parser.add_argument('--random_seed', default=int(999),
                        type=int, help=''' ''')
    parser.add_argument('--soft_update_tau',
                        default=float(1e-3), type=float, help=''' ''')
    parser.add_argument('--model_path', default='training', help=''' ''')
    parser.add_argument('--test_n_run', default=int(100),
                        type=int, help=''' ''')
    parser.add_argument('--epsilon', default=float(1.0),
                        type=float, help=''' ''')
    parser.add_argument('--epsilon_decay',
                        default=float(.995), type=float, help=''' ''')
    parser.add_argument('--main_n_loop', default=int(10),
                        type=int, help=''' ''')
    parser.add_argument('--actor_layer_dim_1',
                        default=int(64), type=int, help=''' ''')
    parser.add_argument('--actor_layer_dim_2',
                        default=int(128), type=int, help=''' ''')
    parser.add_argument('--actor_layer_dim_3',
                        default=int(0), type=int, help=''' ''')
    parser.add_argument('--critic_layer_dim_1',
                        default=int(64), type=int, help=''' ''')
    parser.add_argument('--critic_layer_dim_2',
                        default=int(128), type=int, help=''' ''')
    parser.add_argument('--critic_layer_dim_3',
                        default=int(0), type=int, help=''' ''')

    args = parser.parse_args()

    env = UnityEnvironment(file_name=unity_environment_path)

    project = {}
    project["args"] = args
    project["scores"] = []
    project["rewards"] = []

    try:
        os.mkdir(args.model_path)
    except OSError:
        pass

    for i in range(args.main_n_loop):
        args.loop_counter = i
        print(args)
        reward = train(args)
        project["rewards"].append(reward)
        score = test(args)
        project["scores"].append(score)
        print(score)

    f = open("chkpts/{}/project.json".format(args.model_path), "w")
    f.write(str(project))
    f.close()

    env.close()

    print(np.min(project["scores"]))
    print(np.max(project["scores"]))
    print(np.mean(project["scores"]))
