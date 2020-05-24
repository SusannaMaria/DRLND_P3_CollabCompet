from unityagents import UnityEnvironment
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from agent import MADDPG
import argparse
from tqdm import tqdm
import os

unity_environment_path = "Soccer_Linux/Soccer.x86_64"
env = UnityEnvironment(file_name=unity_environment_path)


def eval():
    # print the brain names
    print(env.brain_names)

    # set the goalie brain
    g_brain_name = env.brain_names[0]
    g_brain = env.brains[g_brain_name]

    # set the striker brain
    s_brain_name = env.brain_names[1]
    s_brain = env.brains[s_brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)

    # number of agents
    num_g_agents = len(env_info[g_brain_name].agents)
    print('Number of goalie agents:', num_g_agents)
    num_s_agents = len(env_info[s_brain_name].agents)
    print('Number of striker agents:', num_s_agents)

    # number of actions
    g_action_size = g_brain.vector_action_space_size
    print('Number of goalie actions:', g_action_size)
    s_action_size = s_brain.vector_action_space_size
    print('Number of striker actions:', s_action_size)

    # examine the state space
    g_states = env_info[g_brain_name].vector_observations
    g_state_size = g_states.shape[1]
    print('There are {} goalie agents. Each receives a state with length: {}'.format(
        g_states.shape[0], g_state_size))
    s_states = env_info[s_brain_name].vector_observations
    s_state_size = s_states.shape[1]
    print('There are {} striker agents. Each receives a state with length: {}'.format(
        s_states.shape[0], s_state_size))

    for i in range(2):                                         # play game for 2 episodes
        # reset the environment
        env_info = env.reset(train_mode=False)
        # get initial state (goalies)
        g_states = env_info[g_brain_name].vector_observations
        # get initial state (strikers)
        s_states = env_info[s_brain_name].vector_observations
        # initialize the score (goalies)
        g_scores = np.zeros(num_g_agents)
        # initialize the score (strikers)
        s_scores = np.zeros(num_s_agents)
        while True:
            # select actions and send to environment
            g_actions = np.random.randint(g_action_size, size=num_g_agents)
            print("#", g_actions, "#",g_action_size)
            
            s_actions = np.random.randint(s_action_size, size=num_s_agents)
            print("ö", g_actions, "ö",s_action_size)

            g_actions = np.array([0,0])
            s_actions = np.array([3,2.4])
            actions = dict(zip([g_brain_name, s_brain_name],
                               [g_actions, s_actions]))
            print(actions)
            env_info = env.step(actions)
            
            # get next states
            g_next_states = env_info[g_brain_name].vector_observations
            s_next_states = env_info[s_brain_name].vector_observations

            # get reward and update scores
            g_rewards = env_info[g_brain_name].rewards
            s_rewards = env_info[s_brain_name].rewards
            g_scores += g_rewards
            s_scores += s_rewards

            # check if episode finished
            done = np.any(env_info[g_brain_name].local_done)

            # roll over states to next time step
            g_states = g_next_states
            s_states = s_next_states

            # exit loop if episode finished
            if done:
                break
        print('Scores from episode {}: {} (goalies), {} (strikers)'.format(
            i+1, g_scores, s_scores))


def train(args):

    # print the brain names
    print(env.brain_names)

    # set the goalie brain
    g_brain_name = env.brain_names[0]
    g_brain = env.brains[g_brain_name]

    # set the striker brain
    s_brain_name = env.brain_names[1]
    s_brain = env.brains[s_brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)

    # number of agents
    num_g_agents = len(env_info[g_brain_name].agents)
    print('Number of goalie agents:', num_g_agents)
    num_s_agents = len(env_info[s_brain_name].agents)
    print('Number of striker agents:', num_s_agents)

    # number of actions
    g_action_size = g_brain.vector_action_space_size
    print('Number of goalie actions:', g_action_size)
    s_action_size = s_brain.vector_action_space_size
    print('Number of striker actions:', s_action_size)

    # examine the state space
    g_states = env_info[g_brain_name].vector_observations
    g_state_size = g_states.shape[1]
    print('There are {} goalie agents. Each receives a state with length: {}'.format(
        g_states.shape[0], g_state_size))
    s_states = env_info[s_brain_name].vector_observations
    s_state_size = s_states.shape[1]
    print('There are {} striker agents. Each receives a state with length: {}'.format(
        s_states.shape[0], s_state_size))

    g_agent = MADDPG(g_state_size,
                     g_action_size,
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
                     critic_layer_dim_3=args.critic_layer_dim_3)

    s_agent = MADDPG(s_state_size,
                     s_action_size,
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
                     critic_layer_dim_3=args.critic_layer_dim_3)

    total_g_rewards = []
    total_s_rewards = []
    avg_s_scores = []
    avg_g_scores = []
    max_avg_s_score = -1
    max_avg_g_score = -1
    max_g_score = -1
    max_s_score = -1
    threshold_init = 20
    noise_t = args.epsilon
    noise_decay = args.epsilon_decay
    latest_avg_g_score = -1
    latest_avg_s_score = -1
    # for early-stopping training if consistently worsen for # episodes
    worsen_s_tolerance = threshold_init
    worsen_g_tolerance = threshold_init
    for i_episode in range(1, 1+args.num_episodes):

        # reset the environment
        env_info = env.reset(train_mode=True)
        # get initial state (goalies)
        g_states = env_info[g_brain_name].vector_observations
        # get initial state (strikers)
        s_states = env_info[s_brain_name].vector_observations
        # initialize the score (goalies)
        g_scores = np.zeros(num_g_agents)
        # initialize the score (strikers)
        s_scores = np.zeros(num_s_agents)

        done = False
        while not done:
            # select an action
            g_actions = g_agent.act(g_states, noise_t)
            s_actions = s_agent.act(s_states, noise_t)
            g_actions_am = np.array([np.argmax(g_actions[0]),np.argmax(g_actions[1])])
            s_actions_am = np.array([np.argmax(s_actions[0]),np.argmax(s_actions[1])])
            actions = dict(zip([g_brain_name, s_brain_name],
                               [g_actions_am, s_actions_am]))
            env_info = env.step(actions)

            # get next states
            g_next_states = env_info[g_brain_name].vector_observations
            s_next_states = env_info[s_brain_name].vector_observations

            # get reward and update scores
            g_rewards = env_info[g_brain_name].rewards
            s_rewards = env_info[s_brain_name].rewards
            g_scores += g_rewards
            s_scores += s_rewards

            done = np.any(env_info[g_brain_name].local_done)

            g_agent.update(g_states, g_actions, g_rewards,
                           g_next_states, [done, done])
            s_agent.update(s_states, s_actions, s_rewards,
                           s_next_states, [done, done])

            noise_t *= noise_decay
            g_states = g_next_states
            s_states = s_next_states

        episode_g_score = np.max(g_scores)
        episode_s_score = np.max(s_scores)
        
        total_s_rewards.append(episode_s_score)
        total_g_rewards.append(episode_g_score)

        print('\rScores from episode {}: {} (goalies), {} (strikers) | Avg Score: {:.4f} (goalies), {:.4f} (strikers)'.format(
            i_episode, episode_g_score, episode_s_score, latest_avg_g_score, latest_avg_s_score))

        if max_s_score <= episode_s_score:
            max_s_score = episode_s_score
            # save best model so far
            s_agent.save(
                "socccer_chkpts/{}/{:02d}_strikers_best_model.checkpoint".format(args.model_path, args.loop_counter))

        if max_g_score <= episode_g_score:
            max_g_score = episode_g_score
            # save best model so far
            g_agent.save(
                "socccer_chkpts/{}/{:02d}_goalies_best_model.checkpoint".format(args.model_path, args.loop_counter))

        # record avg score for the latest 100 steps
        if len(total_s_rewards) >= args.test_n_run:
            latest_avg_s_score = sum(
                total_s_rewards[(len(total_s_rewards)-args.test_n_run):]) / args.test_n_run
            avg_s_scores.append(latest_avg_s_score)

            if max_avg_s_score <= latest_avg_s_score:           # record better results
                worsen_s_tolerance = threshold_init           # re-count tolerance
                max_avg_s_score = latest_avg_s_score
            else:
                if max_avg_s_score > 0.5:
                    worsen_tolerance -= 1                   # count worsening counts
                    print("Loaded from last best model.")
                    # continue from last best-model
                    s_agent.load(
                        "socccer_chkpts/{}/{:02d}_strikers_best_model.checkpoint".format(args.model_path, args.loop_counter))
                if worsen_s_tolerance <= 0:                   # earliy stop training
                    print("Early Stop Training.")
                    break
        # record avg score for the latest 100 steps
        if len(total_g_rewards) >= args.test_n_run:
            latest_avg_g_score = sum(
                total_g_rewards[(len(total_g_rewards)-args.test_n_run):]) / args.test_n_run
            avg_g_scores.append(latest_avg_g_score)

            if max_avg_g_score <= latest_avg_g_score:           # record better results
                worsen_g_tolerance = threshold_init           # re-count tolerance
                max_avg_g_score = latest_avg_g_score
            # else:
            #     if max_avg_g_score > 0.5:
            #         worsen_g_tolerance -= 1                   # count worsening counts
            #         print("Loaded from last best model.")
            #         # continue from last best-model
            #         g_agent.load(
            #             "chkpts/{}/{:02d}_goalies_best_model.checkpoint".format(args.model_path, args.loop_counter))
            #     if worsen_g_tolerance <= 0:                   # earliy stop training
            #         print("Early Stop Training.")
            #         break                
    del g_agent
    del s_agent
    return (total_g_rewards, total_s_rewards) 


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
    parser.add_argument('--batch_size', default=int(128),
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

    project = {}
    project["args"] = args
    project["scores"] = []
    project["rewards"] = []

    #eval()

    try:
        os.mkdir("socccer_chkpts/{}".format(args.model_path))
    except OSError:
        pass

    for i in range(args.main_n_loop):
        args.loop_counter = i
        print(args)
        (reward_g, reward_s) = train(args)
        project["rewards_s"].append(reward_s)
        project["rewards_g"].append(reward_g)
        # score = test(args)
        # project["scores"].append(score)
        #print(score)

    f = open("socccer_chkpts/{}/project.json".format(args.model_path), "w")
    f.write(str(project))
    f.close()

    env.close()

    print(np.min(project["scores"]))
    print(np.max(project["scores"]))
    print(np.mean(project["scores"]))
