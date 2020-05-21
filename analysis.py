import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf


def draw(scores, path="fig.png", title="Performance", xlabel="Episode #", ylabel="Score"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path, dpi=300)


def readproject(filename):
    with open(filename, 'r') as f:
        s = f.read()
        s = s.replace("\'args\'", "\"args\"")
        s = s.replace("\'rewards\'", "\"rewards\"")
        s = s.replace("\'scores\'", "\"scores\"")
        s = s.replace("Namespace(", "\"")
        s = s.replace(")", "\"")
        ds = json.loads(s)

    args = ds['args']
    argdict = {}

    for arg in args.split(", "):
        key, val = arg.split("=")
        if val.find("1e") == 0:
            try:
                val = int(val)
            except ValueError:
                val = float(val)

        elif val.find(".") >= 0:
            try:
                val = float(val)
            except ValueError:
                pass

        elif val.isnumeric():
            val = int(val)

        argdict[key] = val

    ds['args'] = argdict

    return ds

if __name__ == "__main__":
        
    projects = ["01Run", "02Run", "03Run", "04Run", "05Run","06Run", "07Run", "08Run", "09Run"]

    dfs_args = []
    for project in projects:
        ds = readproject("{}/project.json".format(project))
        df_args = pd.DataFrame([ds['args']])

        scores_min = np.min(ds['scores'])
        scores_max = np.max(ds['scores'])
        scores_mean = np.mean(ds['scores'])
        scores_std = np.std(ds['scores'])

        df_args['scores_min'] = scores_min
        df_args['scores_max'] = scores_max
        df_args['scores_mean'] = scores_mean
        df_args['scores_std'] = scores_std

        df_scores = pd.DataFrame(ds['scores'])
        df_scores = df_scores.T

        dfs_args.append(df_args)
        df_rewards = pd.DataFrame(ds['rewards'])
        df_rewards = df_rewards.T
        draw(df_rewards, path="{}/rewards.png".format(project))

    col_useless = ['model_path', 'loop_counter']
    col_individual = ['actor_layer_dim_1', 'actor_layer_dim_2', 'actor_layer_dim_3', 'batch_size', 'critic_layer_dim_1',
                    'critic_layer_dim_2', 'critic_layer_dim_3', 'lr_actor', 'lr_critic', 'scores_min', 'scores_max', 'scores_mean', 'scores_std']
    col_common = ['epsilon', 'epsilon_decay', 'gamma', 'lr_decay', 'main_n_loop',
                'num_episodes', 'random_seed', 'replay_buff_size', 'soft_update_tau', 'test_n_run']

    df_args_fin = pd.concat(dfs_args)

    for c in col_useless:
        del df_args_fin[c]

    df_runs = df_args_fin.copy()

    for c in col_common:
        del df_runs[c]

    for c in col_individual:
        del df_args_fin[c]

    print(df_args_fin.T.to_markdown())
    print(df_runs.T.to_markdown())
