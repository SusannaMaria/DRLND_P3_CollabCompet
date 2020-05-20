import json
import pandas as pd    
import numpy as np   
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

def readproject(filename):
    with open(filename, 'r') as f:
        s = f.read()
        s = s.replace("\'args\'", "\"args\"")
        s = s.replace("\'rewards\'", "\"rewards\"")
        s = s.replace("\'scores\'", "\"scores\"")    
        s = s.replace("Namespace(", "\"")
        s = s.replace(")", "\"")    
        ds=json.loads(s)
        # with open("p.json", 'w') as w:
        #     w.write(s)

    args = ds['args']
    argdict = {}

    for arg in args.split(", "):
        key,val = arg.split("=")
        if val.find("1e") == 0:
            try:
                val =int(val)
            except ValueError:
                val =float(val)

        elif val.find(".") >= 0:
            try:
                val =float(val)
            except ValueError:
                pass

        elif val.isnumeric():
            val = int(val)

        argdict[key] = val
    
    ds['args'] = argdict
    
    return ds


projects = ["01Run_standard","02Run_Batchsize128","03Run_lractor1e-4","04Run_",
            "05Run_ActorLayer_4","06Run_ActorLayer_4_batchsize128","07Run_",
            "08Run_Actor64","09Run_Actor128"]

dfs_args = []
for project in projects:
    ds = readproject("{}/project.json".format(project))
    df_args = pd.DataFrame([ds['args']])

    scores_min = np.min(ds['scores'])
    scores_max = np.max(ds['scores'])
    scores_mean = np.mean(ds['scores'])
    scores_std = np.std(ds['scores'])

    df_args['scores_min']= scores_min
    df_args['scores_max']= scores_max
    df_args['scores_mean']= scores_mean
    df_args['scores_std']= scores_std

    df_scores = pd.DataFrame(ds['scores'])
    df_scores = df_scores.T

    dfs_args.append(df_args)
    df_rewards = pd.DataFrame(ds['rewards'])   
    df_rewards = df_rewards.T


df_args_fin = pd.concat(dfs_args)
del df_args_fin['model_path']
print(df_args_fin)


# #df = dft.cumsum()
# plt.figure()
# dft.plot()
# plt.show()
