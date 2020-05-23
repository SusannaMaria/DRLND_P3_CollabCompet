import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from sklearn import preprocessing

# 1: 01Run/00_best_model.checkpoint vs. 01Run/00_best_model.checkpoint = 2.1578000321611763
f = open("static/tournament_result.txt", "r")

agents1 = []
agents2 = []
results = []

for x in f:
    [match_nr,a1] = x.split(': ')
    [agent1,a2]= a1.split(' vs. ')
    [agent2,result] = a2.split(' = ')
    result = float(result)
    agents1.append(agent1)
    agents2.append(agent2)
    results.append(result)
    print(match_nr,agent1,agent2,result)

#normalized = (results-np.min(results))/(np.max(results)-np.min(results))

map = np.reshape(np.array(results), ( -1,10))

plt.imshow(map, cmap='RdBu', interpolation='nearest')
plt.colorbar()
plt.show()