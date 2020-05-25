import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


torch.manual_seed(999)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):
    """
    Actor (Policy) Network.
    """

    def __init__(self, state_dim, action_dim, fc1_dim,fc2_dim, fc3_dim):
        """Initialize parameters and build model.
        :state_dim (int): Dimension of each state
        :action_dim (int): Dimension of each action
        """
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)

        self.fc3_dim = fc3_dim

        if fc3_dim != 0:
            self.fc2 = nn.Linear(fc1_dim, fc2_dim)
            self.fc3 = nn.Linear(fc2_dim, fc3_dim)
            self.fc4 = nn.Linear(fc3_dim, action_dim)
        else:
            self.fc2 = nn.Linear(fc1_dim, fc2_dim)
            self.fc3 = nn.Linear(fc2_dim, action_dim)

        self.reset_parameters()
        
    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        if self.fc3_dim >0:
            self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
            self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        else:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """
        Maps a state to actions
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        probs = F.softmax( self.fc3(x), dim=1 )

        dist = Categorical( probs )

        action = dist.sample()

        #print (action)

        log_prob = dist.log_prob( action )
        entropy = dist.entropy()
        return action.float()              
        # if self.fc3_dim >0:
        #     x = F.relu(self.fc3(x))
        #     return F.tanh(self.fc4(x))
        # else:
        #     return F.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """
    Critic (State-Value) Network.
    """

    def __init__(self, state_dim, action_dim,fc1_dim,fc2_dim, fc3_dim):
        """
        Initialize parameters and build model
        :state_dim (int): Dimension of each state
        :action_dim (int): Dimension of each action
        """
        super(CriticNetwork, self).__init__()
        self.state_fc = nn.Linear(state_dim, fc1_dim)
        self.fc1 = nn.Linear(1+fc1_dim, fc2_dim)
        self.fc3_dim = fc3_dim

        if fc3_dim != 0:
            self.fc2 = nn.Linear(fc2_dim, fc3_dim)
            self.fc3 = nn.Linear(fc3_dim, fc1_dim)
            self.fc4 = nn.Linear(fc1_dim, 1)
        else:
            self.fc2 = nn.Linear(fc2_dim, fc1_dim)
            self.fc3 = nn.Linear(fc1_dim, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        if self.fc3_dim >0:
            self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
            self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        else:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Maps a state-action pair to Q-values
        """
        state, action = state.squeeze(), action.squeeze()
        x = F.relu(self.state_fc(state))
        # print("state:",state.size())
        # print("action",action.size())
        action = action.view(-1,1)
        x = torch.cat((x, action),dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.fc3_dim >0:
            x = F.relu(self.fc3(x))
            return self.fc4(x)
        else:
            return self.fc3(x)
        
        

    
