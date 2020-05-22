
import torch
from torch import nn
from torchviz import make_dot, make_dot_from_trace
from graphviz import Digraph
from agent import DDPGAgent


x = torch.randn(5,24).cuda() 
y = torch.randn(5,2).cuda() 

agent = DDPGAgent(state_dim=24, action_dim=2)

dot = make_dot(agent.critic_local(x,y), params=dict(agent.critic_local.named_parameters()))
dot.format = 'png'
dot.render("static/ddpg_critic_model")

dot = make_dot(agent.actor_local(x), params=dict(agent.actor_local.named_parameters()))
dot.format = 'png'
dot.render("static/ddpg_actor_model")
