from Custom_env import custom_environment_v06

from pettingzoo.utils.conversions import aec_to_parallel
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy,A2CPolicy,RainbowPolicy,IQNPolicy, TD3Policy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import QRDQNPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import torch

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple
from tianshou.policy import FQFPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import FractionProposalNetwork, FullQuantileFunction
import gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer, onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

def reward_metric(rews):
    print(rews)
    return rews[:, 1]

def save_checkpoint_fn(epoch, env_step, gradient_step):
    # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    # Example: saving by epoch num
    #ckpt_path = os.path.join('./log/', "checkpoint.pth")
    ckpt_path=''
    if epoch%5==0:
        ckpt_path = os.path.join('./log/', f"checkpoint_{epoch}_0.pth")
        torch.save(policy.state_dict(), ckpt_path)
    return ckpt_path

# ======== tensorboard logging setup =========
log_path = os.path.join('./log', 'test_1', 'ppo')
writer = SummaryWriter(log_path)
writer.add_text("args", 'test')
logger = TensorboardLogger(writer)

#create train/test env
env=PettingZooEnv(custom_environment_v06.env(ip_address="127.0.0.1"))

'''
net = Net((512,512,3),4, hidden_sizes=[128, 128, 128], device='cuda').to('cuda')
actor = Actor(net, 4, device='cuda').to('cuda')
optim = torch.optim.Adam(actor.parameters(), lr=0.00003)
critica = Critic(net, device='cuda').to('cuda')
optima = torch.optim.Adam(critica.parameters(), lr=0.00003)
criticb = Critic(net, device='cuda').to('cuda')
optimb = torch.optim.Adam(criticb.parameters(), lr=0.00003)
net2 = Net((512,512,3),4, hidden_sizes=[128, 128, 128], device='cuda').to('cuda')
actor2 = Actor(net2, 4, device='cuda').to('cuda')
optim2 = torch.optim.Adam(actor2.parameters(), lr=0.00003)
critic2a = Critic(net2, device='cuda').to('cuda')
optim2a = torch.optim.Adam(critic2a.parameters(), lr=0.00003)
critic2b = Critic(net2, device='cuda').to('cuda')
optim2b = torch.optim.Adam(critic2b.parameters(), lr=0.00003)

net = Net((512,512,3),4, hidden_sizes=[128, 128, 128], device='cuda').to('cuda')
actor = Actor(net, 4, device='cuda').to('cuda')
optim = torch.optim.Adam(actor.parameters(), lr=0.00003)

net2 = Net((512,512,3),4, hidden_sizes=[128, 128, 128], device='cuda').to('cuda')
actor2 = Actor(net2, 4, device='cuda').to('cuda')
optim2 = torch.optim.Adam(actor2.parameters(), lr=0.00003)

agent_learn = QRDQNPolicy(actor,optim,discount_factor =0.9,target_update_freq=500)
agent_opponent = QRDQNPolicy(actor2,optim2,discount_factor =0.9,target_update_freq=500)

'''

#dist = torch.distributions.Categorical

feature_net = Net((512,512,3), hidden_sizes=[128,128,128], device='cuda',softmax=False)
net = FullQuantileFunction(feature_net,action_shape=4, hidden_sizes=[128,128,128],num_cosines=64,device='cuda')
optim = torch.optim.Adam(net.parameters(), lr=0.00003)
fraction_net = FractionProposalNetwork(32, net.input_dim)
fraction_optim = torch.optim.RMSprop(fraction_net.parameters(), lr=0.0000025)

feature_net2 = Net((512,512,3), hidden_sizes=[128,128,128], device='cuda',softmax=False)
net2 = FullQuantileFunction(feature_net2,action_shape=4, hidden_sizes=[128,128,128],num_cosines=64,device='cuda')
optim2 = torch.optim.Adam(net2.parameters(), lr=0.00003)
fraction_net2 = FractionProposalNetwork(32, net2.input_dim)
fraction_optim2 = torch.optim.RMSprop(fraction_net2.parameters(), lr=0.0000025)

agent_opponent = FQFPolicy(net,optim,fraction_net,fraction_optim,0.97,32,0.005,3,target_update_freq=250).to('cuda')
agent_learn = FQFPolicy(net2,optim2,fraction_net2,fraction_optim2,0.97,32,0.005,3,target_update_freq=250).to('cuda')
agent_opponent.set_eps(0.15)
agent_learn.set_eps(0.15)

agents = [agent_opponent,agent_learn]
policy = MultiAgentPolicyManager(agents, env)


train_envs = DummyVectorEnv([lambda: PettingZooEnv(custom_environment_v06.env(ip_address="127.0.0.1",val=False))])
#test_envs = DummyVectorEnv([lambda: PettingZooEnv(custom_environment_v06.env(ip_address="127.0.0.1",val=True))])

train_collector = Collector(policy,train_envs,ReplayBuffer(5000),exploration_noise=True)
#test_collector = Collector(policy, test_envs, exploration_noise=True)
#policy.load_state_dict(torch.load('./log/checkpoint_35_0.pth'))

#result = onpolicy_trainer(policy,train_collector,test_collector=None, max_epoch=1000, step_per_epoch=1500, step_per_collect=128, episode_per_test=5, batch_size=16,update_per_step=0.9,repeat_per_collect=1,logger=logger,save_checkpoint_fn=save_checkpoint_fn)
result = offpolicy_trainer(policy,train_collector,test_collector=None, max_epoch=250, step_per_epoch=1000, step_per_collect=256, episode_per_test=0, batch_size=32,update_per_step=0.75,logger=logger,save_checkpoint_fn=save_checkpoint_fn)

print(results)
#torch.save((policy.policies['DroneFollower0']).state_dict(), 'policya_0.pth')
#torch.save((policy.policies['DroneFollower1']).state_dict(), 'policya_1.pth')