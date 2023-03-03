from Custom_env import custom_environment_v05

from pettingzoo.utils.conversions import aec_to_parallel
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy,A2CPolicy 
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic
import torch

import argparse
import os
from copy import deepcopy
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    DQNPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net

def reward_metric(rews):
    print(rews)
    return rews[:, 1]

# ======== tensorboard logging setup =========
log_path = os.path.join('./log', 'test', 'dqn')
writer = SummaryWriter(log_path)
writer.add_text("args", 'test')
logger = TensorboardLogger(writer)

#create train/test env
env=PettingZooEnv(custom_environment_v05.env(ip_address="127.0.0.1"))
net = Net((512,512,3),4, hidden_sizes=[128, 128, 128, 128], device='cuda').to('cuda')
optim = torch.optim.Adam(net.parameters(), lr=0.0001)
agent_learn = DQNPolicy(net,optim,target_update_freq =100)
agent_opponent = deepcopy(agent_learn)

agents = [agent_opponent, agent_learn]
policy = MultiAgentPolicyManager(agents, env)


#net = Net((512,512,3),4, hidden_sizes=[64, 64, 64], device='cuda').to('cuda')
#actor = Actor(net, 4, device='cuda').to('cuda')
#critic = Critic(net, device='cuda').to('cuda')
#actor_critic = ActorCritic(actor, critic)
#optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0001)


#dist = torch.distributions.Categorical
#agent_learn = A2CPolicy(actor, critic, optim,dist)
#agent_opponent = deepcopy(agent_learn)


agents = [agent_learn,agent_opponent]
policy = MultiAgentPolicyManager(agents, env)

train_envs = DummyVectorEnv([lambda: PettingZooEnv(custom_environment_v05.env(ip_address="127.0.0.1"))])
#test_envs = DummyVectorEnv([lambda: PettingZooEnv(custom_environment_v05.env(ip_address="127.0.0.1",val=True))])

train_collector = Collector(policy,train_envs,VectorReplayBuffer(1000, len(train_envs)),exploration_noise=True)
#test_collector = Collector(policy, test_envs, exploration_noise=True)


result = offpolicy_trainer(policy,train_collector,test_collector=None, max_epoch=100, step_per_epoch=1000, step_per_collect=25, episode_per_test=10, batch_size=8,update_per_step=0.5,logger=logger)


torch.save((policy.policies['DroneFollower0']).state_dict(), 'policya_0.pth')
torch.save((policy.policies['DroneFollower1']).state_dict(), 'policya_1.pth')