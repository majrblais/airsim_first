from Custom_env import custom_environment_v09

from pettingzoo.utils.conversions import aec_to_parallel
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv, PettingZooEnv
from tianshou.policy import MultiAgentPolicyManager, RandomPolicy,A2CPolicy,RainbowPolicy,IQNPolicy, TD3Policy
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
        torch.save((policy.policies['DroneFollower0']).state_dict(), ckpt_path)
        ckpt_path = os.path.join('./log/', f"checkpoint_{epoch}_1.pth")
        torch.save((policy.policies['DroneFollower1']).state_dict(), ckpt_path)
    return ckpt_path

# ======== tensorboard logging setup =========
log_path = os.path.join('./log', 'test_1', 'ppo')
writer = SummaryWriter(log_path)
writer.add_text("args", 'test')
logger = TensorboardLogger(writer)

#create train/test env
env=PettingZooEnv(custom_environment_v09.env(ip_address="127.0.0.1"))

net = Net((512,512,3),4, hidden_sizes=[128,128], device='cuda').to('cuda')
actor = Actor(net, 4, device='cuda').to('cuda')
critic = Critic(net, device='cuda').to('cuda')
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

net2 = Net((512,512,3),4, hidden_sizes=[128,128], device='cuda').to('cuda')
actor2 = Actor(net2, 4, device='cuda').to('cuda')
critic2 = Critic(net2, device='cuda').to('cuda')
actor_critic2 = ActorCritic(actor2, critic2)
optim2 = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)


dist = torch.distributions.Categorical

agent_learn = PPOPolicy(actor, critic, optim,dist,action_space=4)
agent_opponent = PPOPolicy(actor2, critic2, optim2,dist,action_space=4)


agents = [agent_opponent,agent_learn]
policy = MultiAgentPolicyManager(agents, env)

train_envs = DummyVectorEnv([lambda: PettingZooEnv(custom_environment_v07.env(ip_address="127.0.0.1",val=False))])
#test_envs = DummyVectorEnv([lambda: PettingZooEnv(custom_environment_v06.env(ip_address="127.0.0.1",val=True))])

train_collector = Collector(policy,train_envs,VectorReplayBuffer(5000, len(train_envs)),exploration_noise=True)
#test_collector = Collector(policy, test_envs, exploration_noise=True)
#(policy.policies['DroneFollower0']).load_state_dict(torch.load('./log/checkpoint_35_0.pth'))
#(policy.policies['DroneFollower1']).load_state_dict(torch.load('./log/checkpoint_35_1.pth'))

result = onpolicy_trainer(policy,train_collector,test_collector=None, max_epoch=1000, step_per_epoch=1500, step_per_collect=128, episode_per_test=5, batch_size=16,update_per_step=0.9,repeat_per_collect=1,logger=logger,save_checkpoint_fn=save_checkpoint_fn)
print(results)
#torch.save((policy.policies['DroneFollower0']).state_dict(), 'policya_0.pth')
#torch.save((policy.policies['DroneFollower1']).state_dict(), 'policya_1.pth')