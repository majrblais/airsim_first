import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from torchvision import models
import numpy as np
import numpy.ma as ma
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from airgym.envs import single


train_envs = DummyVectorEnv([lambda: DummyVectorEnv(single.AirSimcustomEnv_base(ip_address="127.0.0.1"))])

# model & optimizer
net = Net(env.observation_space.shape, hidden_sizes=[64, 64], device=device)
actor = Actor(net, env.action_space.n, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

# PPO policy
dist = torch.distributions.Categorical
policy = PPOPolicy(actor, critic, optim, dist, action_space=env.action_space, deterministic_eval=True)
        
          
# collector
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))


# trainer
result = onpolicy_trainer(
    policy,
    train_collector,
    test_collector=None,
    max_epoch=10,
    step_per_epoch=50000,
    repeat_per_collect=10,
    episode_per_test=10,
    batch_size=256,
    step_per_collect=2000,
    stop_fn=lambda mean_reward: mean_reward >= 195,
)
print(result)