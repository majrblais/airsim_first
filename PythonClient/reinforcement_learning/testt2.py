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

from tianshou.data import Collector, VectorReplayBuffer,ReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic

from airgym.envs import single
env=single.AirSimcustomEnv_base(ip_address="127.0.0.1")
train_envs = DummyVectorEnv([lambda: single.AirSimcustomEnv_base(ip_address="127.0.0.1")])

feature_net = Net((512,512,3),4, hidden_sizes=[128,128], device='cuda',softmax=False)
net = FullQuantileFunction(feature_net,(512,512,3),4, hidden_sizes=[128,128],num_cosines=64,device=args.device)

optim = torch.optim.Adam(net.parameters(), lr=0.0001)
fraction_net = FractionProposalNetwork(args.num_fractions, net.input_dim)
fraction_optim = torch.optim.RMSprop(fraction_net.parameters(), lr=0.0000025)

policy = FQFPolicy(
    net,
    optim,
    fraction_net,
    fraction_optim,
    0.9,
    32,
    10,
    5,
    target_update_freq=125
).to('cuda')

# collector
train_collector = Collector(policy, train_envs, ReplayBuffer(2000))


# trainer
result = onpolicy_trainer( policy,train_collector,test_collector=None,max_epoch=100,step_per_epoch=2500,repeat_per_collect=2,episode_per_test=1,batch_size=256,step_per_collect=2000,logger=logger,save_checkpoint_fn=save_checkpoint_fn)

print(result)