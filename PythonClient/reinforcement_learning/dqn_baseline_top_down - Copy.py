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

import gym
import numpy as np
import torch

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.discrete import Actor, Critic



from airgym.envs import custom_env_top_baseline_rnd
env = custom_env_top_baseline_rnd.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=1, image_shape=(512, 512, 3),)

env = DummyVectorEnv([lambda: env])
env = VecTransposeImage(env)

model = DQN("CnnPolicy",env,learning_starts=100,batch_size=64, exploration_initial_eps=0.9,exploration_fraction=0.15,exploration_final_eps=0.01,verbose=1,device="cuda",buffer_size=5000,)

model.learn(total_timesteps=3e4)
model.save("dqn_airsim_drone_policy")