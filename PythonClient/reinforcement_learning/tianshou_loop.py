from Custom_env import custom_environment_v03
#env=custom_environment_v03.env(ip_address="127.0.0.1")
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

# Step 1: Load the PettingZoo environment
env=custom_environment_v03.env(ip_address="127.0.0.1")

from pettingzoo.classic import chess_v5
#env = chess_v5.env(render_mode="human")

#env=aec_to_parallel(env)
# Step 2: Wrap the environment for Tianshou interfacing
env = PettingZooEnv(env)

device='cuda'

net = Net((512,512,3), hidden_sizes=[64, 64], device=device)
actor = Actor(net, 4, device=device).to(device)
critic = Critic(net, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

# PPO policy
dist = torch.distributions.Categorical
policy = PPOPolicy(actor, critic, optim, dist, action_space=4, deterministic_eval=True)

# Step 3: Define policies for each agent
policies = MultiAgentPolicyManager([policy, policy], env)
# Step 4: Convert the env to vector format
env = DummyVectorEnv([lambda: env])

# Step 5: Construct the Collector, which interfaces the policies with the vectorised environment
collector = Collector(policies, env)

# Step 6: Execute the environment with the agents playing for 1 episode, and render a frame every 0.1 seconds
result = collector.collect(n_episode=10, render=0.1)