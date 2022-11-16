import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from airgym.envs import custom_env_base

env = custom_env_base.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=1, image_shape=(128, 128, 1),)
#env = AirSimDroneEnv(ip_address="127.0.0.1",step_length=0.5,image_shape=(84, 84, 1),)

env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)

model = DQN("CnnPolicy",env,learning_starts=100,batch_size=64, exploration_initial_eps=0.9,exploration_fraction=0.15,exploration_final_eps=0.01,verbose=1,device="cuda",buffer_size=5000,)

model.learn(total_timesteps=2e4)
model.save("dqn_airsim_drone_policy")
model.learn(total_timesteps=2e4)
model.save("dqn_airsim_drone_policy_2")