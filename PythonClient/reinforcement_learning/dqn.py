import setup_path
import gym
import airgym
import time

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from airgym.envs import custom_env

env = custom_env.AirSimcustomEnv(ip_address="127.0.0.1",step_length=2, image_shape=(84, 84, 1),)
#env = AirSimDroneEnv(ip_address="127.0.0.1",step_length=0.5,image_shape=(84, 84, 1),)

env = DummyVecEnv([lambda: env])
env = VecTransposeImage(env)

model = DQN("CnnPolicy",env,learning_starts=0,verbose=1,device="cuda",)

model.learn(total_timesteps=5e5)