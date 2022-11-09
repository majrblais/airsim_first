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

model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.00025,
    verbose=1,
    batch_size=32,
    train_freq=4,
    target_update_interval=10000,
    learning_starts=0,
    buffer_size=5000,
    max_grad_norm=10,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    device="cuda",
    tensorboard_log="./tb_logs/",
)


model.learn(
    total_timesteps=5e5
)