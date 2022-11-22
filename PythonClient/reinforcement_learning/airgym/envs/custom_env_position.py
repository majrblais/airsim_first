import airsim
import os
from PIL import Image
import numpy as np
from gym import spaces
import math
# = airsim.MultirotorClient()
#client.confirmConnection()
import time
#print(client.getMultirotorState())
from airgym.envs.airsim_env import AirSimEnv
import gym

val=0

class AirSimcustomEnv_base(gym.Env):
    def __init__(self,ip_address="127.0.0.1", step_length=1, image_shape=(84, 84, 1),):
        self.step_length = step_length
        self.image_shape = image_shape
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        
        #Current state of agent    
        self.state = {"position": np.zeros(3),"collision": False,"prev_position": np.zeros(3),"val_seg": 0.0, "Prev_D" : 0.0}
        
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        
        self.action_space = spaces.Discrete(4)
        
        #Used for segmentation map
        #success = self.drone.simSetSegmentationObjectID("oil", 54);
        #self.image_request = airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)
        
        self.image_request = airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)
        #self.image_request = airsim.ImageRequest(0, airsim.ImageType.DepthPerspective, True, False)
        self.counter=0
          
        self._setup_flight()
    
    
    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def __del__(self):
        self.drone.reset()    
    
    def _setup_flight(self):
        print("setting-up")
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        self.drone.moveToPositionAsync(0, 0, -30, 5).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5).join()
        

        
    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state
        
    def _compute_reward(self):
        #desired location & current location
        pts = [np.array([120, -120, -30.0]),]
        quad_pt = np.array(list((self.state["position"].x_val,self.state["position"].y_val,self.state["position"].z_val,)))
        
        #If collision or out of bounds
        if self.state["collision"]:
            reward = -100
            #print("collision")
        elif self.state["position"].x_val>= 200 or self.state["position"].x_val <=-20 or self.state["position"].y_val>= 50 or self.state["position"].y_val <=-200:
            reward= -100
        else:
            
            dist = np.linalg.norm(pts[0][0:2]-quad_pt[0:2])
            reward = -dist / 100
            reward +=1
            
            
        #print(reward)            
        done = 0
        if reward <=-10:
            done = 1
        return reward, done    
    
    def _get_obs(self):
        global val
        
        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        
        val+=1
        return self.drone_state.kinematics_estimated.position
    
    def _do_action(self, action):
        #print(action)
        offset = self.interpret_action(action)
        #pos=self.drone.getMultirotorState().kinematics_estimated.position
        #self.drone.moveToPositionAsync(pos.x_val + offset[0],pos.y_val + offset[1],pos.z_val, 25).join()
        
        #Move by velocity
        pos=self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(pos.x_val + offset[0],pos.y_val + offset[1],0, 5).join()
        

        
    def interpret_action(self, action):
        
        #move forward
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        #move right
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        #move down
        #elif action == 2:
        #    quad_offset = (0, 0, self.step_length)
        #move backward
        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        #move left    
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)
        #move up    
        #elif action == 5:
        #    quad_offset = (0, 0, -self.step_length)
        #dont move    
        else:
            quad_offset = (0, 0, 0)

        return quad_offset


