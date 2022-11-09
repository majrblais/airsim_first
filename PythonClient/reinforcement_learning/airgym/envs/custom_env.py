import airsim
import os
from PIL import Image
import numpy as np
from gym import spaces
# = airsim.MultirotorClient()
#client.confirmConnection()
import time
#print(client.getMultirotorState())
from airgym.envs.airsim_env import AirSimEnv
import gym

class AirSimcustomEnv(gym.Env):
    def __init__(self,ip_address="127.0.0.1", step_length=10, image_shape=(84, 84, 1),):
        self.step_length = step_length
        self.image_shape = image_shape
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        
        #Current state of agent
        self.state = {"position": np.zeros(3),"collision": False,"prev_position": np.zeros(3),"val_seg": 0.0,}
        
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        
        self.action_space = spaces.Discrete(5)
        #self.action_space = spaces.Discrete(7)
        success = self.drone.simSetSegmentationObjectID("oil", 54);
        print(success)
        self.image_request = airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False)
        
        self.counter=0
        
        
        
        self._setup_flight()
    
    
    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def __del__(self):
        self.drone.reset()    
    
    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(0, 0, -50, 5).join()
        #self.drone.moveByVelocityAsync(1,1, 1, 5).join()
        
        
    def step(self, action):
        print(action)
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state
        
    def _compute_reward(self):
        #Reward thresh and neg reward in case lower than thres
        thresh_dist = 40
        z = -10
        
        #desired location
        pts = [np.array([100, 100, -50.0]),]
        quad_pt = np.array(list((self.state["position"].x_val,self.state["position"].y_val,self.state["position"].z_val,)))
        
        print("test")
        if self.state["collision"]:
            reward = -100
        else:
            dist = 10000000
            for i in range(0, len(pts)):
                dist = np.linalg.norm(pts[0][0:1]-quad_pt[0:1])
                
            reward = -dist

        self.counter+=1
        done = 0
        print(reward)
        if self.counter >= 10:
            done = 1
            self.counter=0

        return reward, done    
    
    def _get_obs(self):
        from PIL import Image
        #get image segmentation map
        responses = self.drone.simGetImages([self.image_request])
        #transform
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        img_rgb = np.flipud(img_rgb)
        image = Image.fromarray(img_rgb)
        im_final = np.array(image.resize((84, 84)).convert("L"))
        
        
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision

        return im_final.reshape([84, 84, 1])
    
    def _do_action(self, action):
        offset = self.interpret_action(action)
        #pos=self.drone.getMultirotorState().kinematics_estimated.position
        #self.drone.moveToPositionAsync(pos.x_val + offset[0],pos.y_val + offset[1],pos.z_val + offset[2], 5).join()
        
        #Move by velocity
        pos=self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(pos.x_val + offset[0],pos.y_val + offset[1],pos.z_val + offset[2], 5).join()
        

        
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
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        #move left    
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        #move up    
        #elif action == 5:
        #    quad_offset = (0, 0, -self.step_length)
        #dont move    
        else:
            quad_offset = (0, 0, 0)

        return quad_offset


#F=AirSimDroneEnv()
#print(F.drone.getMultirotorState().kinematics_estimated.position)
#print("forward")
#F._do_action(0)
#time.sleep(2.5)
#print(F.drone.getMultirotorState().kinematics_estimated.position)
#F._get_obs()
#print(F._compute_reward())

#print("right")
#F._do_action(1)
#time.sleep(2.5)
#print("down")
#F._do_action(2)
#time.sleep(2.5)
#print("backward")
#F._do_action(3)
#time.sleep(2.5)
#print("left")
#F._do_action(4)
#time.sleep(2.5)
#print("up")
#F._do_action(5)
#time.sleep(2.5)
#print("Nothing")
#F._do_action(7)
