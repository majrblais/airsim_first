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
        
        self.image_request = airsim.ImageRequest(3, airsim.ImageType.Scene, False, False)
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
        
        #set the base distance, take the desired point, current state (0,0,-20) and calculate the euclidean distance between them, save the best distance
        #pts = [np.array([90, 120, -20.0]),]
        #self.drone_state = self.drone.getMultirotorState()
        #self.state["position"] = self.drone_state.kinematics_estimated.position
        #quad_pt = np.array(list((self.state["position"].x_val,self.state["position"].y_val,self.state["position"].z_val,)))
        #dist = np.linalg.norm(pts[0][0:2]-quad_pt[0:2])
        #self.state["Prev_D"] = dist
        
        
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
            print("collision")
        elif self.state["position"].x_val>= 200 or self.state["position"].x_val <=-20 or self.state["position"].y_val>= 50 or self.state["position"].y_val <=-200:
            reward= -100
        else:
            #Calculate distance for current location
            #dist = 10000000
            #for i in range(0, len(pts)):
            print(pts[0][0:2])
            print(quad_pt[0:2])
            #pts_abs=[abs(i) for i in pts]
            
            dist = np.linalg.norm(pts[0][0:2]-quad_pt[0:2])
            print(self.drone_state.kinematics_estimated.position)
            print(dist)
            #Get reward based on distance
            reward = -dist / 100
            reward +=1
            
            #if current distance is smaller than the best: add 5 to reward and set new best, else remove 5.
            #if dist <= self.state["Prev_D"]:
            #    print("best")
             #   print(dist)
            #    reward += 1
            #    self.state["Prev_D"] = dist
            #else:
            #    reward -= 1
                
                #reward_speed = (np.linalg.norm([self.state["velocity"].x_val,self.state["velocity"].y_val])- 0.5)
                #reward = reward_dist + reward_speed+5
                
            #reward = -dist / 10
            
            
            
        print(reward)            
        done = 0
        if reward <=-10:
            done = 1
        return reward, done    
    
    def _get_obs(self):
        global val
        from PIL import Image
        import cv2
        #get image depth
        responses = self.drone.simGetImages([self.image_request])
        response = responses[0]
        
        #DEPTH SCENE
        #img1d = np.array(responses[0].image_data_float, dtype=np.float)
        #img1d = 255 / np.maximum(np.ones(img1d.size), img1d)
        #img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        #cv2.imwrite(str(val)+'_t.png', img2d)
        #image = Image.fromarray(img2d)
        #im_final = np.array(image.resize((128, 128)).convert("L"))
        #img = im_final.reshape([128, 128, 1])
        
        #RGB SCENE IMAGES
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        image = Image.fromarray(img_rgb)
        im_final = np.array(image.resize((128, 128)))
        #img_rgb = np.flipud(img_rgb)
        #image = Image.fromarray(img_rgb)
        # = np.array(image.convert("L"))
        #cv2.imwrite(str(val)+'_t.png', img_rgb)
        #im_final = np.array(image.resize((256, 256)).convert("L"))
        #img = im_final.reshape([256, 256, 3])
        #cv2.imwrite(str(val)+'_t.png', img)


        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        
        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        
        val+=1
        return im_final
    
    def _do_action(self, action):
        print(action)
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
