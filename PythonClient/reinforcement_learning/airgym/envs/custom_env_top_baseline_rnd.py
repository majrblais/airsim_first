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
import random
from datetime import datetime
random.seed(datetime.now().timestamp())
import torch
val=0

import cv2
import numpy as np
import numpy.ma as ma
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

class AirSimcustomEnv_base(gym.Env):
    def __init__(self,ip_address="127.0.0.1", step_length=1, image_shape=(84, 84, 1),):
        self.step_length = step_length
        self.image_shape = image_shape
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        
        #Current state of agent    
        self.state = {"position0": np.zeros(3),"position1": np.zeros(3),"collision0": False, "collision1": False, "collision2": False ,"prev_position0": np.zeros(3), "prev_position1": np.zeros(3)}
        
        self.detectors = {"detector1": [np.array([-45.0,20.0,-50.0])],"detector2": [np.array([30.0,40.0,-50.0])],"detector3": [np.array([-30.0,9.0,-50.0])],"detector4": [np.array([-13.0,-34.0,-50.0])],"detector5": [np.array([-45.0,-34.0,-50.0])],"detector6": [np.array([ -5.0, 8.0,-50.0])],"detector7": [np.array([31.0,-32.0,-50.0])],"detector8": [np.array([44.0,-37.0,-50.0])],"detector9": [np.array([-7.0,42.0,-50.0])] }
        self.tot=0
        
        self.current_drone="DroneFollower1"
        self.action_space = spaces.Discrete(4)
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.track = []
        
        self.pos_1=None
        self.pos_2=None
        
        self.curr_re1=-25
        self.curr_re2=-25
        
        self.max_re1=-25
        self.max_re2=-25
        
        self.image_request_seg = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Segmentation,False, False)
        self.image_request_rgb = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Scene,False, False)
        
        self._setup_flight()
    
    
    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def __del__(self):
        self.drone.reset()    
    
    def _setup_flight(self):
        print("setting-up")
        self.drone.reset()
        self.curr_re1=-25
        self.curr_re2=-25
        
        self.max_re1=-25
        self.max_re2=-25
        
        self.pos_1=random.choice(list(self.detectors.items()))
        self.pos_2=random.choice(list(self.detectors.items()))
        while self.pos_1==self.pos_2:
            self.pos_2=random.choice(list(self.detectors.items()))
        
        #Controllable setup
        self.drone.enableApiControl(True, "DroneFollower1")
        self.drone.enableApiControl(True, "DroneFollower2")
        
        self.drone.armDisarm(True, "DroneFollower1")
        self.drone.armDisarm(True, "DroneFollower2")
        
        #takeoff, move to side by side
        self.drone.takeoffAsync(vehicle_name="DroneFollower1").join()
        self.drone.moveToPositionAsync(5, 0, -50, 10, vehicle_name="DroneFollower1").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower1").join()
        
        self.drone.takeoffAsync(vehicle_name="DroneFollower2").join()
        self.drone.moveToPositionAsync(-5, 0, -51, 10, vehicle_name="DroneFollower2").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower2").join()
        
        
    def step(self, action):
        
        if self.current_drone=="DroneFollower1":
            self._do_action(action)
        
            obs = self._get_obs()
            reward, done = self._compute_reward()
            self.current_drone="DroneFollower2"
            return obs, reward, done, self.state
            
        elif self.current_drone=="DroneFollower2":
            self._do_action(action)

            obs = self._get_obs()
            reward, done = self._compute_reward()
            self.current_drone="DroneFollower1"
            return obs, reward, done, self.state
            
        else:
            print("error step")
            exit()
        
    def _compute_reward(self):
        #desired location & current location
        
        drone_name=self.current_drone
        if drone_name == "DroneFollower1":
            end_pts = self.pos_1[1]
            quad_pt0 = np.array(list((self.state["position0"].x_val,self.state["position0"].y_val,self.state["position0"].z_val,)))
            #should theo not happen since it flys higher
            if self.state["collision0"]:
                reward = -100
                print("collisionFollower1")
                
            elif self.state["position0"].x_val>=44 or self.state["position0"].x_val<= -49 or self.state["position0"].y_val >=44 or self.state["position0"].y_val <=-49:
                reward = -100
                print("OoBFollower1")
            else:
                dist = np.linalg.norm(end_pts[0][0:2]-quad_pt0[0:2])
                reward = - np.sqrt(dist)+1
                self.curr_re1=reward
                if reward>=self.max_re1:
                    self.max_re1=reward
                
                print("reward:")
                print(reward)
                
            done=0
            if reward <=-25:
                done=1
            return reward,done
            
        elif drone_name == "DroneFollower2":
            end_pts = self.pos_2[1]
            quad_pt0 = np.array(list((self.state["position1"].x_val,self.state["position1"].y_val,self.state["position1"].z_val,)))
            #should theo not happen since it flys higher
            if self.state["collision0"]:
                reward = -100
                print("collisionFollower2")
                
            elif self.state["position0"].x_val>=44 or self.state["position1"].x_val<= -49 or self.state["position1"].y_val >=44 or self.state["position1"].y_val <=-49:
                reward = -100
                print("OoBFollower2")
            else:
                dist = np.linalg.norm(end_pts[0][0:2]-quad_pt0[0:2])
                reward = - np.sqrt(dist)+1
                self.curr_re2=reward
                if reward>=self.max_re2:
                    self.max_re2=reward
                
                print("reward:")
                print(reward)
                
            done=0
            if reward <=-25 or (self.curr_re1>=0.25 and self.curr_re2>=0.25):
            #if True:
                done=1
                row_contents = [self.tot,self.pos_1, self.state["position0"].x_val,self.state["position0"].y_val, self.curr_re1, self.pos_2,self.state["position1"].x_val,self.state["position1"].y_val, self.curr_re2,  ]
                self.tot+=1
                append_list_as_row('out.csv', row_contents)
            return reward,done
            
        else:
            print("crit error, reward")
            exit()
    
    
    def _get_obs(self):
        import cv2
        if self.current_drone == "DroneFollower1":
            self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
            self.drone.simSetSegmentationObjectID("DroneFollower1", 10, True)
            self.drone.simSetSegmentationObjectID("DroneFollower2", 20, True)
            self.drone.simSetSegmentationObjectID(self.pos_1[0], 30, True)
            print(self.pos_1[0])
            
        elif self.current_drone == "DroneFollower2":
            self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
            self.drone.simSetSegmentationObjectID("DroneFollower1", 20, True)
            self.drone.simSetSegmentationObjectID("DroneFollower2", 10, True)
            self.drone.simSetSegmentationObjectID(self.pos_2[0], 30, True)
            print(self.pos_2[0])
        else:
            print("obs error")
            exit()
    
            
        responses = self.drone.simGetImages([self.image_request_seg], external=True)
        response_seg = responses[0]
        img_seg = np.fromstring(response_seg.image_data_uint8, dtype=np.uint8)
        img_rgb_seg = img_seg.reshape(response_seg.height, response_seg.width, 3)
        
        try:
            image_seg = Image.fromarray(img_rgb_seg)
            im_final_seg = np.array(image_seg.resize((512, 512)))

        except:
            im_final_seg = np.zeros((512,512,3), np.uint8)
            
            
        mask_=im_final_seg.copy()
        mask_[np.where((mask_ == [199, 26, 29]).all(axis=2))] = [10,10,10]
        mask_[np.where((mask_ == [70, 52, 146]).all(axis=2))] = [20,20,20]
        mask_[np.where((mask_ == [123, 21, 124]).all(axis=2))] = [30,30,30]
        
        mask1=mask_[:,:,0]
        a=mask1!=0
        b=mask1!=10 
        c=mask1!=20 
        d=mask1!=30
        
        t=a|b|c|d
        newm=np.where(t,mask1,0)
        mask=torch.from_numpy(newm) 
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        
        print(obj_ids)
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        
        print(boxes)
        #ttt = np.zeros((512,512,3), np.uint8)
        ttt=cv2.rectangle(im_final_seg, (int(boxes[0][0])-5,int(boxes[0][1])-5), (int(boxes[0][2])+5,int(boxes[0][3])+5), (0, 0, 255),-1)
        ttt=cv2.rectangle(ttt, (int(boxes[1][0])-5,int(boxes[1][1])-5), (int(boxes[1][2])+5,int(boxes[1][3])+5), (0, 255, 0),-1)
        ttt=cv2.rectangle(ttt, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0),-1)

            
        self.drone_state0 = self.drone.getMultirotorState(vehicle_name="DroneFollower1")
        self.drone_state1 = self.drone.getMultirotorState(vehicle_name="DroneFollower2")
        
        self.state["prev_position0"] = self.state["position0"]
        self.state["position0"] = self.drone_state0.kinematics_estimated.position
        self.state["velocity0"] = self.drone_state0.kinematics_estimated.linear_velocity
        
        self.state["prev_position1"] = self.state["position1"]
        self.state["position1"] = self.drone_state1.kinematics_estimated.position
        self.state["velocity1"] = self.drone_state1.kinematics_estimated.linear_velocity
        

        collision0 = self.drone.simGetCollisionInfo(vehicle_name="DroneFollower1").has_collided
        collision1 = self.drone.simGetCollisionInfo(vehicle_name="DroneFollower2").has_collided
        
        
        self.state["collision0"] = collision0
        self.state["collision1"] = collision1

        
        print(self.current_drone)
        cv2.imwrite(self.current_drone+".png",ttt)
        return im_final_seg
    
    def _do_action(self, action):
        #print(action)
        drone_name=self.current_drone
        offset = self.interpret_action(action)
        pos=self.drone.getMultirotorState(vehicle_name=drone_name).kinematics_estimated.position
        self.drone.moveToPositionAsync(pos.x_val + offset[0],pos.y_val + offset[1],pos.z_val, 1, vehicle_name=drone_name).join()

        
    def interpret_action(self, action):
        
        #move forward
        if action == 0:
            quad_offset = (self.step_length, 0, 0)

        elif action == 1:
            quad_offset = (0, self.step_length, 0)

        elif action == 2:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 3:
            quad_offset = (0, -self.step_length, 0)
        else:
            quad_offset = (0, 0, 0)


        return quad_offset


