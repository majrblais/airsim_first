import airsim
import os
from PIL import Image
import numpy as np
from gym import spaces
import math
# = airsim.MultirotorClient()
#client.confirmConnection()
import time
##print(client.getMultirotorState())
from airgym.envs.airsim_env import AirSimEnv
import gym
import random
from datetime import datetime
random.seed(datetime.now().timestamp())
import torch
val=0
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

import cv2
import numpy as np
import numpy.ma as ma
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
ObsType = TypeVar("ObsType")

class Agent():  # properties of agent entities
    def __init__(self):
        # state
        self.obs=None
        self.pos = np.zeros(3)
        self.prev_pos = np.zeros(3)
        self.col=False
        # physical action (1,2,3,4)
        self.action = 0  
        #name
        self.name='Empty'
        #id
        self.id=0
        #firespot
        self.spot=FireSpot()

class FireSpot():  # properties of agent entities
    def __init__(self):
        # state
        self.pos = np.zeros(3)
        #name
        self.name='Empty'
        #id
        self.id=0
        

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)

class AirSimcustomEnv_base(gym.Env):
    def __init__(self,ip_address="127.0.0.1", step_length=1, image_shape=(84, 84, 1),):
        super().__init__()
        ####print("test")
        #metadata
        self.metadata = {"render_modes": ["human", "rgb_array"],"is_parallelizable": True,"render_fps": 10,"name":None}
        
        #init parameters
        self.done_flag=False
        self.val=val
        self.step_length = 0.5
        self.image_shape = (512, 512, 3)
        self.render_mode = 'none'
        self.metadata["name"] = "testenv"
        self.image_request_seg = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Segmentation,False, False)
        
        self.observation_space = spaces.Box(0, 255, shape=image_shape, dtype=np.uint8)
        self.action_space = spaces.Discrete(4)
        
        self.current_landmark=None
        #####################################################
        #START AIRSIM
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        
        #####################################################
        
        

    
        #####################################################
        #create rando agents
        num_agents=2
        off=0
        self.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(self.agents):
            agent.obs=None
            agent.name = f"DroneFollower{i}"
            agent.col = False
            agent.action = None
            agent.pos = np.zeros(3)
            agent.prev_pos = np.zeros(3)
            agent.pres_rew = 0.0
            agent.id=10+off
            off+=1
        #####################################################
        
        self.curr_idx=0
        #####################################################

        #####################################################
        #create landmarks
        num_landmark=16
        self.landmarks = [FireSpot() for i in range(num_landmark)]
        
        for i, landmark in enumerate(self.landmarks):
            landmark.name = f"fire{i}"
            landmark.pos = [np.array([self.drone.simGetObjectPose(landmark.name).position.x_val,self.drone.simGetObjectPose(landmark.name).position.y_val,-10.0])]
        #####################################################

        
        

        self._setup_flight()
    
    def _setup_flight(self):
        ####print("setting-up")
        self.drone.reset()
        #Controllable setup
        offset=0
        self.done_flag=False

        tmp_choice=random.choice(self.landmarks)
        self.current_landmark=tmp_choice
        
        #RANDO
        self.drone.enableApiControl(True, "DroneFollower0")
        self.drone.armDisarm(True, "DroneFollower0")
        self.drone.takeoffAsync(vehicle_name="DroneFollower0").join()
        self.drone.moveToPositionAsync(5, 0, -11, 1, vehicle_name="DroneFollower0").join()
        g=self.drone.simGetCollisionInfo(vehicle_name="DroneFollower0").has_collided
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower0").join()
        self.drone.hoverAsync(vehicle_name="DroneFollower0").join()


        self.drone.enableApiControl(True, "DroneFollower1")
        self.drone.armDisarm(True, "DroneFollower1")
        self.drone.takeoffAsync(vehicle_name="DroneFollower1").join()
        self.drone.moveToPositionAsync(-5, 0, -12, 1, vehicle_name="DroneFollower1").join()
        g=self.drone.simGetCollisionInfo(vehicle_name="DroneFollower1").has_collided
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower1").join()
        self.drone.hoverAsync(vehicle_name="DroneFollower1").join()
    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def __del__(self):
        self.drone.reset()
        
    
    def step(self, action):
    
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, {}

    def _do_action(self,action):
        ####print(action)
        #offset = self.interpret_action(action)
        angt=self.agents[self.curr_idx]
        
        pos=self.drone.getMultirotorState(vehicle_name=angt.name).kinematics_estimated.position
        quad_offset = (0,0, 0)
        
        if action == 0:
        #if action == 0 and pos.x_val<43:
            quad_offset = (self.step_length, 0, 0)
            ####print("act0")
        elif action == 1:
        #elif action == 1 and pos.y_val<45:
            quad_offset = (0, self.step_length, 0)
            ####print("act1")
        elif action == 2 :
        #elif action == 2 and pos.x_val> -52:
            quad_offset = (-self.step_length, 0, 0)
            ####print("act2")
        elif action == 3 :
        #elif action == 3 and pos.y_val>-45:
            quad_offset = (0, -self.step_length, 0)
            ####print("act3")
            
        pos=self.drone.getMultirotorState(vehicle_name=angt.name).kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(pos.x_val + quad_offset[0],pos.y_val + quad_offset[1],0, 1, vehicle_name=angt.name).join()  
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=angt.name).join()
        self.drone.hoverAsync(vehicle_name=angt.name).join()
        

    def _compute_reward(self):
        reward=0
        # Agents are rewarded based on log'd difference between the current squared distance and the previous.
        angt=self.agents[self.curr_idx]
        
        end_pts = angt.spot.pos

        quad_pt = np.array(list((angt.pos.x_val,angt.pos.y_val,angt.pos.z_val,)))
        dist = np.linalg.norm(end_pts[0][0:2]-quad_pt[0:2])
        done=0
        if angt.col:
            reward = -100
            done=1
            
        #elif self.done_flag==True or self.trainer.pos.x_val>44 or self.trainer.pos.x_val< -52 or self.trainer.pos.y_val >45 or self.trainer.pos.y_val <-45:
        elif self.done_flag==True:
            print("collision")
            reward = -100
            done=1
            
        elif dist<= self.trainer.pres_rew:
            reward=1
            done=0
        
        elif dist>= self.trainer.pres_rew:
            reward=-1
            done=0
            
        else:
            reward=0
            #print("error")
            done=0
        
        if dist <2:
            ###print("Destination found")
            reward=100
            done=1

                
        self.trainer.pres_rew=dist
        return reward,done
            

    def _get_obs(self):
            
        ####print(agent_)
        #Set IDs
        self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
        self.drone.simSetSegmentationObjectID(self.trainer.name, self.trainer.id, True)
        self.drone.simSetSegmentationObjectID(self.trainer.spot.name, self.trainer.spot.id, True)
                
        for agent in self.agents:
            self.drone.simSetSegmentationObjectID(agent.name, agent.id, True)

        
        #Get image
        responses = self.drone.simGetImages([self.image_request_seg], external=True)
        response_seg = responses[0]
        img_seg = np.fromstring(response_seg.image_data_uint8, dtype=np.uint8)
        img_rgb_seg = img_seg.reshape(response_seg.height, response_seg.width, 3)
        

        image_seg = Image.fromarray(img_rgb_seg)
        im_final_seg = np.array(image_seg.resize((512, 512)))


            
        
        #create seg map
        mask_=im_final_seg.copy()
        #blue
        mask_[np.where((mask_ == [199, 26, 29]).all(axis=2))] = [10,10,10]
        mask_[np.where((mask_ == [239, 16, 102]).all(axis=2))] = [11,11,11]
        mask_[np.where((mask_ == [146, 107, 242]).all(axis=2))] = [12,12,12]
        #pink
        mask_[np.where((mask_ == [70, 52, 146]).all(axis=2))] = [20,20,20]


        
        mask1=mask_[:,:,0]
        a=mask1!=0
        b=mask1!=10 
        c=mask1!=11
        d=mask1!=12
        e=mask1!=20
        
        t=a|b|c|d|e
        newm=np.where(t,mask1,0)
        mask=torch.from_numpy(newm) 
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        
        ####print(obj_ids)
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        
        ####print(boxes)
        #ttt = np.zeros((512,512,3), np.uint8)
        ttt = np.zeros((512,512,3), np.uint8)
        
        try:
            ttt=cv2.rectangle(ttt, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (0, 0, 255),-1)
            ttt=cv2.rectangle(ttt, (int(boxes[1][0]),int(boxes[1][1])), (int(boxes[1][2]),int(boxes[1][3])), (255, 0, 0),-1)
            ttt=cv2.rectangle(ttt, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0),-1)
            ttt=cv2.rectangle(ttt, (int(boxes[3][0]),int(boxes[3][1])), (int(boxes[3][2]),int(boxes[3][3])), (0, 255, 0),-1)
        except:
        
            ttt=cv2.rectangle(ttt, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (255, 0, 0),-1)
            ttt=cv2.rectangle(ttt, (int(boxes[1][0]),int(boxes[1][1])), (int(boxes[1][2]),int(boxes[1][3])), (255, 0, 0),-1)
            ttt=cv2.rectangle(ttt, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (0, 255, 0),-1)
            
            self.done_flag=True

        
        self.drone_state = self.drone.getMultirotorState(vehicle_name=self.trainer.name)
        self.trainer.prev_pos=self.trainer.pos
        self.trainer.pos=self.drone_state.kinematics_estimated.position
        self.trainer.col = self.drone.simGetCollisionInfo(vehicle_name=self.trainer.name).has_collided
        
        

        #cv2.imwrite('gggg.png',ttt)
        
        ttt=ttt.astype(np.float32)

        
        return ttt/255.00