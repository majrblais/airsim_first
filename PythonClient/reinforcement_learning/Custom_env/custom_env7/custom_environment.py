import os
import cv2
import torch
import airsim
import random
import numpy as np
from PIL import Image
from copy import copy
from gym import spaces

from pettingzoo import AECEnv
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes

from pettingzoo.utils.agent_selector import agent_selector
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar

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
        self.pres_rew=0
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


class env(AECEnv):
    def __init__(self,ip_address="127.0.0.1", val=False):
        super().__init__()
        ###print("test")
        #metadata
        self.metadata = {"render_modes": ["human", "rgb_array"],"is_parallelizable": True,"render_fps": 10,"name":None}
        
        #init parameters
        self.val=val
        self.step_length = 1
        self.image_shape = (512, 512, 3)
        self.render_mode = 'none'
        self.metadata["name"] = "testenv"
        self.image_request_seg = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Segmentation,False, False)
        self.max_cycles = 20
        self.local_ratio = 0.25
        self.possible_agents=["DroneFollower0", "DroneFollower1"]
        #####################################################
        
        #####################################################
        #START AIRSIM
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        #####################################################
        
        #AGENTS
        #create agents
        num_agents=2
        self.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(self.agents):
            agent.obs=None
            agent.name = f"DroneFollower{i}"
            agent.col = False
            agent.action = None
            agent.pos = np.zeros(3)
            agent.prev_pos = np.zeros(3)
            agent.pres_rew = 0.0
        
        self.agents[0].id=10
        self.agents[1].id=20
        
        self._agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.agents)}

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self._agent_selector.reset()
        #####################################################
        
        #####################################################
        #FIRE SPOTS
        #fire positions
        if val:
            num_landmark=2
            self.landmarks = [FireSpot() for i in range(num_landmark)]
            
            for i, landmark in enumerate(self.landmarks):
                landmark.name = f"fireval{i}"
                landmark.pos = [np.array([self.drone.simGetObjectPose(landmark.name).position.x_val,self.drone.simGetObjectPose(landmark.name).position.y_val,-10.0])]
        else:
            num_landmark=16
            #self.detectors = [[np.array([-42.0,-12.0,-10.0])]]

            self.landmarks = [FireSpot() for i in range(num_landmark)]
            
            for i, landmark in enumerate(self.landmarks):
                landmark.name = f"fire{i}"
                landmark.pos = [np.array([self.drone.simGetObjectPose(landmark.name).position.x_val,self.drone.simGetObjectPose(landmark.name).position.y_val,-10.0])]
        
        ###print("test")
        self.possible_landmarks = self.landmarks
        self.current_landmark=None
        
        
       #for i,landmark in enumerate(self.landmarks):
        #    ##print(landmark.name)
       #     ##print(landmark.pos)
        #####################################################
        
        #####################################################
        #create spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        state_dim = 0
        
        self.possible_agents=["DroneFollower0", "DroneFollower1"]
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=255, shape=(512, 512, 3), dtype=np.uint8),
                    "action_mask": spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8),
                }
            )
            for name in self.possible_agents
        }
        
       
        
        
        self.action_spaces = {name: spaces.Discrete(4) for name in self.possible_agents}
        
        ###print(self.observation_spaces)
        ###print(self.action_spaces)
        
        self.steps = 0
        self.current_actions = [None] * self.num_agents
        #####################################################  
        self._setup_flight()
        
    def _setup_flight(self):
        ###print("setting-up")
        self.drone.reset()
        #Controllable setup
        offset=0


        tmp_choice=random.choice(self.possible_landmarks)
        self.current_landmark=tmp_choice
        
        for agent in self.agents:
            self.drone.enableApiControl(True, agent.name)
            self.drone.armDisarm(True, agent.name)
            self.drone.takeoffAsync(vehicle_name=agent.name).join()
           
            
            if agent.name == "DroneFollower0":
                self.drone.moveToPositionAsync(offset, 0, -10, 1, vehicle_name=agent.name).join()
                agent.spot=self.current_landmark
                agent.spot.id=30
            elif agent.name == "DroneFollower1":
                self.drone.moveToPositionAsync(offset, 0, -11, 1, vehicle_name=agent.name).join()
                tmp_choice=random.choice(self.possible_landmarks)
                while tmp_choice==self.current_landmark:
                    tmp_choice=random.choice(self.possible_landmarks)
                self.current_landmark=tmp_choice
                agent.spot=self.current_landmark
                agent.spot.id=40
            else:
                exit()            
            
            #this is necessary for the second drone for some reason (well the reason is that the initial collision is only cleared after calling it, first drone doesnt have this issue, only second)
            g=self.drone.simGetCollisionInfo(vehicle_name=agent.name).has_collided
            self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=agent.name).join()
            self.drone.hoverAsync(vehicle_name=agent.name).join()
            offset+=3
            
            self.agents[self._index_map[agent.name]].col=False
            self.agents[self._index_map[agent.name]].pres_rew=0.0
            ###print(self.drone.simGetCollisionInfo(vehicle_name=agent.name).has_collided)
        
    def reset(self, seed=None, return_info=False, options=None):
        self._setup_flight()

        self.agents = self._agents[:]
        self.rewards = {name: 0.0 for name in self.possible_agents}
        self._cumulative_rewards = {name: 0.0 for name in self.possible_agents}
        self.terminations = {name: False for name in self.possible_agents}
        self.truncations = {name: False for name in self.possible_agents}
        self.infos = {name: {} for name in self.possible_agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents
        
        
        self.drone_state0 = self.drone.getMultirotorState(vehicle_name='DroneFollower0')
        self.agents[self._index_map["DroneFollower0"]].pos=self.drone_state0.kinematics_estimated.position
        self.agents[self._index_map["DroneFollower0"]].col = self.drone.simGetCollisionInfo(vehicle_name='DroneFollower0').has_collided
        
        self.drone_state1 = self.drone.getMultirotorState(vehicle_name='DroneFollower1')
        self.agents[self._index_map["DroneFollower1"]].pos=self.drone_state1.kinematics_estimated.position
        self.agents[self._index_map["DroneFollower1"]].col = self.drone.simGetCollisionInfo(vehicle_name='DroneFollower1').has_collided
        

    
    def observation(self, agent_):
    
        ###print(agent_)
        #Set IDs
        self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
        for agent in self.agents:
            self.drone.simSetSegmentationObjectID(agent.name, agent.id, True)
            if agent.name==agent_.name:
                self.drone.simSetSegmentationObjectID(agent.spot.name, agent.spot.id, True)
        
        #Get image
        responses = self.drone.simGetImages([self.image_request_seg], external=True)
        response_seg = responses[0]
        img_seg = np.fromstring(response_seg.image_data_uint8, dtype=np.uint8)
        img_rgb_seg = img_seg.reshape(response_seg.height, response_seg.width, 3)
        

        image_seg = Image.fromarray(img_rgb_seg)
        im_final_seg = np.array(image_seg.resize((512, 512)))


            
        
        #create seg map
        mask_=im_final_seg.copy()
        mask_[np.where((mask_ == [199, 26, 29]).all(axis=2))] = [10,10,10]
        mask_[np.where((mask_ == [70, 52, 146]).all(axis=2))] = [20,20,20]
        if agent_.name=="DroneFollower0":
            mask_[np.where((mask_ == [123, 21, 124]).all(axis=2))] = [30,30,30]
        elif agent_.name=="DroneFollower1":
            mask_[np.where((mask_ == [214, 254, 86]).all(axis=2))] = [30,30,30]
        else:
            ##print("crit")
            exit()
        
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
        
        ###print(obj_ids)
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        
        ###print(boxes)
        #ttt = np.zeros((512,512,3), np.uint8)
        ttt = np.zeros((512,512,3), np.uint8)
        ttt=cv2.rectangle(ttt, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (0, 0, 255),-1)
        ttt=cv2.rectangle(ttt, (int(boxes[1][0]),int(boxes[1][1])), (int(boxes[1][2]),int(boxes[1][3])), (0, 255, 0),-1)
        #ttt=cv2.rectangle(ttt, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0),-1)

        if agent_.name=="DroneFollower0":
            ttt=cv2.rectangle(ttt, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (100, 100, 255),-1)
        elif agent_.name=="DroneFollower1":
            ttt=cv2.rectangle(ttt, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (100, 255, 100),-1)
        else:
            ##print("crit2")
            exit()
        
        self.drone_state = self.drone.getMultirotorState(vehicle_name=agent_.name)
        

        self.agents[self._index_map[agent_.name]].prev_pos=agent_.pos
        self.agents[self._index_map[agent_.name]].pos=self.drone_state.kinematics_estimated.position

        self.agents[self._index_map[agent_.name]].col = self.drone.simGetCollisionInfo(vehicle_name=agent_.name).has_collided
        
        #self.agents[self._index_map[agent_.name]].obs=ttt
        
        cv2.imwrite(str(agent_.name)+'.png',ttt)
        
        return ttt/255.00
    
    
    def masking(self, agent):
        ###print(agent)
        action_mask=np.ones(4)
        pos=self.drone.getMultirotorState(vehicle_name=agent.name).kinematics_estimated.position
        
        if pos.x_val>=43:
            action_mask[0]=0
        if pos.y_val>=45:
            action_mask[1]=0
        if pos.x_val<= -52:
            action_mask[2]=0
        if pos.y_val<=-45:
            action_mask[3]=0
        
        return action_mask
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        observation=self.observation(self.agents[self._index_map[agent]]).astype(np.float32)
        action_mask=self.masking(self.agents[self._index_map[agent]]).astype(np.float32)
        return {"observation": observation, "action_mask": action_mask}

    def state(self):
        states = tuple(self.observation(self.agents[self._index_map[name]]).astype(np.float32) for name in self.possible_agents)
        return np.concatenate(states, axis=None)

    

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.agents):
            action = self.current_actions[i]
            self._do_action(agent, action)
            
        #self.step()
        
        for agent in self.agents:
            agent_reward = float(self.reward(agent))
            reward = agent_reward
            self.rewards[agent.name] = reward
            ###print("step")
            ###print(self.rewards)
            ###print("step")
          
    def step(self, action):
    
        ###print(action)
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return
        
        #get action for one agent
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()
        
        agnt=self.agents[current_idx]
        self.current_actions[current_idx] = action
        self._do_action(agnt, action)
        
        
        agent_reward = float(self.reward(agnt))
        reward = agent_reward
        self.rewards[agnt.name] = reward
        

        self.steps += 1
        if self.steps >= self.max_cycles:
            for a in self.agents:
                self.truncations[a] = True
                    
                    


        observations = {name: None for name in self.possible_agents}
        
        #rewards = {name: self.reward(self.agents[self._index_map[name]]) for name in self.possible_agents}
        infos = {name: {} for name in self.possible_agents}
        
        return observations, self.rewards, self.terminations, self.truncations, infos

    def _do_action(self,agent,action):
        ###print(action)
        #offset = self.interpret_action(action)
        pos=self.drone.getMultirotorState(vehicle_name=agent.name).kinematics_estimated.position

        quad_offset = (0,0, 0)

        if action == 0 and pos.x_val<=43:
            quad_offset = (self.step_length, 0, 0)
            ###print("act0")
        elif action == 1 and pos.y_val<=45:
            quad_offset = (0, self.step_length, 0)
            ###print("act1")
        elif action == 2 and pos.x_val>= -52:
            quad_offset = (-self.step_length, 0, 0)
            ###print("act2")
        elif action == 3 and pos.y_val>=-45:
            quad_offset = (0, -self.step_length, 0)
            ###print("act3")
            
        pos=self.drone.getMultirotorState(vehicle_name=agent.name).kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(pos.x_val + quad_offset[0],pos.y_val + quad_offset[1],pos.z_val, 1, vehicle_name=agent.name).join()  
        self.drone.hoverAsync(vehicle_name=agent.name).join()
        
        
    def reward(self, agent):
        reward=0
        # Agents are rewarded based on log'd difference between the current squared distance and the previous.
        end_pts = agent.spot.pos
        quad_pt = np.array(list((agent.pos.x_val,agent.pos.y_val,agent.pos.z_val,)))
        dist = np.linalg.norm(end_pts[0][0:2]-quad_pt[0:2])
        #dist = np.sqrt(dist)
        #dist= -dist
        
        
        print(agent.name)
        print(agent.pos)
        print(agent.spot.pos)
        
        print(dist)
        print(agent.pres_rew)
        
        if agent.col:
            reward = -3
            ##print("collisionFollower")
            self.terminations = {agent: True for agent in self.possible_agents}
            
        elif agent.pos.x_val>=43 or agent.pos.x_val<= -52 or agent.pos.y_val >=45 or agent.pos.y_val <=-45:
            reward = -3
            ##print("OoBFollower")
            self.terminations = {agent: True for agent in self.possible_agents}
        elif dist<= agent.pres_rew:
            reward=2
        
        elif dist>= agent.pres_rew:
            reward=-1
        else:
            reward=0
            print("error")
        
        if dist <3:
            ##print("Destination found")
            reward=5
            self.terminations = {agent: True for agent in self.possible_agents}

                
        agent.pres_rew=dist
        return reward
            
    def make_world(self):
        pass
        
    def render(self):
        pass
