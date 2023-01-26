from pettingzoo.utils.env import ParallelEnv
import functools
import random
from copy import copy
from gym import spaces
import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete, Box
import airsim
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ParallelEnv
import torch
from PIL import Image
from torchvision.ops import masks_to_boxes
from pettingzoo import AECEnv
from torchvision.utils import draw_bounding_boxes
class CustomEnvironment(AECEnv):
    metadata = {"render_modes": ["rgb_array"],"name": "CustomEnvironment","is_parallelizable": True}

    def __init__(self,ip_address="127.0.0.1", step_length=1, image_shape=(84, 84, 1),):
        self.step_length = step_length
        self.image_shape = image_shape
        #self.observation_space = Box(0, 255, shape=image_shape, dtype=np.uint8)
        #self.observation_space = MultiDiscrete([512 * 512 * 3] * 256)
        #self.action_space = Discrete(4)
        self.render_mode = None
        self._none = 4
        self.states = {"position0": np.zeros(3),"position1": np.zeros(3),"collision0": False, "collision1": False ,"prev_position0": np.zeros(3), "prev_position1": np.zeros(3)}
        self.detectors = {"detector1": [np.array([-45.0,20.0,-50.0])]}
        self.tot=0
        self.pos1=None
        
        self.possible_agents = ["DroneFollower1", "DroneFollower2"]
        self.observation_spaces = dict(zip(self.possible_agents,[Box(low=0,high=255,shape=self.image_shape,dtype=np.uint8,)]* len(self.possible_agents),))
        self.action_spaces = dict(zip(self.possible_agents, [Discrete(4)] *len(self.possible_agents)))
        
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.image_request_seg = airsim.ImageRequest("FixedCamera1", airsim.ImageType.Segmentation,False, False)
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        self.state = {agent: self._none for agent in self.agents}
        self.observations = {agent: self._none for agent in self.agents}
        self.reset()

        
        
    def reset(self, seed=None, return_info=False, options=None):
        self._setup_flight()
        observations = {a:(self._get_obs(drone_name=a)) for a in self.possible_agents}

        return observations

    def __del__(self):
        pass
    
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
        
        
    def _setup_flight(self):
        print("setting-up")
        self.drone.reset()
        
        self.pos_1=random.choice(list(self.detectors.items()))

        #Controllable setup
        self.drone.enableApiControl(True, "DroneFollower1")
        self.drone.enableApiControl(True, "DroneFollower2")
        
        self.drone.armDisarm(True, "DroneFollower1")
        self.drone.armDisarm(True, "DroneFollower2")
        
        #takeoff, move to side by side
        self.drone.takeoffAsync(vehicle_name="DroneFollower1").join()
        self.drone.moveToPositionAsync(5, 0, -50, 10, vehicle_name="DroneFollower1").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower1").join()
        self.drone.hoverAsync(vehicle_name="DroneFollower1").join()
        
        self.drone.takeoffAsync(vehicle_name="DroneFollower2").join()
        self.drone.moveToPositionAsync(-5, 0, -51, 10, vehicle_name="DroneFollower2").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower2").join()
        self.drone.hoverAsync(vehicle_name="DroneFollower2").join()


    def _get_obs(self, drone_name):
        import cv2
        if drone_name == "DroneFollower1":
            self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
            self.drone.simSetSegmentationObjectID("DroneFollower1", 10, True)
            self.drone.simSetSegmentationObjectID("DroneFollower2", 20, True)
            self.drone.simSetSegmentationObjectID(self.pos_1[0], 30, True)

        elif drone_name == "DroneFollower2":
            self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
            self.drone.simSetSegmentationObjectID("DroneFollower1", 10, True)
            self.drone.simSetSegmentationObjectID("DroneFollower2", 20, True)
            self.drone.simSetSegmentationObjectID(self.pos_1[0], 30, True)
            
        else:
            print("crit error")
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
            
        
        cv2.imwrite('t.png',im_final_seg)
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
        
        #print(obj_ids)
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        
        #print(boxes)
        #ttt = np.zeros((512,512,3), np.uint8)
        ttt=cv2.rectangle(im_final_seg, (int(boxes[0][0])-5,int(boxes[0][1])-5), (int(boxes[0][2])+5,int(boxes[0][3])+5), (0, 0, 255),-1)
        ttt=cv2.rectangle(ttt, (int(boxes[1][0])-5,int(boxes[1][1])-5), (int(boxes[1][2])+5,int(boxes[1][3])+5), (0, 255, 0),-1)
        ttt=cv2.rectangle(ttt, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0),-1)
        #cv2.imwrite(drone_name+".png",ttt)
            
        self.drone_state0 = self.drone.getMultirotorState(vehicle_name="DroneFollower1")
        self.drone_state1 = self.drone.getMultirotorState(vehicle_name="DroneFollower2")
        
        self.states["prev_position0"] = self.states["position0"]
        self.states["position0"] = self.drone_state0.kinematics_estimated.position
        self.states["velocity0"] = self.drone_state0.kinematics_estimated.linear_velocity
        
        self.states["prev_position1"] = self.states["position1"]
        self.states["position1"] = self.drone_state1.kinematics_estimated.position
        self.states["velocity1"] = self.drone_state1.kinematics_estimated.linear_velocity
        

        collision0 = self.drone.simGetCollisionInfo(vehicle_name="DroneFollower1").has_collided
        collision1 = self.drone.simGetCollisionInfo(vehicle_name="DroneFollower2").has_collided
        
        
        self.states["collision0"] = collision0
        self.states["collision1"] = collision1

        
        #print(drone_name)
        #cv2.imwrite(self.current_drone+".png",ttt)
        return im_final_seg

        
    def step(self, actions):

        
        D1=actions['DroneFollower1']
        D2=actions['DroneFollower2']
        self._do_action(D1,'DroneFollower1')
        self._do_action(D2,'DroneFollower2')
        
        terminations = {a: False for a in self.possible_agents}
        rewards = {a: (self._compute_reward(drone_name=a)) for a in self.possible_agents}
        if rewards["DroneFollower1"]<=-25 or rewards["DroneFollower1"]<=-25:
            terminations = {a: True for a in self.possible_agents}
            
        truncations = {a: False for a in self.possible_agents}
        
        if self.tot > 100:
            rewards = {"DroneFollower1": 0, "DroneFollower2": 0}
            truncations = {"DroneFollower1": True, "DroneFollower2": True}
        self.tot+=1
        
        observations = {a:(self._get_obs(drone_name=a)) for a in self.possible_agents}
        
        infos = {a: {} for a in self.possible_agents}
        return observations, rewards, terminations, truncations, infos

    def _do_action(self, action,current_drone):
        offset = self.interpret_action(action)
        pos=self.drone.getMultirotorState(vehicle_name=current_drone).kinematics_estimated.position
        self.drone.moveToPositionAsync(pos.x_val + offset[0],pos.y_val + offset[1],pos.z_val, 1, vehicle_name=current_drone).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=current_drone).join()
        self.drone.hoverAsync(vehicle_name=current_drone).join()
        
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
    
    def _compute_reward(self,drone_name):
        #desired location & current location
        
        if drone_name == "DroneFollower1":
            end_pts = self.pos_1[1]
            quad_pt0 = np.array(list((self.states["position0"].x_val,self.states["position0"].y_val,self.states["position0"].z_val,)))

            if self.states["collision0"]:
                reward = -100
                print("collisionFollower1")
                
            elif self.states["position0"].x_val>=44 or self.states["position0"].x_val<= -49 or self.states["position0"].y_val >=44 or self.states["position0"].y_val <=-49:
                reward = -100
                print("OoBFollower1")
            else:
                dist = np.linalg.norm(end_pts[0][0:2]-quad_pt0[0:2])
                reward = - np.sqrt(dist)+1
            return reward
            
        elif drone_name == "DroneFollower2":
            end_pts = self.pos_1[1]
            quad_pt0 = np.array(list((self.states["position1"].x_val,self.states["position1"].y_val,self.states["position1"].z_val,)))
            #should theo not happen since it flys higher
            if self.states["collision0"]:
                reward = -100                
            elif self.states["position0"].x_val>=44 or self.states["position1"].x_val<= -49 or self.states["position1"].y_val >=44 or self.states["position1"].y_val <=-49:
                reward = -100
            else:
                dist = np.linalg.norm(end_pts[0][0:2]-quad_pt0[0:2])
                reward = - np.sqrt(dist)+1

            return reward
            
        else:
            print("crit error, reward")
            exit()
            
    def observe(self, agent):
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])
   
    
    def render(self):
        pass
