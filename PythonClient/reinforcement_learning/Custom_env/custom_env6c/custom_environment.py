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
        self.pos = np.zeros(2)
        self.prev_pos = np.zeros(2)
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
        self.pos = np.zeros(2)
        #name
        self.name='Empty'
        #id
        self.id=0


class env(AECEnv):


    def random_of_ranges(s):
        arr1 = np.random.randint(32,192)
        arr2 = np.random.randint(320,480)
        out = np.stack((arr1,arr2))
        out = np.random.choice(out)
        return out


    def __init__(self,val=False):
        super().__init__()
        ###print("test")
        #metadata
        self.val=val
        self.val_idx=0
        self.metadata = {"render_modes": ["human", "rgb_array"],"is_parallelizable": True,"render_fps": 10,"name":None}
        self.possible_agents=["Drone0", "Drone1"]
        self.BOARD_SIZE = (512, 512)
        self.BOARD_SIZE = (512, 512)
        self.window_surface = None
        #init parameters
        self.step_length = 1
        self.image_shape = (512, 512, 3)
        self.map=np.zeros((512,512,3), np.uint8)
        self.render_mode = 'human'
        self.metadata["name"] = "testenv"
        #####################################################
        #AGENTS
        #create agents
        num_agents=2
        self.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(self.agents):
            agent.obs=None
            agent.name = f"Drone{i}"
            agent.col = False
            agent.action = None
            agent.pos = np.zeros(2)
            agent.prev_pos = np.zeros(2)
            agent.pres_rew = 0.0
        
        self.agents[0].id=10
        self.agents[1].id=20
        
        self._agents = self.agents[:]
        self._index_map = {agent.name: idx for idx, agent in enumerate(self.agents)}

        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection = self._agent_selector.reset()
        #####################################################
        self.max_cycles=1000
        #####################################################
        #self.detectors = [[np.array([-42.0,-12.0,-10.0])]]
        self.current_landmark=FireSpot()
        self.done_flag=False
        
       #for i,landmark in enumerate(self.landmarks):
        #    ##print(landmark.name)
       #     ##print(landmark.pos)
        #####################################################
        
        #####################################################
        #create spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        state_dim = 0
        
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
        #print("setting-up")
        #Controllable setup
        offset=0

        if self.val:
            self.current_landmark=FireSpot()
            if self.val_idx==0:
                self.current_landmark.pos[0]=32
                self.current_landmark.pos[1]=32
            if self.val_idx==1:
                self.current_landmark.pos[0]=32
                self.current_landmark.pos[1]=480

            if self.val_idx==2:
                self.current_landmark.pos[0]=480
                self.current_landmark.pos[1]=32
                
            if self.val_idx==3:
                self.current_landmark.pos[0]=480
                self.current_landmark.pos[1]=480
                
            self.val_idx+=1
            if self.val_idx>3:
                self.val_idx=0
            
        else:
            self.current_landmark=FireSpot()
            self.current_landmark.pos[0]=self.random_of_ranges()
            self.current_landmark.pos[1]=self.random_of_ranges()
        

        #cv2.imwrite('g.png',ttt)

        for agent in self.agents:
            agent.spot=self.current_landmark
            agent.spot.id=30
            agent.pos[0]=240+offset
            agent.pos[1]=256
            offset+=32
            
            self.agents[self._index_map[agent.name]].col=False

        
    def reset(self, seed=None, return_info=False, options=None):
        try:
            cv2.destroyAllWindows()
        except:
            pass
        self._setup_flight()
        self.done_flag=False
        self.agents = self._agents[:]
        self.rewards = {name: 0.0 for name in self.possible_agents}
        self._cumulative_rewards = {name: 0.0 for name in self.possible_agents}
        self.terminations = {name: False for name in self.possible_agents}
        self.truncations = {name: False for name in self.possible_agents}
        self.infos = {name: {} for name in self.possible_agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions = [None] * self.num_agents
        observations = {name: self.observe(name) for name in self.possible_agents}
        infos = {name: {} for name in self.possible_agents}
        
        return observations, infos


    def observation(self, agent_):
    
        ttt = np.zeros((512,512,3), np.uint8)
        cv2.rectangle(ttt, (int(agent_.spot.pos[0])-16,int(agent_.spot.pos[1])-16), (int(agent_.spot.pos[0])+16,int(agent_.spot.pos[1])+16), (255, 0, 0),-1)
        for agent in self.agents:
            if agent.name == agent_.name:
                cv2.rectangle(ttt, (int(agent.pos[0])-3,int(agent.pos[1])-3), (int(agent.pos[0])+3,int(agent.pos[1])+3), (0, 255, 0),-1)
            else :
                cv2.rectangle(ttt, (int(agent.pos[0])-3,int(agent.pos[1])-3), (int(agent.pos[0])+3,int(agent.pos[1])+3), (0, 0, 255),-1)
                
        self.map=ttt
        return ttt/255.00
    
 
    def masking(self, agent):
        #print(agent)
        action_mask=np.ones(4)
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

    

    def step(self, action):
    
        ###print(action)
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            return self._was_dead_step(action)
        
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

        observations = {name: self.observe(name) for name in self.possible_agents}
        
        #rewards = {name: self.reward(self.agents[self._index_map[name]]) for name in self.possible_agents}
        infos = {name: {} for name in self.possible_agents}
        #if self.val:
        if True:
            self.render()
        return observations, self.rewards, self.terminations, self.truncations, infos

    def _do_action(self,agent,action):
        if action == 0:
            agent.pos[0]+=2.0
        elif action == 1:
            agent.pos[1]+=2.0
        elif action == 2:
            agent.pos[0]-=2.0
        elif action == 3:
            agent.pos[1]-=2.0
            


        
        
    def reward(self, agent):
        reward=0
        #print(self.agents[0].pos)
        #print(self.agents[1].pos)
        #print(self.agents[0].spot.pos)
        distbetween=np.linalg.norm(self.agents[0].pos-self.agents[1].pos)
        dist=np.linalg.norm(agent.pos-agent.spot.pos)
        if distbetween < 6:
            self.done_flag=True
            reward = -100
            #print("collision")
            self.terminations = {agent: True for agent in self.possible_agents}
        elif distbetween > 64:
            reward = -100
            self.terminations = {agent: True for agent in self.possible_agents}
            #print("bnetween")
        elif agent.pos[0] >= 511 or agent.pos[0] <= 1 or agent.pos[1] >= 511 or agent.pos[1] <= 1 :
            reward = -100
            self.terminations = {agent: True for agent in self.possible_agents}  
            #print("oob")
        else:
            reward=-np.sqrt(dist)-(np.sqrt(distbetween)/5)
            
        if dist <5 and distbetween <16:
            #print("Destination found")
            reward=100
            self.terminations = {agent: True for agent in self.possible_agents}

        #print(reward)
        return reward
            
    def make_world(self):
        pass
        
    def render(self):
        import pylab as pl
        cv2.imshow('f',self.map)
        cv2.waitKey(1)
