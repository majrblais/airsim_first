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
        self.state = {"position1": np.zeros(3),"position2": np.zeros(3), "collision1": False, "collision2": False , "prev_position1": np.zeros(3), "prev_position2": np.zeros(3), "track_pos" : 0}
        
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.track = []
        for i in range(1,201):
            self.track.append([0,-i])

        for i in range(1,101):
            self.track.append([-i,-200])

        for i in range(1,301):
            self.track.append([-100,-200-i])

        for i in range(1,126):
            self.track.append([-100-i,-500])

        for i in range(1,401):
            self.track.append([-225,-500-i])

        #4actionsx2drones
        self.action_space = spaces.Discrete(8)
        
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
        self.drone.enableApiControl(True, "Drone1")
        self.drone.enableApiControl(True, "Drone2")
        
        self.drone.armDisarm(True, "Drone1")
        self.drone.armDisarm(True, "Drone2")

        self.drone.moveToPositionAsync(0, 0, -75, 5, vehicle_name="Drone1").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5, vehicle_name="Drone1").join()
        
        self.drone.moveToPositionAsync(0, 0, -75, 5, vehicle_name="Drone2").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5, vehicle_name="Drone2").join()
        
        self.state["track_pos"]=0
        
        #print(self.drone.getMultirotorState(vehicle_name="Drone1"))
        #print(self.drone.getMultirotorState(vehicle_name="Drone2"))
        
        
    def step(self, action1, action2):
        
        #print(self.track[self.state["track_pos"]])
        
        #print("current state")
        #print(self.drone.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position.x_val)
        #print(self.drone.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position.y_val)
        #print(self.drone.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position.x_val)
        #print(self.drone.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position.y_val)
        
        #print(action1,action2)
        
        #print("next state")
        self._do_action(action1, action2)
        
        #print(self.drone.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position.x_val)
        #print(self.drone.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position.y_val)
        #print(self.drone.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position.x_val)
        #print(self.drone.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position.y_val)
        
        obs = self._get_obs()
        reward, done = self._compute_reward()
        self.state["track_pos"]+=1

        return obs, reward, done, self.state
        
    def _compute_reward(self):
        #desired location & current location
        pts = [np.array([self.track[self.state["track_pos"]][0],self.track[self.state["track_pos"]][1], -75.0])]
        
        
        quad_pt1 = np.array(list((self.state["position1"].x_val,self.state["position1"].y_val,self.state["position1"].z_val,)))
        quad_pt2 = np.array(list((self.state["position2"].x_val,self.state["position2"].y_val,self.state["position2"].z_val,)))
        
        #If collision or out of bounds
        
        print("distance between drones:")
        print(np.linalg.norm(quad_pt1[0:2]-quad_pt2[0:2]))
        
        print("distance between center:")
        print(np.linalg.norm(pts[0][0:2]-quad_pt1[0:2]))
        print(np.linalg.norm(pts[0][0:2]-quad_pt2[0:2]))
        
        if self.state["collision1"] or self.state["collision2"]:
            reward = -100
            print("collision")
        elif np.linalg.norm(quad_pt1[0:2]-quad_pt2[0:2]) >= 10:
            reward = -100
            print("distance")
            
        elif (np.linalg.norm(pts[0][0:2]-quad_pt1[0:2]) >= 10) or (np.linalg.norm(pts[0][0:2]-quad_pt2[0:2])>=10):
            reward = -100
            print("circle")
        else:
            
            dist1 = np.linalg.norm(pts[0][0:2]-quad_pt1[0:2])
            dist2 = np.linalg.norm(pts[0][0:2]-quad_pt2[0:2])
            reward = (-dist1 + -dist2) 
            print("reward:")
            print(reward)
            
        #print(reward)            
        done = 0
        if reward <=-15:
            done = 1
            
        return reward, done    
    
    def _get_obs(self):
        global val
        
        self.drone_state1 = self.drone.getMultirotorState(vehicle_name="Drone1")
        self.drone_state2 = self.drone.getMultirotorState(vehicle_name="Drone2")
        
        self.state["prev_position1"] = self.state["position1"]
        self.state["position1"] = self.drone_state1.kinematics_estimated.position
        self.state["velocity1"] = self.drone_state1.kinematics_estimated.linear_velocity
        
        self.state["prev_position2"] = self.state["position2"]
        self.state["position2"] = self.drone_state2.kinematics_estimated.position
        self.state["velocity2"] = self.drone_state2.kinematics_estimated.linear_velocity
        
        collision1 = self.drone.simGetCollisionInfo(vehicle_name="Drone1").has_collided
        collision2 = self.drone.simGetCollisionInfo(vehicle_name="Drone2").has_collided
        
        self.state["collision1"] = collision1
        self.state["collision2"] = collision2
        
        
        val+=1

        return self.drone_state1.kinematics_estimated.position, self.drone_state2.kinematics_estimated.position, self.track[self.state["track_pos"]]
    
    def _do_action(self, action1, action2):
        #print(action)
        offset1, offset2 = self.interpret_action(action1, action2)
        #pos=self.drone.getMultirotorState().kinematics_estimated.position
        #self.drone.moveToPositionAsync(pos.x_val + offset[0],pos.y_val + offset[1],pos.z_val, 25).join()
        
        #Move by velocity
        pos1=self.drone.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position
        pos2=self.drone.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position
        
        
        print(self.drone.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position)
        print(self.drone.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position)
        print(offset1)
        print(offset2)
        #self.drone.moveByVelocityAsync(pos1.x_val + offset1[0],pos1.y_val + offset1[1],0, 1, vehicle_name="Drone1").join()
        #self.drone.moveByVelocityAsync(pos2.x_val + offset2[0],pos2.y_val + offset2[1],0, 1, vehicle_name="Drone2").join()
        
        self.drone.moveToPositionAsync(pos1.x_val + offset1[0],pos1.y_val + offset1[1],-75, 1, vehicle_name="Drone1").join()
        self.drone.moveToPositionAsync(pos2.x_val + offset2[0],pos2.y_val + offset2[1],-75, 1, vehicle_name="Drone2").join()
        
        
        print(self.drone.getMultirotorState(vehicle_name="Drone1").kinematics_estimated.position)
        print(self.drone.getMultirotorState(vehicle_name="Drone2").kinematics_estimated.position)
        

        
    def interpret_action(self, action1, action2):
        
        #move forward
        if action1 == 0:
            quad_offset1 = (self.step_length, 0, 0)

        elif action1 == 1:
            quad_offset1 = (0, self.step_length, 0)

        elif action1 == 2:
            quad_offset1 = (-self.step_length, 0, 0)
        elif action1 == 3:
            quad_offset1 = (0, -self.step_length, 0)
        else:
            quad_offset1 = (0, 0, 0)

        if action2 == 0:
            quad_offset2 = (self.step_length, 0, 0)

        elif action2 == 1:
            quad_offset2 = (0, self.step_length, 0)

        elif action2 == 2:
            quad_offset2 = (-self.step_length, 0, 0)
        elif action2 == 3:
            quad_offset2 = (0, -self.step_length, 0)
        else:
            quad_offset2 = (0, 0, 0)

        return quad_offset1, quad_offset2


