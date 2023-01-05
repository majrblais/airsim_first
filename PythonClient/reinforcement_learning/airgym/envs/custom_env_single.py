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
        self.state = {"position0": np.zeros(3),"position1": np.zeros(3),"position2": np.zeros(3), "collision1": False, "collision2": False , "prev_position1": np.zeros(3), "prev_position2": np.zeros(3), "track_pos" : 0}
        
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.track = []
        for i in range(1,201,5):
            self.track.append([0,-i])

        for i in range(1,101,5):
            self.track.append([-i,-200])

        for i in range(1,301,5):
            self.track.append([-100,-200-i])

        for i in range(1,126,5):
            self.track.append([-100-i,-500])

        for i in range(1,401,5):
            self.track.append([-225,-500-i])
          
        self._setup_flight()
    
    
    def reset(self):
        self._setup_flight()
        return self._get_obs()

    def __del__(self):
        self.drone.reset()    
    
    def _setup_flight(self):
        print("setting-up")
        self.drone.reset()
        
        self.drone.enableApiControl(True, "DroneLeader")
        self.drone.enableApiControl(True, "DroneFollower1")
        self.drone.enableApiControl(True, "DroneFollower2")
        
        self.drone.armDisarm(True, "DroneLeader")
        self.drone.armDisarm(True, "DroneFollower1")
        self.drone.armDisarm(True, "DroneFollower2")

        self.drone.moveToPositionAsync(0, 0, -80, 5, vehicle_name="DroneLeader").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5, vehicle_name="DroneLeader").join()
        
        self.drone.moveToPositionAsync(2, 0, -74, 5, vehicle_name="DroneFollower1").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5, vehicle_name="DroneFollower1").join()
        
        self.drone.moveToPositionAsync(-2, 0, -76, 5, vehicle_name="DroneFollower2").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5, vehicle_name="DroneFollower2").join()
        
        
        self.state["track_pos"]=0
        
        
    def step(self, action,drone_name):
        
        self._do_action(action,drone_name)
        
        obs = self._get_obs()
        reward, done = self._compute_reward(drone_name)
        self.state["track_pos"]+=1
        

        self.drone.moveToPositionAsync(self.track[self.state["track_pos"]][0], self.track[self.state["track_pos"]][1], self.state["position0"].z_val, 1, vehicle_name="DroneLeader").join()
        

        return obs, reward, done, self.state
        
    def _compute_reward(self, drone_name):
        #desired location & current location
        
        
        
        #followers 
        pts = np.array(list((self.state["position0"].x_val,self.state["position0"].y_val,self.state["position0"].z_val,)))
        quad_pt1 = np.array(list((self.state["position1"].x_val,self.state["position1"].y_val,self.state["position1"].z_val,)))
        quad_pt2 = np.array(list((self.state["position2"].x_val,self.state["position2"].y_val,self.state["position2"].z_val,)))
        

        #if collide (note, even if the loop is similar the rewards are specific to the drone, made to be mod.)
        #If we used an or then we couldnt be sure if drone 1 collided even when we look at drone 2
        if self.state["collision1"] and drone_name == "DroneFollower1":
            reward = -100
            print("collision1")
            
        elif self.state["collision2"] and drone_name == "DroneFollower2":
            reward = -100
            print("collision2")    

        #same function for both followers, distance between the two followers must be smaller than X
        elif np.linalg.norm(quad_pt1[0:2]-quad_pt2[0:2]) >= 30:
            reward = -100
            print("distance")
        
        #function to determine if distance of follower# from leader is at less than X
        elif (np.linalg.norm(pts[0:2]-quad_pt1[0:2]) >= 20) and drone_name == "DroneFollower1":
            reward = -100
            print("circle1")
            
        elif (np.linalg.norm(pts[0:2]-quad_pt2[0:2]) >= 20) and drone_name == "DroneFollower2":
            reward = -100
            print("circle2")
            
            
        elif drone_name == "DroneFollower1":
            dist = np.linalg.norm(pts[0:2]-quad_pt1[0:2])
            reward = -dist
            
        elif drone_name == "DroneFollower2":
            dist = np.linalg.norm(pts[0:2]-quad_pt2[0:2])
            reward = -dist
            
            
        else:
            reward=0
            
            

        #print(reward)            
        done = 0
        if reward <=-30:
            done = 1
            
        return reward, done    
    
    def _get_obs(self):
    
        self.drone_state0 = self.drone.getMultirotorState(vehicle_name="DroneLeader")
        self.drone_state1 = self.drone.getMultirotorState(vehicle_name="DroneFollower1")
        self.drone_state2 = self.drone.getMultirotorState(vehicle_name="DroneFollower2")
        
        #leader position
        self.state["position0"] = self.drone_state0.kinematics_estimated.position

        
        self.state["prev_position1"] = self.state["position1"]
        self.state["position1"] = self.drone_state1.kinematics_estimated.position
        self.state["velocity1"] = self.drone_state1.kinematics_estimated.linear_velocity
        
        self.state["prev_position2"] = self.state["position2"]
        self.state["position2"] = self.drone_state2.kinematics_estimated.position
        self.state["velocity2"] = self.drone_state2.kinematics_estimated.linear_velocity
        
        collision1 = self.drone.simGetCollisionInfo(vehicle_name="DroneFollower1").has_collided
        collision2 = self.drone.simGetCollisionInfo(vehicle_name="DroneFollower2").has_collided
        
        self.state["collision1"] = collision1
        self.state["collision2"] = collision2
        
        

        return self.drone_state1.kinematics_estimated.position, self.drone_state2.kinematics_estimated.position, self.drone_state0.kinematics_estimated.position
    
    def _do_action(self, action, drone_name):
        #print(action)
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


