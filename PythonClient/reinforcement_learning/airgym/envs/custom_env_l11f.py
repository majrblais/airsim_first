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
        self.state = {"position0": np.zeros(3),"positionLeader": np.zeros(3),"positionFollower": np.zeros(3), "collisionLeader": False, "collisionFollower": False , "prev_positionLeader": np.zeros(3), "prev_positionFollower": np.zeros(3)}
        
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

          
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
        
        self.drone.armDisarm(True, "DroneLeader")
        self.drone.armDisarm(True, "DroneFollower1")

        self.drone.moveToPositionAsync(0, 0, -75, 5, vehicle_name="DroneLeader").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5, vehicle_name="DroneLeader").join()
        
        self.drone.moveToPositionAsync(2, 0, -75, 5, vehicle_name="DroneFollower1").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 5, vehicle_name="DroneFollower1").join()
        
        
        self.state["track_pos"]=0
        
        
    def step(self, action,drone_name):
        
        self._do_action(action,drone_name)
        
        obs = self._get_obs()
        reward, done = self._compute_reward(drone_name)
        
        return obs, reward, done, self.state
        
    def _compute_reward(self, drone_name):
        #desired location & current location
        
        
        if drone_name == "DroneLeader":
            end_pts = [np.array([-225, -950, -85.0]),]
            quad_pt0 = np.array(list((self.state["positionLeader"].x_val,self.state["positionLeader"].y_val,self.state["positionLeader"].z_val,)))
            #should theo not happen since it flys higher
            if self.state["collisionLeader"]:
                reward = -100
                print("collisionLeader")
            else:
                dist = np.linalg.norm(end_pts[0][0:2]-quad_pt0[0:2])
                reward = - np.sqrt(dist)/10
                print("rewardLeader:")
                print(reward)
                
            done=0
            if reward <=-30:
                done=1
            return reward,done


        elif drone_name == "DroneFollower1":
            #leader position
            pts = np.array(list((self.state["positionLeader"].x_val,self.state["positionLeader"].y_val,self.state["positionLeader"].z_val,)))
            #followers 
            quad_pt1 = np.array(list((self.state["positionFollower"].x_val,self.state["positionFollower"].y_val,self.state["positionFollower"].z_val,)))
            

            #if collide (note, even if the loop is similar the rewards are specific to the drone, made to be mod.)
            #If we used an or then we couldnt be sure if drone 1 collided even when we look at drone 2
            if self.state["collisionFollower"]:
                reward = -100
                print("collisionFollower")


            #function to determine if distance of follower# from leader is at less than X
            elif (np.linalg.norm(pts[0:2]-quad_pt1[0:2]) >= 5) :
                reward = -100
                print("DistancefromLeader")

                
                
            else:
                dist = np.linalg.norm(pts[0:2]-quad_pt1[0:2])
                reward = -dist
                print("rewardFollower:")
                print(reward)
                

            #print(reward)            
            done = 0
            if reward <=-10:
                done = 1

            return reward, done  
        
        else:
            print("Critical error")
            exit()
    
    
    def _get_obs(self):
    
        self.drone_state0 = self.drone.getMultirotorState(vehicle_name="DroneLeader")
        self.drone_state1 = self.drone.getMultirotorState(vehicle_name="DroneFollower1")
        
        #leader position
        self.state["positionLeader"] = self.drone_state0.kinematics_estimated.position

        
        self.state["prev_positionFollower"] = self.state["prev_positionFollower"]
        self.state["positionFollower"] = self.drone_state1.kinematics_estimated.position
        
        
        collision1 = self.drone.simGetCollisionInfo(vehicle_name="DroneLeader").has_collided
        collision2 = self.drone.simGetCollisionInfo(vehicle_name="DroneFollower1").has_collided
        
        self.state["collisionLeader"] = collision1
        self.state["collisionFollower"] = collision2
        
        

        return  self.drone_state0.kinematics_estimated.position, self.drone_state1.kinematics_estimated.position
    
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


