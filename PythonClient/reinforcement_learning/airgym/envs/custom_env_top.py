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
        self.state = {"position0": np.zeros(3),"position1": np.zeros(3),"collision0": False, "collision1": False, "collision2": False ,"prev_position0": np.zeros(3), "prev_position1": np.zeros(3)}
        
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.track = []
        
        
        success= self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
        print(success)
        
        
        success = self.drone.simSetSegmentationObjectID("DroneFollower1", 25, True);
        print(success)
        
        success = self.drone.simSetSegmentationObjectID("DroneFollower2", 78, True);
        print(success)        
        
        #s=self.drone.simSetSegmentationObjectID("Drone[\w]*", 255, True)
        #print(s)
        
        print(self.drone.simGetSegmentationObjectID("DroneFollower1"))
        #print(self.drone.simGetSegmentationObjectID("DroneFollower2"))
        
        
        self.image_request_seg = airsim.ImageRequest(3, airsim.ImageType.Segmentation, False, False)
        self.image_request_rgb = airsim.ImageRequest(3, airsim.ImageType.Scene, False, False)
          
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
        
        
        
        
        
    def step(self, action,drone_name):
        
        self._do_action(action,drone_name)
        
        obs = self._get_obs()
        reward, done = self._compute_reward(drone_name)

        return obs, reward, done, self.state
        
    def _compute_reward(self, drone_name):
        #desired location & current location
        
        
        if drone_name == "DroneLeader":
            end_pts = [np.array([-225, -950, -85.0]),]
            quad_pt0 = np.array(list((self.state["position0"].x_val,self.state["position0"].y_val,self.state["position0"].z_val,)))
            #should theo not happen since it flys higher
            if self.state["collision0"]:
                reward = -100
                print("collisionLeader")
            else:
                dist = np.linalg.norm(end_pts[0][0:2]-quad_pt0[0:2])
                reward = - np.sqrt(dist)/10
                print("reward:")
                print(reward)
                
            done=0
            if reward <=-15:
                done=1
            return reward,done


        elif drone_name == "DroneFollower1" or drone_name == "DroneFollower2":
            #leader position
            pts = np.array(list((self.state["position0"].x_val,self.state["position0"].y_val,self.state["position0"].z_val,)))
            #followers 
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
            elif np.linalg.norm(quad_pt1[0:2]-quad_pt2[0:2]) >= 20:
                reward = -100
                print("distance")
            
            #function to determine if distance of follower# from leader is at less than X
            elif (np.linalg.norm(pts[0:2]-quad_pt1[0:2]) >= 10) and drone_name == "DroneFollower1":
                reward = -100
                print("circle1")
                
            elif (np.linalg.norm(pts[0:2]-quad_pt2[0:2]) >= 10) and drone_name == "DroneFollower2":
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
            if reward <=-25 and drone_name == "DroneLeader":
                done = 1
            if reward <=-25 and drone_name != "DroneLeader":
                done = 1
                
            return reward, done    
    
    
    def _get_obs(self):
        import cv2
    
        responses = self.drone.simGetImages([self.image_request_rgb],vehicle_name="DroneLeader")
        response = responses[0]
        
        img = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img.reshape(response.height, response.width, 3)
        try:
            image = Image.fromarray(img_rgb)
            im_final = np.array(image.resize((128, 128)))

        except:
            im_final = np.zeros((128,128,3), np.uint8)
        
        

        
        responses_seg = self.drone.simGetImages([self.image_request_seg],vehicle_name="DroneLeader")
        response_seg = responses_seg[0]
        img_seg = np.fromstring(response_seg.image_data_uint8, dtype=np.uint8)
        img_rgb_seg = img_seg.reshape(response_seg.height, response_seg.width, 3)
        
        try:
            image_seg = Image.fromarray(img_rgb_seg)
            im_final_seg = np.array(image_seg.resize((128, 128)))

        except:
            im_final_seg = np.zeros((128,128,3), np.uint8)




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

        

        return im_final, im_final_seg
    
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


