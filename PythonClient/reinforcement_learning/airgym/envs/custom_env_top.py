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
        
        #set everything black except 2drones and 2fire spots
        success= self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
        success = self.drone.simSetSegmentationObjectID("DroneFollower1", 10, True) #[29, 26, 199]
        success = self.drone.simSetSegmentationObjectID("DroneFollower2", 20, True)  # [146, 52, 70]      
        success = self.drone.simSetSegmentationObjectID("detector_1", 30, True) #[226, 149, 143], pos:-50, 15, -50
        success = self.drone.simSetSegmentationObjectID("detector_6", 40, True) #[151, 126, 171, pos:78, 69, -50 (detector_2 is written as detector_6 due to ue4)
        
        self.pos_1=[np.array([-45.0,20.0,-50.0]),]
        self.pos_6=[np.array([30.0,40.0,-50.0]),]
        
        
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
        
        #Controllable setup
        self.drone.enableApiControl(True, "DroneFollower1")
        self.drone.enableApiControl(True, "DroneFollower2")
        
        self.drone.armDisarm(True, "DroneFollower1")
        self.drone.armDisarm(True, "DroneFollower2")
        
        #takeoff, move to side by side
        self.drone.takeoffAsync(vehicle_name="DroneFollower1").join()
        self.drone.moveToPositionAsync(5, 0, -50, 1, vehicle_name="DroneFollower1").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower1").join()
        
        self.drone.takeoffAsync(vehicle_name="DroneFollower2").join()
        self.drone.moveToPositionAsync(-5, 0, -50, 1, vehicle_name="DroneFollower2").join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1, vehicle_name="DroneFollower2").join()
        
        
        
        
        
    def step(self, action,drone_name):
        
        self._do_action(action,drone_name)
        
        obs = self._get_obs()
        reward, done = self._compute_reward(drone_name)

        return obs, reward, done, self.state
        
    def _compute_reward(self, drone_name):
        #desired location & current location
        
        
        if drone_name == "DroneFollower1":
            end_pts = self.pos_1
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
                reward = - np.sqrt(dist)
                print("reward:")
                print(reward)
                
            done=0
            if reward <=-25:
                done=1
            return reward,done
            
        elif drone_name == "DroneFollower2":
            end_pts = self.pos_6
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
                reward = - np.sqrt(dist)
                print("reward:")
                print(reward)
                
            done=0
            if reward <=-25:
                done=1
            return reward,done
            
        else:
            print("crit error, reward")
            exit()
    
    
    def _get_obs(self):
        import cv2
    
        responses = self.drone.simGetImages([self.image_request_rgb,self.image_request_seg], external=True)
        response_rgb = responses[0]
        response_seg = responses[1]
        
        
        img = np.fromstring(response_rgb.image_data_uint8, dtype=np.uint8)
        img_rgb = img.reshape(response_rgb.height, response_rgb.width, 3)
        try:
            image = Image.fromarray(img_rgb)
            im_final = np.array(image.resize((1080, 1080)))

        except:
            im_final = np.zeros((1080,1080,3), np.uint8)
        
        
        img_seg = np.fromstring(response_seg.image_data_uint8, dtype=np.uint8)
        img_rgb_seg = img_seg.reshape(response_seg.height, response_seg.width, 3)
        
        try:
            image_seg = Image.fromarray(img_rgb_seg)
            im_final_seg = np.array(image_seg.resize((1080, 1080)))

        except:
            im_final_seg = np.zeros((1080,1080,3), np.uint8)




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

