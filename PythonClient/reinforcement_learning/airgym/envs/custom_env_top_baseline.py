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
        
        self.current_drone="DroneFollower1"
        self.action_space = spaces.Discrete(4)
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip_address)
        self.drone.confirmConnection()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)
        self.track = []
        
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
        if self.current_drone == "DroneFollower1":
            self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
            self.drone.simSetSegmentationObjectID("DroneFollower1", 10, True)
            self.drone.simSetSegmentationObjectID("DroneFollower2", 20, True)
            self.drone.simSetSegmentationObjectID("detector_1", 30, True)
            
        elif self.current_drone == "DroneFollower2":
            self.drone.simSetSegmentationObjectID("[\w]*", 0, True)
            self.drone.simSetSegmentationObjectID("DroneFollower1", 20, True)
            self.drone.simSetSegmentationObjectID("DroneFollower2", 10, True)
            self.drone.simSetSegmentationObjectID("detector_6", 30, True)
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
        cv2.imwrite(self.current_drone+".png",im_final_seg)
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


