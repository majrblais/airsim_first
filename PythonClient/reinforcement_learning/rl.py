import airsim
import os
from PIL import Image
import numpy as np
from gym import spaces
# = airsim.MultirotorClient()
#client.confirmConnection()

#print(client.getMultirotorState())




class AirSimDroneEnv():
    def __init__(self, step_length=1, image_shape=(84, 84, 1)):
        self.step_length = step_length
        self.image_shape = image_shape
        
        #Current state of agent
        self.state = {"position": np.zeros(3),"collision": False,"prev_position": np.zeros(3),"val_seg": 0.0,}
        
        #Used to connect to ue/rl
        self.drone = airsim.MultirotorClient(ip="127.0.0.1")
        self.drone.confirmConnection()
        
        self.action_space = spaces.Discrete(7)
        self._setup_flight()
        
        
    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        self.drone.moveToPositionAsync(0, 0, -20, 5).join()
        #self.drone.moveByVelocityAsync(1,1, 1, 5).join()
        
        
    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()

        return obs, reward, done, self.state
        
    
    
    
    def _do_action(self, action):
        offset = self.interpret_action(action)
        pos=self.drone.getMultirotorState().kinematics_estimated.position
        self.drone.moveToPositionAsync(pos.x_val + offset[0],pos.y_val + offset[1],pos.z_val + offset[2], 5).join()
        

        
    def interpret_action(self, action):
        
        #move forward
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        #move right
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        #move up
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        #move backward
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        #move left    
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        #move down    
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)
        #dont move    
        else:
            quad_offset = (0, 0, 0)

        return quad_offset


F=AirSimDroneEnv()
print("forward")
F._do_action(0)
print("right")
F._do_action(1)
print("up")
F._do_action(2)
print("backward")
F._do_action(3)
print("left")
F._do_action(4)
print("down")
F._do_action(5)
print("Nothing")
F._do_action(7)
