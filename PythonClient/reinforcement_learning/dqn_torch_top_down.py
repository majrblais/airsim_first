import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image
from torchvision import models
from torchsummary import summary
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TransitionLeader = namedtuple('Transition',('state_i0', 'action','next_state_i0','reward'))


class ReplayMemoryL(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(TransitionLeader(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)


class DQNLeader(nn.Module):
    def __init__(self):
        super(DQNLeader, self).__init__()
        
        self.conv1a = nn.Conv2d(3, 256, kernel_size=5, stride=2)
        self.bn1a = nn.BatchNorm2d(256)
        self.conv2a = nn.Conv2d(256, 128, kernel_size=5, stride=4)
        self.bn2a = nn.BatchNorm2d(128)
        self.conv3a = nn.Conv2d(128, 64, kernel_size=5, stride=4)
        self.bn3a = nn.BatchNorm2d(64)
        self.conv4a = nn.Conv2d(64, 32, kernel_size=5, stride=4)
        self.bn4a = nn.BatchNorm2d(32)

        self.lin5a = nn.Linear(2048,1024) 
        self.lin6a = nn.Linear(1024,256)  
        self.lin6b = nn.Linear(256,128)   
        self.lin4d = nn.Linear(128,64)
        self.lin5d = nn.Linear(64,4)
        

        
        
    def forward(self, x_img):
        img1=x_img
        
        
        img1 = img1.to(device)
        img1 = F.relu(self.bn1a(self.conv1a(img1)))
        img1 = F.relu(self.bn2a(self.conv2a(img1)))
        img1 = F.relu(self.bn3a(self.conv3a(img1)))
        img1 = F.relu(self.bn4a(self.conv4a(img1)))
        img1 = img1.view(img1.size(0), -1)
        
        img1 = F.relu((self.lin5a(img1)))
        out1 = self.lin6a(img1)
        out1 = self.lin6b(out1)
        out1 = self.lin4d(out1)
        out1 = self.lin5d(out1)
        
        
        
        return out1


resize = T.Compose([T.ToPILImage(),T.Resize(1080, interpolation=Image.CUBIC),T.ToTensor()])


from airgym.envs import custom_env_top
env = custom_env_top.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=1, image_shape=(128, 128, 1),)

import cv2
import numpy as np
import numpy.ma as ma
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes
#https://microsoft.github.io/AirSim/seg_rgbs.txt
def get_screen(drone_name):
    img,img_seg = env._get_obs()

    if drone_name == "DroneFollower1":
        img_tosend=img.copy()
        mask_=img_seg.copy()
        #find dron1 and fire1
        mask_[np.where((mask_ == [199, 26, 29]).all(axis=2))] = [10,10,10]
        mask_[np.where((mask_ == [70, 52, 146]).all(axis=2))] = [20,20,20]
        mask_[np.where((mask_ == [123, 21, 124]).all(axis=2))] = [30,30,30]
        #keep only 1 channel with background, d1 and f1
        mask1=mask_[:,:,0]
        a=mask1!=0
        b=mask1!=10 #d1
        c=mask1!=20 #d2
        d=mask1!=30
        #make sure there is no other values with np.where.IF there is conflicting values (e.i other than 0,10 or 30) then switch it to 0. should not happen
        t=a|b|c|d
        newm=np.where(t,mask1,0)
        mask=torch.from_numpy(newm) 
        #find ids, remove first (background)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        #get masks->then boxes
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)  
        
        #draw boxes, drone will ALWAYS be green while destination will always be red. This allows us to add infinite drones.
        #d1,d2,p1 (green,red,blue)
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (0, 0, 255) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[1][0]),int(boxes[1][1])), (int(boxes[1][2]),int(boxes[1][3])), (0, 255, 0) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0) )
        
        #img_tosend=cv2.resize(img_tosend,(512,512))
        cv2.imwrite('p1.png',img_tosend)
        #transpose, resize and transform into tensor
        img = img_tosend.transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32) #/ 255
        img = torch.from_numpy(img)
        
        return resize(img).unsqueeze(0)
        
    elif drone_name == "DroneFollower2":
        img_tosend=img.copy()
        mask_=img_seg.copy()
        mask_[np.where((mask_ == [199, 26, 29]).all(axis=2))] = [10,10,10]
        mask_[np.where((mask_ == [70, 52, 146]).all(axis=2))] = [20,20,20]
        mask_[np.where((mask_ == [214, 254, 86]).all(axis=2))] = [40,40,40]
        mask1=mask_[:,:,0]
        a=mask1!=0
        b=mask1!=10 #d1
        c=mask1!=20 #d2
        d=mask1!=40
        t=a|b|c|d
        newm=np.where(t,mask1,0)
        mask=torch.from_numpy(newm) 
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        boxes = masks_to_boxes(masks)
        
        #d1,d2,p2 (red,green,blue)
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[0][0]),int(boxes[0][1])), (int(boxes[0][2]),int(boxes[0][3])), (0, 255, 0) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[1][0]),int(boxes[1][1])), (int(boxes[1][2]),int(boxes[1][3])), (0, 0, 255) )
        img_tosend=cv2.rectangle(img_tosend, (int(boxes[2][0]),int(boxes[2][1])), (int(boxes[2][2]),int(boxes[2][3])), (255, 0, 0) )

        cv2.imwrite('p2.png',img_tosend)
        img = img_tosend.transpose((2, 0, 1))
        img = np.ascontiguousarray(img, dtype=np.float32) #/ 255
        img = torch.from_numpy(img)
        return resize(img).unsqueeze(0)
        
    else:
        print("crititcal error, F")
        exit()

BATCH_SIZE = 16
GAMMA = 0.997
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


img  = get_screen(drone_name="DroneFollower1")
img  = get_screen(drone_name="DroneFollower2")

#print(n_actions)

policy_netLeader = DQNLeader().to(device)
print(policy_netLeader)
target_netLeader = DQNLeader().to(device)
target_netLeader.load_state_dict(policy_netLeader.state_dict())

optimizerLeader = optim.AdamW(policy_netLeader.parameters())

memoryLeader = ReplayMemoryL(2000)

steps_done = 0

def select_action(img):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
    #if True:
        with torch.no_grad():
            act=policy_netLeader(x_img=img)
            act1=act.max(1)[1].view(1, 1)
            return act1


    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long),torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        
episode_durations = []

def optimize_model():
    
    
    if (len(memoryLeader) < BATCH_SIZE):
        return
        
    transitions = memoryLeader.sample(BATCH_SIZE)
    batch = TransitionLeader(*zip(*transitions))

    if all(s is None for s in batch.next_state_i0):
        return
         
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    
    
    print("loss1")
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask_i0 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_i0)), device=device, dtype=torch.bool)

    non_final_next_states_i0 = torch.cat([s for s in batch.next_state_i0 if s is not None])

                                                     
    state_batch_i0 = torch.cat(batch.state_i0)
    action_batch1 = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values1=policy_netLeader(x_img=state_batch_i0).gather(1, action_batch1)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values1 = torch.zeros(BATCH_SIZE, device=device)
    next_state_values1[non_final_mask_i0]=(target_netLeader(x_img=non_final_next_states_i0)).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values1 = (next_state_values1 * GAMMA) + reward_batch
    
    #Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss1 = criterion(state_action_values1, expected_state_action_values1.unsqueeze(1))

    optimizerLeader.zero_grad()
    loss1.backward()
    for param in policy_netLeader.parameters():
        #print(param.grad.data)
        param.grad.data.clamp_(-1, 1)
    optimizerLeader.step()


rew = []
num_episodes = 2500

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


tot=0
for i_episode in range(num_episodes):
    env.reset()

    
    init_img = get_screen(drone_name="DroneFollower1")
    last_img = init_img
    state_img = init_img
    max_re1=-25
    max_re2=-25
    
    for t in count():

        
        action = select_action(img=state_img)
        
        _, reward_, done, _ = env.step(action.item(),drone_name="DroneFollower1")
        
        if reward_>max_re1:
            max_re1=reward_
            
        reward = torch.tensor([reward_], device=device)
        last_img= init_img
        current_screen = get_screen(drone_name="DroneFollower1")
        
        if not done:
            next_state=  current_screen
        else:
             next_state = None

        memoryLeader.push(state_img, action,next_state, reward)
        state_img = next_state
        
        
        if not done:
            init_img = get_screen(drone_name="DroneFollower2")
            last_img = init_img
            state_img = init_img
            action = select_action(img=state_img)

            _, reward_, done, _ = env.step(action.item(),drone_name="DroneFollower2")
            
            if reward_>max_re2:
                max_re2=reward_
                
            reward = torch.tensor([reward_], device=device)
        

            last_img= init_img
            current_screen = get_screen(drone_name="DroneFollower2")

            if not done:
                next_state=  current_screen
            else:
                next_state = None

            memoryLeader.push(state_img, action,next_state, reward)
            state_img = next_state
        

        optimize_model()
        if done:
            print("done")
            #episode_durations.append(t + 1)
            row_contents = [tot, max_re1, max_re2]
            tot+=1
            append_list_as_row('out.csv', row_contents)
            #plot_durations()
            break

        # Update the target network, copying all weights and biases in DQN
        if t % TARGET_UPDATE == 0:
            target_netLeader.load_state_dict(policy_netLeader.state_dict())
        

print('Complete')
#print(env._get_obs())

torch.save(target_netLeader.state_dict(), './model_under_2d.pth')

