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
        
        self.conv1a = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1a = nn.BatchNorm2d(16)
        self.conv2a = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2a = nn.BatchNorm2d(32)
        self.conv3a = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3a = nn.BatchNorm2d(32)
        self.lin5a = nn.Linear(128,128)

        self.conv1b = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1b = nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2b = nn.BatchNorm2d(32)
        self.conv3b = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3b = nn.BatchNorm2d(32)
        self.lin5b = nn.Linear(128,128)
        
        
        self.lin2c = nn.Linear(256,128)
        
        
        self.lin4d = nn.Linear(128,64)
        self.lin5d = nn.Linear(64,4)
        
        self.lin4e = nn.Linear(128,64)
        self.lin5e = nn.Linear(64,4)

        
        
    def forward(self, x_img ,x_img_seg):
        img1=x_img
        img2=x_img_seg
        
        
        img1 = img1.to(device)
        img1 = F.relu(self.bn1a(self.conv1a(img1)))
        img1 = F.relu(self.bn2a(self.conv2a(img1)))
        img1 = F.relu(self.bn3a(self.conv3a(img1)))
        img1 = img1.view(img1.size(0), -1)
        img1 = F.relu((self.lin5a(img1)))

        img2 = img2.to(device)
        img2 = F.relu(self.bn1b(self.conv1b(img2)))
        img2 = F.relu(self.bn2b(self.conv2b(img2)))
        img2 = F.relu(self.bn3b(self.conv3b(img2)))
        img2 = img2.view(img2.size(0), -1)
        img2 = F.relu((self.lin5b(img2)))

        combinedimg = torch.cat((img1,img2),1)
        combinedimg = combinedimg.view(combinedimg.size(0), -1)
        
        outc = self.lin2c(combinedimg)
        
        
        out1 = self.lin4d(outc)
        out1 = self.lin5d(out1)
        
        out2 = self.lin4e(outc)
        out2 = self.lin5e(out2)
        
        
        return out1. out2


resize = T.Compose([T.ToPILImage(),T.Resize(128, interpolation=Image.CUBIC),T.ToTensor()])


from airgym.envs import custom_env_top
env = custom_env_top.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=1, image_shape=(128, 128, 1),)


def get_screen():
    img,img_seg = env._get_obs()
    
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) #/ 255
    img = torch.from_numpy(img)
    
    img_seg = img_seg.transpose((2, 0, 1))
    img_seg = np.ascontiguousarray(img_seg, dtype=np.float32) #/ 255
    img_seg = torch.from_numpy(img_seg)
    
    
    return resize(img).unsqueeze(0), resize(img_seg).unsqueeze(0)

BATCH_SIZE = 4
GAMMA = 0.997
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


img, img_seg  = get_screen()


#print(n_actions)

policy_netLeader = DQNLeader().to(device)
print(policy_netLeader)
target_netLeader = DQNLeader().to(device)
target_netLeader.load_state_dict(policy_netLeader.state_dict())

optimizerLeader = optim.AdamW(policy_netLeader.parameters())

memoryLeader = ReplayMemoryL(2000)

steps_done = 0

def select_action(img, img_seg):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #if sample > eps_threshold:
    if True:
        with torch.no_grad():
            act=policy_netLeader(x_img=img,x_img_seg=img_seg)
            act1=act[0].max(1)[1].view(1, 1)
            act2=act[1].max(1)[1].view(1, 1)
            return act1,act2


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
        
    if all(s is None for s in batch.next_state_p3):
        return
        
        
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    
    
    print("loss1")
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask_i0 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_i0)), device=device, dtype=torch.bool)
    non_final_mask_p3 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p3)), device=device, dtype=torch.bool)

    non_final_next_states_i0 = torch.cat([s for s in batch.next_state_i0 if s is not None])
    non_final_next_states_p3 = torch.cat([s for s in batch.next_state_p3 if s is not None])

                                                     
    state_batch_i0 = torch.cat(batch.state_i0)
    state_batch_p3 = torch.cat(batch.state_p3)
    action_batch1 = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values1=policy_netLeader(x_img=state_batch_i0,x_p3=state_batch_p3).gather(1, action_batch1)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values1 = torch.zeros(BATCH_SIZE, device=device)
    next_state_values1[non_final_mask_p3]=(target_netLeader(x_img=non_final_next_states_i0,x_p3=non_final_next_states_p3)).max(1)[0].detach()
    
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
num_episodes = 500
max_re= -10

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


tot=0
for i_episode in range(num_episodes):
    max_re = -25
    env.reset()

    
    init_img, init_img_seg = get_screen()
    last_img, last_img_seg = get_screen()
    state_img,state_img_seg = init_img, init_img_seg
    
    
    for t in count():

        
        action1, action2 = select_action(x_img=state_img,x_img_seg=state_img_seg)
        
        _, reward_, done, _ = env.step(action1.item(),drone_name="DroneFollower1")
        
        _, reward_, done, _ = env.step(action2.item(),drone_name="DroneFollower2")
        
        
        reward = torch.tensor([reward_], device=device)
        if reward_ > max_re:
            max_re_leader = reward_
        
        last_screen_p1, last_img_1, last_screen_p2, last_screen_p3=  init_screen_p1,init_img_1, init_screen_p2, init_screen_p3 
        current_screen_p1,current_screen_i1, current_screen_p2, current_screen_p3 = get_screen()
        
        if not done:
            next_state_p1,next_state_i1, next_state_p2, next_state_p3 =  current_screen_p1,current_screen_i1, current_screen_p2, current_screen_p3 
        else:
             next_state_p1,next_state_i1, next_state_p2, next_state_p3 = None, None, None, None

        memoryLeader.push(state_i1,state_p3, action,next_state_i1, next_state_p3, reward)
        state_p1,state_i1, state_p2, state_p3 = next_state_p1,next_state_i1, next_state_p2, next_state_p3
        

        optimize_modelFollower()
        if done:
            print("done")
            episode_durations.append(t + 1)
            row_contents = [tot, max_re]
            tot+=1
            append_list_as_row('out.csv', row_contents)
            #plot_durations()
            break

        # Update the target network, copying all weights and biases in DQN
        if t % TARGET_UPDATE == 0:
            target_netLeader.load_state_dict(policy_netLeader.state_dict())
            target_netFollower.load_state_dict(policy_netFollower.state_dict())
        

print('Complete')
#print(env._get_obs())

torch.save(policy_net.state_dict(), './model_under_cam.pth')
env.close()
plt.ioff()
plt.show()
print(rew)

plt.figure(2)
plt.clf()
durations_t = torch.tensor(rew, dtype=torch.float)
plt.title('Training...')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(durations_t.numpy())
# Take 100 episode averages and plot them too
if len(durations_t) >= 100:
    means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    means = torch.cat((torch.zeros(99), means))
    plt.plot(means.numpy())

plt.pause(0.001)  # pause a bit so that plots are updated
plt.savefig('graph.png')


