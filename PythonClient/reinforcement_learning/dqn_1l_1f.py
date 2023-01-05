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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TransitionLeader = namedtuple('Transition',('state_leader', 'action','next_state_leader','reward'))

TransitionFollower = namedtuple('Transition',('state_foll','state_lead', 'action', 'next_state_foll', 'next_state_lead','reward'))


class ReplayMemoryL(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(TransitionLeader(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)

class ReplayMemoryF(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(TransitionFollower(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)


class DQNLeader(nn.Module):
    def __init__(self, ln):
        super(DQNLeader, self).__init__()
        self.lin1 = nn.Linear(ln,32)
        self.lin2 = nn.Linear(32,64)
        self.lin3 = nn.Linear(64,128)
        self.lin4 = nn.Linear(128,128)
        
        
        self.lin5 = nn.Linear(128,64)
        self.lin6 = nn.Linear(64,4)
        
        
    def forward(self, x_leader):
        p=x_leader
        p = p.to(device).float()
        p = self.lin1(p)
        p = self.lin2(p)
        p = self.lin3(p)
        p = self.lin4(p)
        p = self.lin5(p)
        out = self.lin6(p)
        
        return out

class DQNFollower(nn.Module):
    def __init__(self, ln1, ln2):
        super(DQNFollower, self).__init__()
        
        self.lin1a = nn.Linear(ln1,32)
        self.lin2a = nn.Linear(32,64)
        self.lin3a = nn.Linear(64,128)


        self.lin1b = nn.Linear(ln2,32)
        self.lin2b = nn.Linear(32,64)
        self.lin3b = nn.Linear(64,128)

        
        self.lin2d = nn.Linear(256,128)

        
        self.lin4e = nn.Linear(128,64)
        self.lin5e = nn.Linear(64,4)
        
        
        
    def forward(self, x_leader, x_follower):
        p1=x_leader
        p2=x_follower

        
        #if p2 is none means p1 will pass, if p1 is none then p2 is passed
        #its basioally inversed logically, it works as it should, don't mess w/it
        #could do p1 is not all(None) or something in that manner

        p1 = p1.to(device).float()
        p1 = self.lin1a(p1)
        p1 = self.lin2a(p1)
        p1 = self.lin3a(p1)
        

        p2 = p2.to(device).float()
        p2 = self.lin1b(p2)
        p2 = self.lin2b(p2)
        p2 = self.lin3b(p2)



        combinedp = torch.cat((p1,p2),1)
        outp = combinedp.view(combinedp.size(0), -1)   
        outp = self.lin2d(outp)
        out1 = self.lin4e(outp)
        out = self.lin5e(out1)
        return out
   

from airgym.envs import custom_env_l11f
env = custom_env_l11f.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=1, image_shape=(128, 128, 1),)


def get_screen():
    screen1,screen2 = env._get_obs()
   
    #x is leafer, y is follower
    x1=screen1.x_val
    y1=screen1.y_val
    z1=screen1.z_val

    x2=screen2.x_val
    y2=screen2.y_val
    z2=screen2.z_val

 

    #screen = np.array([float(x),float(y),float(z)])
    pos1 = np.array([float(str(round(x1, 2))),float(str(round(y1, 2)))])
    pos2 = np.array([float(str(round(x2, 2))),float(str(round(y2, 2)))])
    #screen = torch.from_numpy(np.expand_dims(screen, axis=0))
    pos1 = torch.from_numpy(pos1)
    pos2 = torch.from_numpy(pos2)
    
    return pos1.unsqueeze(0) , pos2.unsqueeze(0)


BATCH_SIZE = 256
GAMMA = 0.997
EPS_START = 0.90
EPS_END = 0.05
EPS_DECAY = 1000
TARGET_UPDATE = 100


init_screen_p1, init_screen_p2  = get_screen()

policy_netLeader = DQNLeader(ln=len(init_screen_p1[0])).to(device)
print(policy_netLeader)
target_netLeader = DQNLeader(ln=len(init_screen_p1[0])).to(device)
target_netLeader.load_state_dict(policy_netLeader.state_dict())



policy_netFollower= DQNFollower(ln1=len(init_screen_p1[0]),ln2=len(init_screen_p2[0])).to(device)
print(policy_netFollower)
target_netFollower = DQNFollower(ln1=len(init_screen_p1[0]),ln2=len(init_screen_p2[0])).to(device)
target_netFollower.load_state_dict(policy_netFollower.state_dict())

optimizerFollower = optim.AdamW(policy_netFollower.parameters())
optimizerLeader = optim.AdamW(policy_netFollower.parameters())


memoryFollower = ReplayMemoryF(5000)
memoryLeader = ReplayMemoryL(5000)


steps_done = 0

def select_action(state_p1=None, state_p2=None,mode=None):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
    #if True:
        with torch.no_grad():
            if mode == "Leader":
                act=policy_netLeader(x_leader=state_p1)
                act1=act.max(1)[1].view(1, 1)
                return act1
                
            if mode == "Follower":
                act=policy_netFollower(x_leader=state_p1,x_follower=state_p2)
                act2=act.max(1)[1].view(1, 1)
                return act2

    else:
        return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        #return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long),torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        
episode_durations = []



def optimize_modelLeader():
    
    
    if (len(memoryLeader) < BATCH_SIZE):
        return
        
    transitions = memoryLeader.sample(BATCH_SIZE)
    batch = TransitionLeader(*zip(*transitions))

    if all(s is None for s in batch.next_state_leader):
        return
        
        
        
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask_leader = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_leader)), device=device, dtype=torch.bool)

    non_final_next_states_leader = torch.cat([s for s in batch.next_state_leader if s is not None])

                                                    
    state_batch_leader = torch.cat(batch.state_leader)
    action_batch1 = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values1=policy_netLeader(x_leader=state_batch_leader).gather(1, action_batch1)
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values1 = torch.zeros(BATCH_SIZE, device=device)
    next_state_values1[non_final_mask_leader]=(target_netLeader(x_leader=non_final_next_states_leader)).max(1)[0].detach()
    
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


def optimize_modelFollower():
    

    if (len(memoryFollower) < BATCH_SIZE):
        return

    transitions = memoryFollower.sample(BATCH_SIZE)
    batch = TransitionFollower(*zip(*transitions))
    
    if all(s is None for s in batch.next_state_lead):
        return
        
    if all(s is None for s in batch.next_state_foll):
        return
  

    non_final_mask_lead = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_lead)), device=device, dtype=torch.bool)
    non_final_mask_foll = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_foll)), device=device, dtype=torch.bool)


    non_final_next_states_lead = torch.cat([s for s in batch.next_state_lead if s is not None])
    non_final_next_states_foll = torch.cat([s for s in batch.next_state_foll if s is not None])

                                                     
    state_batch_lead = torch.cat(batch.state_lead)
    state_batch_foll = torch.cat(batch.state_foll)
    action_batch2 = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    
    state_action_values2 = policy_netFollower(x_leader=state_batch_lead,x_follower=state_batch_foll).gather(1, action_batch2)

    next_state_values2 = torch.zeros(BATCH_SIZE, device=device)
    next_state_values2[non_final_mask_foll]=(target_netFollower(x_leader=non_final_next_states_lead, x_follower=non_final_next_states_foll)).max(1)[0].detach()
    
    expected_state_action_values2 = (next_state_values2 * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss2 = criterion(state_action_values2, expected_state_action_values2.unsqueeze(1))
    #loss3 = criterion(state_action_values3, expected_state_action_values3.unsqueeze(1))
    
    
    optimizerFollower.zero_grad()
    loss2.backward()
    for param in policy_netFollower.parameters():
        param.grad.data.clamp_(-1, 1)
    
    optimizerFollower.step()

rew = []
num_episodes = 500
max_re= -20

from csv import writer
def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a+', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


tot=0
for i_episode in range(num_episodes):
    max_re_lead = -25
    max_re_foll = -25
    
    # Initialize the environment and state
    env.reset()
    #last_screen = get_screen()
    #current_screen = get_screen()
    #state = current_screen #- last_screen
    
    init_screen_p1_leader, init_screen_p2_follower  = get_screen()
    last_screen_p1_leader, last_screen_p2_follower  = get_screen()
    state_p1_leader, state_p2_follower = init_screen_p1_leader, init_screen_p2_follower
    
    
    for t in count():
        curr=0
        # Select and perform an action
        #print(state)
        #print(state.shape)
        
        action = select_action(state_p1=state_p1_leader,mode="Leader")
        _, reward_lead, done, _ = env.step(action.item(),drone_name="DroneLeader")
        reward = torch.tensor([reward_lead], device=device)
        if reward_lead > max_re_lead:
            max_re_leader = reward_lead
        
        last_screen_p1_leader, last_screen_p2_follower=  init_screen_p1_leader, init_screen_p2_follower 
        current_screen_p1_leader, current_screen_p2_follower = get_screen()
        
        if not done:
            next_state_p1_leader, next_state_p2_follower, =  current_screen_p1_leader, current_screen_p2_follower 
        else:
             next_state_p1_leader, next_state_p2_follower = None, None

        memoryLeader.push(state_p1_leader, action, next_state_p1_leader, reward)
        state_p1_leader, state_p2_follower = next_state_p1_leader, next_state_p2_follower  
        
        optimize_modelLeader()

        
        #Follower1
        if not done:
            
            action = select_action(state_p1=state_p1_leader,state_p2=state_p2_follower,mode="Follower")
            _, reward_fol, done, _ = env.step(action.item(),drone_name="DroneFollower1")
            reward = torch.tensor([reward_fol], device=device)
            if reward_fol > max_re_foll:
                max_re_foll = reward_fol
            last_screen_p1_leader, last_screen_p2_follower=  init_screen_p1_leader, init_screen_p2_follower 
            current_screen_p1_leader, current_screen_p2_follower = get_screen()
            
            if not done:
                next_state_p1_leader, next_state_p2_follower =  current_screen_p1_leader, current_screen_p2_follower 
            else:
                 next_state_p1_leader, next_state_p2_follower = None, None

            memoryFollower.push(state_p1_leader, state_p2_follower, action, next_state_p1_leader, next_state_p2_follower, reward)
            state_p1_leader, state_p2_follower = next_state_p1_leader, next_state_p2_follower  
            
        
        # Perform one step of the optimization (on the policy network)
        optimize_modelFollower() 

        rows=[curr,state_p1_leader,reward_lead, state_p2_follower, reward_fol]
        append_list_as_row('out_all.csv', rows)
        curr+=1
        
        if done:
            print("done")
            episode_durations.append(t + 1)
            row_contents = [tot, max_re_lead,max_re_foll]
            tot+=1
            append_list_as_row('out.csv', row_contents)
            break

        # Update the target network, copying all weights and biases in DQN
        if t % TARGET_UPDATE == 0:
            target_netLeader.load_state_dict(policy_netLeader.state_dict())
            target_netFollower.load_state_dict(policy_netFollower.state_dict())
        

print('Complete')
#print(env._get_obs())

torch.save(policy_netLeader.state_dict(), './model_leader.pth')
torch.save(policy_netFollower.state_dict(), './model_follower.pth')




