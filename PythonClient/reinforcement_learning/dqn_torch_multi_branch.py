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
Transition = namedtuple('Transition',('state_p1','state_p2','state_p3', 'action1', 'action2', 'next_state_p1', 'next_state_p2','next_state_p3','reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
        
    def __len__(self):
        return len(self.memory)
        

class DQN(nn.Module):
    def __init__(self, ln1, ln2, ln3=None):
        super(DQN, self).__init__()
        
        self.lin1a = nn.Linear(ln1,32)
        self.lin2a = nn.Linear(32,64)
        self.lin3a = nn.Linear(64,128)


        self.lin1b = nn.Linear(ln2,32)
        self.lin2b = nn.Linear(32,64)
        self.lin3b = nn.Linear(64,128)

        self.lin1c = nn.Linear(ln3,32)
        self.lin2c = nn.Linear(32,64)
        self.lin3c = nn.Linear(64,128)
        
        
        
        
        self.lin2d = nn.Linear(256,128)

        
        self.lin4e = nn.Linear(128,64)
        self.lin5e = nn.Linear(64,4)
        
        self.lin4f = nn.Linear(128,64)
        self.lin5f = nn.Linear(64,4)
        
        
    def forward(self, x_p1=None, x_p2=None, x_p3=None):
        print("forward")
        p1=x_p1
        p2=x_p2
        p3=x_p3
        
        #if p2 is none means p1 will pass, if p1 is none then p2 is passed
        #its basioally inversed logically, it works as it should, don't mess w/it
        #could do p1 is not all(None) or something in that manner
        if p2 is None:
            print("p1")
            p1 = p1.to(device).float()
            p1 = self.lin1a(p1)
            p1 = self.lin2a(p1)
            p1 = self.lin3a(p1)
        
        if p1 is None:
            print("p2")
            p2 = p2.to(device).float()
            p2 = self.lin1b(p2)
            p2 = self.lin2b(p2)
            p2 = self.lin3b(p2)
        
        p3 = p3.to(device).float()
        p3 = self.lin1c(p3)
        p3 = self.lin2c(p3)
        p3 = self.lin3c(p3)
        
        if p2 is None :
            print("p1")
            combinedp = torch.cat((p1,p3),1)
            outp = combinedp.view(combinedp.size(0), -1)   
            outp = self.lin2d(outp)
            
        if p1 is None:
            print("p2")
            combinedp = torch.cat((p2, p3),1)
            outp = combinedp.view(combinedp.size(0), -1)   
            outp = self.lin2d(outp)
        #combinedp = torch.cat((p1,p2, p3),1)

        

        
        if p2 is None:
            print("p1")
            out1 = self.lin4e(outp)
            out1 = self.lin5e(out1)
            return out1
            
        if p1 is None:
            print("p2")
            out2 = self.lin4f(outp)
            out2 = self.lin5f(out2)
            return out2
        
        #return out1,out2
        
resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.CUBIC),T.ToTensor()])


from airgym.envs import custom_env_multi_2_branch
env = custom_env_multi_2_branch.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=1, image_shape=(128, 128, 1),)


def get_screen():
    screen1,screen2, screen3 = env._get_obs()
   
    x1=screen1.x_val
    y1=screen1.y_val
    z1=screen1.z_val

    x2=screen2.x_val
    y2=screen2.y_val
    z2=screen2.z_val
    
    x3=screen3[0]
    y3=screen3[1]
 

    #screen = np.array([float(x),float(y),float(z)])
    pos1 = np.array([float(x1),float(y1)])
    pos2 = np.array([float(x2),float(y2)])
    pos3 = np.array([float(x3),float(y3)])
    #screen = torch.from_numpy(np.expand_dims(screen, axis=0))
    pos1 = torch.from_numpy(pos1)
    pos2 = torch.from_numpy(pos2)
    pos3 = torch.from_numpy(pos3)
    
    
    
    return pos1.unsqueeze(0), pos2.unsqueeze(0) , pos3.unsqueeze(0)
    

BATCH_SIZE = 8
GAMMA = 0.997
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


init_screen_p1, init_screen_p2, init_screen_p3  = get_screen()


n_actions = env.action_space.n
#print(n_actions)

policy_net = DQN(ln1=len(init_screen_p1[0]),ln2=len(init_screen_p2[0]),ln3=len(init_screen_p3[0])).to(device)
print(policy_net)
target_net = DQN(ln1=len(init_screen_p1[0]),ln2=len(init_screen_p2[0]),ln3=len(init_screen_p3[0])).to(device)
target_net.load_state_dict(policy_net.state_dict())




def paramgen(policy_net=policy_net, p1=None, p2=None):
    ls1=["lin1a.weight","lin2a.weight","lin3a.weight","lin4e.weight", "lin5e.weight","lin1a.bias","lin2a.bias","lin3a.bias","lin4e.bias", "lin5e.bias"]
    ls2=["lin1b.weight","lin2b.weight","lin3b.weight","lin4f.weight", "lin5f.weight","lin1b.bias","lin2b.bias","lin3b.bias","lin4f.bias", "lin5f.bias"]
    for param in policy_net.named_parameters():

        #yield ls1 model, the in part is inverted in a sense
        
        if (param[0] not in ls2) and p1:
            yield param[1] 
        #yields ls2 model
        if (param[0] not in ls1) and p2:
            yield param[1]
    


#print(paramgen(policy_net,p1=1))
#print(paramgen(policy_net,p2=1))

#for p in paramgen(policy_net,p1=1):
#    print(p)

#for p in paramgen(policy_net,p2=1):
#    print(p) 
    

optimizer1 = optim.AdamW(paramgen(policy_net,p1=1))
optimizer2 = optim.AdamW(paramgen(policy_net,p2=1))

memory = ReplayMemory(2000)
steps_done = 0

def select_action(state_p1=None, state_p2=None, state_p3=None):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
    #if True:
        with torch.no_grad():
            if state_p2 is None:
                act=policy_net(x_p1=state_p1,x_p3=state_p3)
                act1=act.max(1)[1].view(1, 1)
                return act1
                
            if state_p1 is None:
                act=policy_net(x_p2=state_p2,x_p3=state_p3)
                act2=act.max(1)[1].view(1, 1)
                return act2
            #print(act[0].max(1)[1].view(1, 1))
            #print(act[1].max(1)[1].view(1, 1))
            
            #act1=act[0].max(1)[1].view(1, 1)
            #act2=act[1].max(1)[1].view(1, 1)
            
            #return act1,act2
    else:
        if state_p1:
            torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        if state_p2:
            torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        #return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long),torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long)
        
episode_durations = []

def optimize_model(p1=None, p2=None):
    
    
    if (len(memory) < BATCH_SIZE):
        return
    transitions = memory.sample(BATCH_SIZE)
    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    

    if all(s is None for s in batch.next_state_p1):
        return
        
    if all(s is None for s in batch.next_state_p2):
        return
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    
    
    print("loss1")
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask_p1 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p1)), device=device, dtype=torch.bool)
    non_final_mask_p3 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p3)), device=device, dtype=torch.bool)

    non_final_next_states_p1 = torch.cat([s for s in batch.next_state_p1 if s is not None])
    non_final_next_states_p3 = torch.cat([s for s in batch.next_state_p3 if s is not None])

                                                     

    state_batch_p1 = torch.cat(batch.state_p1)
    state_batch_p3 = torch.cat(batch.state_p3)
    action_batch1 = torch.cat(batch.action1)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values1=policy_net(x_p1=state_batch_p1,x_p3=state_batch_p3).gather(1, action_batch1)

    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values1 = torch.zeros(BATCH_SIZE, device=device)
    next_state_values1[non_final_mask_p1]=(target_net(x_p1=non_final_next_states_p1, x_p3=non_final_next_states_p3)).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values1 = (next_state_values1 * GAMMA) + reward_batch
    
    #Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss1 = criterion(state_action_values1, expected_state_action_values1.unsqueeze(1))

    optimizer1.zero_grad()
    loss1.backward()
    for param in paramgen(policy_net,p1=1):
        #print(param.grad.data)
        param.grad.data.clamp_(-1, 1)
    optimizer1.step()
    
    print("loss2")
    
    non_final_mask_p2 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p2)), device=device, dtype=torch.bool)
    non_final_mask_p3 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p3)), device=device, dtype=torch.bool)
    
    non_final_next_states_p2 = torch.cat([s for s in batch.next_state_p2 if s is not None])
    non_final_next_states_p3 = torch.cat([s for s in batch.next_state_p3 if s is not None])

    state_batch_p2 = torch.cat(batch.state_p2)
    state_batch_p3 = torch.cat(batch.state_p3)


    action_batch2 = torch.cat(batch.action2)
    reward_batch = torch.cat(batch.reward)

    state_action_values2 = policy_net(x_p2=state_batch_p2,x_p3=state_batch_p3).gather(1, action_batch2)

    next_state_values2 = torch.zeros(BATCH_SIZE, device=device)
    

    t2=(target_net( x_p2=non_final_next_states_p2, x_p3=non_final_next_states_p3)).max(1)[0].detach()
    next_state_values2[non_final_mask_p2] = t2

    expected_state_action_values2 = (next_state_values2 * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss2 = criterion(state_action_values2, expected_state_action_values2.unsqueeze(1))
    #loss3 = criterion(state_action_values3, expected_state_action_values3.unsqueeze(1))
    
    
    optimizer2.zero_grad()
    loss2.backward()
    for param in paramgen(policy_net,p2=1):
        #print(param.grad.data)
        param.grad.data.clamp_(-1, 1)
    optimizer2.step()

    
    
#policy_net.load_state_dict(torch.load('./model.pth'))
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
    # Initialize the environment and state
    env.reset()
    #last_screen = get_screen()
    #current_screen = get_screen()
    #state = current_screen #- last_screen
    
    init_screen_p1, init_screen_p2, init_screen_p3 = get_screen()
    last_screen_p1, last_screen_p2, init_screen_p3 = get_screen()
    
    state_p1, state_p2, state_p3 = init_screen_p1, init_screen_p2, init_screen_p3 #- last_screen
    for t in count():
        # Select and perform an action
        #print(state)
        #print(state.shape)
        action1 = select_action(state_p1=state_p1, state_p3=state_p3)
        action2 = select_action(state_p2=state_p2, state_p3=state_p3)

        _, reward_, done, _ = env.step(action1.item(),action2.item())
        
        reward = torch.tensor([reward_], device=device)
        if reward_ > max_re:
            max_re = reward_
            print("best")
            print(max_re)
            

        # Observe new state
        last_screen_p1, last_screen_p2, last_screen_p3 =  init_screen_p1, init_screen_p2, init_screen_p3
        current_screen_p1, current_screen_p2, current_screen_p3 = get_screen()
        if not done:
            next_state_p1, next_state_p2, next_state_p3 =  current_screen_p1, current_screen_p2, current_screen_p3 #- last_screen      
        else:
            next_state_p1, next_state_p2, next_state_p3 = None, None, None
            

        # Store the transition in memory
        memory.push(state_p1, state_p2,state_p3, action1, action2, next_state_p1, next_state_p2, next_state_p3, reward)
        
        
        # Move to the next state
        state_p1, state_p2, state_p3 = next_state_p1, next_state_p2, next_state_p3

        # Perform one step of the optimization (on the policy network)
        print("opt")
        optimize_model()
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
            target_net.load_state_dict(policy_net.state_dict())
        

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


