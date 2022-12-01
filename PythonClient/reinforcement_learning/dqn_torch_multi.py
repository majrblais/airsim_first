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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',('state_i','state_p',  'action', 'next_state_i', 'next_state_p', 'reward'))

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
    def __init__(self, ln, h, w, outputs):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(ln,32)
        self.lin2 = nn.Linear(32,64)
        self.lin3 = nn.Linear(64,128)
        self.lin4 = nn.Linear(128,128)
        
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.lin5 = nn.Linear(128,128)
        
        self.lin6 = nn.Linear(256,outputs)
        
        
    def forward(self, x_p, x_img):
        p=x_p
        img=x_img
        p = p.to(device).float()
        p = self.lin1(p)
        p = self.lin2(p)
        p = self.lin3(p)
        p = self.lin4(p)
        
        img = img.to(device)
        img = F.relu(self.bn1(self.conv1(img)))
        img = F.relu(self.bn2(self.conv2(img)))
        img = F.relu(self.bn3(self.conv3(img)))
        #img = torch.flatten(img)
        img = img.view(img.size(0), -1)
        img = F.relu((self.lin5(img)))
         
        combined = torch.cat((p,img),1)
        
        
        out = combined.view(combined.size(0), -1)
        out = self.lin6(out)
        
        return out
        
resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.CUBIC),T.ToTensor()])


from airgym.envs import custom_env_multi
env = custom_env_multi.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=0.5, image_shape=(128, 128, 1),)


def get_screen():
    screen, img = env._get_obs()
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) #/ 255
    img = torch.from_numpy(img)
    
    
    x=screen.x_val
    y=screen.y_val
    z=screen.z_val
    #screen = np.array([float(x),float(y),float(z)])
    pos = np.array([float(x),float(y)])
    #screen = torch.from_numpy(np.expand_dims(screen, axis=0))
    pos = torch.from_numpy(pos)
    
    
    return pos.unsqueeze(0),resize(img).unsqueeze(0)
    
#print(get_screen())
#env.reset()
#plt.figure()
#plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),interpolation='none')
#plt.title('Example extracted screen')
#plt.show()


BATCH_SIZE = 2
GAMMA = 0.997
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 100


init_screen_p, init_screen_img = get_screen()


_, _, screen_height, screen_width = init_screen_img.shape

n_actions = env.action_space.n
#print(n_actions)

policy_net = DQN(len(init_screen_p[0]), screen_height, screen_width, n_actions).to(device)
print(policy_net)
target_net = DQN(len(init_screen_p[0]), screen_height, screen_width, n_actions).to(device)



target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(2000)
steps_done = 0


def select_action(state_p, state_img):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #if sample > eps_threshold:
    if True:
        with torch.no_grad():
            act=policy_net(state_p, state_img).max(1)[1].view(1, 1)
            return act
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        
episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask_i = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_i)), device=device, dtype=torch.bool)
    non_final_mask_p = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p)), device=device, dtype=torch.bool)
   
    non_final_next_states_i = torch.cat([s for s in batch.next_state_i if s is not None])
    non_final_next_states_p = torch.cat([s for s in batch.next_state_p if s is not None])


    
    #non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch_i = torch.cat(batch.state_i)
    state_batch_p = torch.cat(batch.state_p)
   
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    print(state_batch_p)
    state_action_values = policy_net(state_batch_p,state_batch_i ).gather(1, action_batch)



    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask_i] = target_net(non_final_next_states_p,non_final_next_states_i).max(1)[0].detach()
    
    print(target_net(non_final_next_states_p,non_final_next_states_i))
    print(target_net(non_final_next_states_p,non_final_next_states_i).max(1)[0].detach())
    print(next_state_values)
    print(next_state_values[non_final_mask_i])
    exit()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    
    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
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
    max_re = -10
    # Initialize the environment and state
    env.reset()
    #last_screen = get_screen()
    #current_screen = get_screen()
    init_screen_p, init_screen_img = get_screen()

    
    state_p, state_img = init_screen_p, init_screen_img #- last_screen
    for t in count():
        # Select and perform an action

        action = select_action(state_p, state_img)
        _, reward_, done, _ = env.step(action.item())

        
        reward = torch.tensor([reward_], device=device)

        if reward_ > max_re:
            max_re = reward_

            

        # Observe new state
        last_screen_p, last_screen_img = init_screen_p, init_screen_img
        current_screen_p, current_screen_i = get_screen()
        if not done:
            next_state_p, next_state_i  = current_screen_p, current_screen_i 
        else:
            next_state_i, next_state_p = None, None
        
        # Store the transition in memory
        
        memory.push(state_img, state_p, action, next_state_i, next_state_p, reward)
        
        
        # Move to the next state
        state_img, state_p = next_state_i, next_state_p

        # Perform one step of the optimization (on the policy network)
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


