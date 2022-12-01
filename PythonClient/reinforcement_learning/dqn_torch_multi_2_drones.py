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
Transition = namedtuple('Transition',('state_i1','state_p1','state_i2','state_p2',  'action1', 'action2', 'next_state_i1', 'next_state_p1','next_state_i2', 'next_state_p2', 'reward'))

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
    def __init__(self, ln1, ln2, h1, w1, h2, w2, outputs):
        super(DQN, self).__init__()
        self.lin1a = nn.Linear(ln1,32)
        self.lin2a = nn.Linear(32,64)
        self.lin3a = nn.Linear(64,128)

        
        self.lin1b = nn.Linear(ln2,32)
        self.lin2b = nn.Linear(32,64)
        self.lin3b = nn.Linear(64,128)


        self.lin1c = nn.Linear(256,128)
        
        
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
        
        
        self.lin2c = nn.Linear(384,128)
        self.lin3c = nn.Linear(128,256)
        
        
        self.lin4c = nn.Linear(256,128)
        
        self.lin4d = nn.Linear(128,64)
        self.lin5d = nn.Linear(64,4)
        
        self.lin4e = nn.Linear(128,64)
        self.lin5e = nn.Linear(64,4)
        
        
    def forward(self, x_p1, x_img1, x_p2, x_img2):
        p1=x_p1
        p2=x_p2
        
        img1=x_img1
        img2=x_img2
        
        
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
        outp = self.lin1c(outp)
        
        
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
        
        
        combined = torch.cat((combinedimg,outp),1)
        out = combined.view(combined.size(0), -1)
        out = self.lin2c(out)
        out = self.lin3c(out)
        out = self.lin4c(out)
        
        
        out1 = self.lin4d(out)
        out1 = self.lin5d(out1)
        

        out2 = self.lin4e(out)
        out2 = self.lin5e(out2)
        
        
        return out1,out2
        
resize = T.Compose([T.ToPILImage(),T.Resize(40, interpolation=Image.CUBIC),T.ToTensor()])


from airgym.envs import custom_env_multi_2_drones
env = custom_env_multi_2_drones.AirSimcustomEnv_base(ip_address="127.0.0.1",step_length=0.5, image_shape=(128, 128, 1),)


def get_screen():
    screen1,screen2, img1, img2 = env._get_obs()

    img1 = img1.transpose((2, 0, 1))
    img1 = np.ascontiguousarray(img1, dtype=np.float32) #/ 255
    img1 = torch.from_numpy(img1)

    img2 = img2.transpose((2, 0, 1))
    img2 = np.ascontiguousarray(img2, dtype=np.float32) #/ 255
    img2 = torch.from_numpy(img2)
    
    
    x1=screen1.x_val
    y1=screen1.y_val
    z1=screen1.z_val

    x2=screen2.x_val
    y2=screen2.y_val
    z=screen2.z_val
    #screen = np.array([float(x),float(y),float(z)])
    pos1 = np.array([float(x1),float(y1)])
    pos2 = np.array([float(x2),float(y2)])
    #screen = torch.from_numpy(np.expand_dims(screen, axis=0))
    pos1 = torch.from_numpy(pos1)
    pos2 = torch.from_numpy(pos2)
    
    
    return pos1.unsqueeze(0), pos2.unsqueeze(0) , resize(img1).unsqueeze(0), resize(img2).unsqueeze(0)
    
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


init_screen_p1, init_screen_p2, init_screen_img1, init_screen_img2 = get_screen()


_, _, screen_height1, screen_width1 = init_screen_img1.shape

_, _, screen_height2, screen_width2 = init_screen_img2.shape


n_actions = env.action_space.n
#print(n_actions)

policy_net = DQN(len(init_screen_p1[0]), len(init_screen_p1[0]), screen_height1, screen_width1, screen_height2, screen_width2, n_actions).to(device)
print(policy_net)
target_net = DQN(len(init_screen_p1[0]), len(init_screen_p2[0]), screen_height1, screen_width1, screen_height2, screen_width2, n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(2000)
steps_done = 0


def select_action(state_p1, state_img1, state_p2, state_img2):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    #if sample > eps_threshold:
    if True:
        with torch.no_grad():
            act=policy_net(state_p1, state_img1, state_p2, state_img2)
            #print(act[0].max(1)[1].view(1, 1))
            #print(act[1].max(1)[1].view(1, 1))
            
            act1=act[0].max(1)[1].view(1, 1)
            act2=act[1].max(1)[1].view(1, 1)
            return act1,act2
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
    non_final_mask_i1 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_i1)), device=device, dtype=torch.bool)
    non_final_mask_p1 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p1)), device=device, dtype=torch.bool)
    non_final_mask_i2 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_i2)), device=device, dtype=torch.bool)
    non_final_mask_p2 = torch.tensor(tuple(map(lambda s: s is not None,batch.next_state_p2)), device=device, dtype=torch.bool)
   
    non_final_next_states_i1 = torch.cat([s for s in batch.next_state_i1 if s is not None])
    non_final_next_states_p1 = torch.cat([s for s in batch.next_state_p1 if s is not None])
    non_final_next_states_i2 = torch.cat([s for s in batch.next_state_i2 if s is not None])
    non_final_next_states_p2 = torch.cat([s for s in batch.next_state_p2 if s is not None])
    
    
                                                                       
    state_batch_i1 = torch.cat(batch.state_i1)
    state_batch_p1 = torch.cat(batch.state_p1)
    state_batch_i2 = torch.cat(batch.state_i2)
    state_batch_p2 = torch.cat(batch.state_p2)

    action_batch1 = torch.cat(batch.action1)
    action_batch2 = torch.cat(batch.action2)

    reward_batch = torch.cat(batch.reward)

    

    
    #state_batch = torch.cat(batch.state)
    #action_batch = torch.cat(batch.action)
    #reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values1 = policy_net(state_batch_p1,state_batch_i1, state_batch_p2,state_batch_i2)[0].gather(1, action_batch1)
    state_action_values2 = policy_net(state_batch_p1,state_batch_i1, state_batch_p2,state_batch_i2)[1].gather(1, action_batch2)
    
    
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values1 = torch.zeros(BATCH_SIZE, device=device)
    next_state_values2 = torch.zeros(BATCH_SIZE, device=device)
    
    #next_state_values1[non_final_mask_i1] = (target_net(state_batch_p1,state_batch_i1, state_batch_p2,state_batch_i2)[0]).max(1)[0].detach()
    t1=(target_net(non_final_next_states_p1,non_final_next_states_i1, non_final_next_states_p2,non_final_next_states_i2)[0]).max(1)[0].detach()
    next_state_values1[non_final_mask_i1] = t1

    t2=(target_net(non_final_next_states_p1,non_final_next_states_i1, non_final_next_states_p2,non_final_next_states_i2)[1]).max(1)[0].detach()
    next_state_values2[non_final_mask_i2] = t2

    
    # Compute the expected Q values
    expected_state_action_values1 = (next_state_values1 * GAMMA) + reward_batch
    expected_state_action_values2 = (next_state_values2 * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss1 = criterion(state_action_values1, expected_state_action_values1.unsqueeze(1))
    loss2 = criterion(state_action_values2, expected_state_action_values2.unsqueeze(1))
    loss = loss1+loss2
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
    #state = current_screen #- last_screen
    
    init_screen_p1, init_screen_p2, init_screen_img1, init_screen_img2 = get_screen()
    last_screen_p1, last_screen_p2, last_screen_img1, last_screen_img2 = get_screen()
    
    state_p1, state_img1, state_p2, state_img2 = init_screen_p1, init_screen_img1, init_screen_p2, init_screen_img2 #- last_screen
    for t in count():
        # Select and perform an action
        #print(state)
        #print(state.shape)
        action1, action2 = select_action(state_p1, state_img1, state_p2, state_img2)
        

        _, reward_, done, _ = env.step(action1.item(),action2.item())
        
        reward = torch.tensor([reward_], device=device)
        if reward_ > max_re:
            max_re = reward_
            print("best")
            print(max_re)
            

        # Observe new state
        last_screen_p1, last_screen_p2, last_screen_img1, last_screen_img2 =  init_screen_p1, init_screen_p2, init_screen_img1, init_screen_img2
        current_screen_p1, current_screen_p2, current_screen_i1, current_screen_i2 = get_screen()
        if not done:
            next_state_p1, next_state_p2, next_state_i1, next_state_i2 =  current_screen_p1, current_screen_p2, current_screen_i1, current_screen_i2 #- last_screen      
        else:
            next_state_p1, next_state_p2, next_state_i1, next_state_i2 = None, None, None, None
            

        # Store the transition in memory
        memory.push(state_img1, state_p1, state_img2, state_p2, action1, action2, next_state_i1, next_state_p1, next_state_i2, next_state_p2, reward)
        
        
        # Move to the next state
        state_p1, state_img1, state_p2, state_img2 = next_state_p1, next_state_i1, next_state_p2, next_state_i2

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


