
import numpy as np
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class board_state(object):

    def __init__(self):
        b = []
        for i in range(6):
            b.append([0]*7)
       # b.append([' ' + str(i) for i in range(1,8)])
        self.current_state = np.array(b) 
        self.turn = 0
        self.whos_turn = 1 
        
            
    def vect(self):
        v = np.array([0 for i in range(6*7)])
        for i in range(6):
            v[7*i:7*(i+1)] = self.current_state[i] 
#        v[6*7] = self.whos_turn
        return v 
        
    def reset(self):
        b = []
        for i in range(6):
            b.append([0]*7)
        self.current_state = np.array(b) 
        self.turn =0 
        self.whos_turn =1

    def print_board(self):
        for row in self.current_state:
            print(''.join(row))


    def play(self,n):
        l= []
        for i in range(6):
            if self.current_state[i,n] ==0:
                l.append(i)
        if l ==[]:
            return False
        if self.whos_turn ==1:
            self.current_state[max(l)][n] = 1 
            self.whos_turn = -1 
            self.turn +=1
        else:
            self.current_state[max(l)][n] = -1 
            self.whos_turn =1
            self.turn += 1 
        return True
    
    def not_full(self):
        nf =[]
        for i in range(7):
            l = []
            for j in range(6):
                if self.current_state[j][i] == 0:
                    l.append(j)
            if l!=[]:
                nf.append(i)
        return nf
    
    def check_win(self,p):
        if p ==1:
            s = 1 
        else:
            s = -1 

        #check horizontal
        def check_h(self,s):
            for i in range(6):
                for j in range(4):
                    c = 0
                    for k in range(4):
                        if self.current_state[i][j+k]== s:
                            c +=1
                    if c==4:
                        return True
            return False

        #check vertical
        def check_v(self, s):
            for i in range(3):
                for j in range(7):
                    c = 0
                    for k in range(4):
                        if self.current_state[i+k][j] == s:
                            c += 1
                        if c ==4:
                            return True
            return False

        #check diagonal
        def check_d(self,s):
            for i in range(3):
                for j in range(4):
                    c = 0
                    for k in range(4):
                        if self.current_state[i+k][j+k] == s:
                            c +=1 
                        if c ==4 :
                            return True
            for i in range(3):
                for j in range(3,7):
                    c = 0
                    for k in range(4):
                        if self.current_state[i+k][j-k] == s:
                            c +=1
                        if c ==4:
                            return True
            return False
            #combind   
        if check_h(self,s) or check_v(self,s) or check_d(self,s):
            return True
        return False

    def step(self,n):
        played = self.play(n)
        end = False
        if self.check_win(1) or self.check_win(2) or self.turn==42:
                end = True
        return [self.vect(), played, end]
        
    

    
   
        


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256,128)
        self.layer4 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
device = torch.device("cpu")

# Get number of actions from gym action space
n_actions = 7 
# Get the number of state observations
board = board_state()
board.reset()
state = board.vect() 
n_observations = len(state) 

policy_net_1 = DQN(n_observations, n_actions).to(device)
policy_net_2 = DQN(n_observations, n_actions).to(device)
policy_net_1.load_state_dict(torch.load("trained_1.pth"))
policy_net_1.load_state_dict(torch.load("trained_2.pth"))

steps_done = 0
def select_action(state,p, learn=True):
    global steps_done
    sample = random.random()
    eps_threshold =  EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if not learn:
        eps_threshold = 0
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            if p==1:
                action_values = policy_net_1(state)[0,:]
                #    return policy_net_1(state).max(1).indices.view(1,1) 
            elif p==-1:
                action_values = policy_net_2(state)[0,:]
            available_action_values = [action_values[i] for i in board.not_full()]
            return torch.tensor([[np.argmax(available_action_values)]], device=device, dtype=torch.long)

            #    return policy_net_2(state).max(1).indices.view(1,1)
    else:
        r = random.randint(0,6)
        return torch.tensor([[r]], device=device, dtype=torch.long)

player = int(input(print("Do you want to play player 1 or 2? Enter either 1 or 2.")))

turns = 0
print(board.current_state)
while not board.check_win(1) and not board.check_win(2):
    state = torch.tensor(board.vect(), dtype=torch.float32, device = device).unsqueeze(0)
    played =False
    if player ==1:
        while not played:
          action = int(input(print("Enter a number from 1 to 7."))) -1
          played = board.play(action)
    else:
        action = select_action(state,1)
        played = board.step(action.item())[1]
    print(board.current_state)
    if board.check_win(1):
        print("Player 1 won!")
        break
    state = torch.tensor(board.vect(), dtype=torch.float32, device = device).unsqueeze(0)
    played = False
    if player ==2:
        while not played:
            action = int(input(print("Enter a number from 1 to 7.")))-1
            played = board.step(action)[1]
    else:
        action = select_action(state,-1)
        played = board.step(action.item())[1]
    print(board.current_state)
    if board.check_win(2):
        print("Player 2 won!")
        break
    turns +=1
    if turns == 6*7:
        print("There are no more possible turns!")
        break 
