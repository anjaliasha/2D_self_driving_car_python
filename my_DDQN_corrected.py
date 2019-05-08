# -*- coding: utf-8 -*-
"""
Created on Wed May  8 17:47:15 2019

@author: JCMat
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 16:50:17 2019

@author: JCMat
"""

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


#first step creating neural network for processing the q values

class NeuralNetwork1(nn.Module):
    def __init__(self, input_size, nb_action):
        super(NeuralNetwork1, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.f1 = nn.Linear(input_size, 30)
        self.f2 = nn.Linear(30, nb_action)
        
    #feed forward network
    def forward(self, state):
        x = F.relu(self.f1(state))
        Q_values = self.f2(x)
        return Q_values
class NeuralNetwork2(nn.Module):
    def __init__(self, input_size, nb_action):
        super(NeuralNetwork2, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.f1 = nn.Linear(input_size, 30)
        self.f2 = nn.Linear(30, nb_action)
        
    #feed forward network
    def forward(self, state):
        x = F.relu(self.f1(state))
        Q_values = self.f2(x)
        return Q_values
       
    
#experience replay to consider previous actions
class ExperienceReplayMemory(object):
    #initialising capacity and creating memory
    def __init__(self, capacity):
        self.c = capacity
        self.memory = []
        
    def push_action(self, event):
        self.memory.append(event)
        if len(self.memory)>self.c:
            del self.memory[0]
    #obtaining a sample of elements from the memory for experience replay        
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
#implementing double q learning
        
class DoubleQNetwork():
    
    def __init__(self, input_size, nb_action, gamma):
        self.i = 0
        self.gamma = gamma
        self.reward_window = []
        self.model1 = NeuralNetwork1(input_size, nb_action)
        self.model2 = NeuralNetwork2(input_size, nb_action)
        self.memory = ExperienceReplayMemory(100000)
        self.optimizer1 = optim.Adam(self.model1.parameters(), lr = 0.001)
        self.optimizer2 = optim.Adam(self.model2.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    #network A is used to select an action ran
    def select_action(self, state):
        self.i+=1 #to itteratively choose the network
        if self.i%2 ==0:
            
            probs = F.softmax(self.model1(Variable(state, volatile = True))*100) # T=100
            action = probs.multinomial()
        else :
            probs = F.softmax(self.model2(Variable(state, volatile = True))*100) # T=100
            action = probs.multinomial()
            
        return action.data[0,0]
    
    #network1 is used to select the best action and network 2 is updated with the new q values
    #in the algorithm else if network B is udated obtain action from network A and useit to update network B
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        if self.i%2 ==0 :
            outputs = self.model2(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            next_outputs = self.model2(batch_next_state).detach().max(1)[0]
            target = self.gamma*next_outputs + batch_reward
            td_loss = F.smooth_l1_loss(outputs, target)
            self.optimizer2.zero_grad()
            td_loss.backward(retain_variables = True)
            self.optimizer2.step()
        else :
            outputs = self.model1(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
            next_outputs = self.model1(batch_next_state).detach().max(1)[0]
            target = self.gamma*next_outputs + batch_reward
            td_loss = F.smooth_l1_loss(outputs, target)
            self.optimizer1.zero_grad()
            td_loss.backward(retain_variables = True)
            self.optimizer1.step()
        
       
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push_action((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model1.state_dict(),
                    'optimizer' : self.optimizer1.state_dict(),
                   }, 'new_brain.pth')
    
    def load(self):
        if os.path.isfile('new_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('new_brain.pth')
            self.model1.load_state_dict(checkpoint['state_dict'])
            self.optimizer1.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
    
    
    
    
    
    
    
    
    
        