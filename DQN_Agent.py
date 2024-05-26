import torch
import random
import math
from DQN import DQN
from Constant import *
from State import *

class DQN_Agent:
    def __init__(self, player = 0, parametes_path = None, train = True, env= None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.player = player
        self.train = train
        self.setTrainMode()
        self.env = env

    def setTrainMode (self):
          if self.train:
              self.DQN.train()  
          else:
              self.DQN.eval()

    def get_action (self,events= None, state:State = None, epoch = 0, train = True, graphics = None) -> tuple:
        
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                actions = self.env.all_legal_action(state)
                if actions == -1: ### add in order to fix bug - rnd on actions(-1)
                    return -1
                else:
                    return random.choice(actions)
        
        state_tensor = state.toTensor()
        actions = self.env.all_legal_action(state)
        if actions == -1: ### add in order to fix bug - rnd on actions(-1)
            return -1
    
        action_tensor = torch.tensor(actions).reshape(-1, 1)
        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(action_tensor),1)) #מעתיק את מספר הSTATE  כמספר הACTIONS
        
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)
        max_index = torch.argmax(Q_values)
        action = actions[max_index]
        return action

    def get_Actions (self, states_tensor: State, dones) -> torch.tensor:
        actions = []
        for i, state_tensor in enumerate(states_tensor):
            if dones[i].item():
                actions.append(-1)
            else:
                state = State.tensorToState(state_tensor,player=self.player)
                actions.append(self.get_action(state=state, train=False))
        return torch.tensor(actions).reshape(-1,1)

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_action(state)