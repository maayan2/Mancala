import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Parameters
input_size = 17 # state: board (14) + turn (1) + is_another (1) + action (1) = 17
layer1 = 64
# layer2 = 128
layer2 = 16
output_size = 1 # Q(state, action)
gamma = 0.90  


class DQN (nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        # self.linear3 = nn.Linear(layer2, layer3)
        self.output = nn.Linear(layer2, output_size)
        self.MSELoss = nn.MSELoss()

    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x)
        x = self.linear2(x)
        x = F.leaky_relu(x)
        # x = self.linear3(x)
        # x = F.leaky_relu(x)
        x = self.output(x)
        return x
    
    def loss (self, Q_value, rewards, Q_next_Values, Dones ):
        Q_new = rewards + gamma * Q_next_Values * (1- Dones)
        return self.MSELoss(Q_value, Q_new)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path), strict=False)

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self):
        return copy.deepcopy(self)

    def __call__(self, states, actions):
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action)