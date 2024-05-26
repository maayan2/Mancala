import torch
from mancala import *
import numpy as np

checkpoint = torch.load("Data/checkpoint1111.pth")
print(checkpoint['best_random2'])
print(checkpoint['best_random_params2'])
print(checkpoint['best_random1'])
print(checkpoint['best_random_params1'])

