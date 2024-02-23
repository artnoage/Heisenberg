import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from utils import *
import time


y=torch.tensor([-0.003,-0.002,-0.001,4,6,-5,100,0])
part2  = torch.where(y == 0, torch.tensor(1.0), (2 * y)/torch.sinh(2 * y))

print(part2)