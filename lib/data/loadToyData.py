import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def loadToyData(type='sin_1d', N=100, batch_size=10, shuffel=True):

    if type == 'sin_1d':
        x = torch.linspace(0, 4*torch.pi, N).reshape(N, 1)
        y = torch.sin(x) + 0.1 * torch.randn(N, 1)




    return x, y 


if 0:

    x, y = loadToyData(type='sin_1d', N=100, batch_size=10, shuffel=True)

    print(x.size())
    print(y.size())

