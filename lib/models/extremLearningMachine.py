import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# References
# [1] Huang et.al Extreme learning machine: a new learning scheme of feedforward neural networks, 2004
# [2] Huang et.al On-line sequential extreme learning machine, 2005
# https://github.com/chickenbestlover/ELM-pytorch/tree/master

class ELM(nn.Module):
    """Define a single layer NN (SLNN) and random assigned input weights+bias"""
    def __init__(self, shape=[1, 10, 1], activation_fun=torch.relu):
        """Constructor
        Keyword arguments:
        shape -- shape of SLNN [input, hidden, output] 
        activation_fun -- activation function of hidden layers, passed as torch.ReLu()
        """
        super().__init__()
        self.shape = shape
        self.afun_hidden = activation_fun 
   
        self.f_in = nn.Linear(shape[0], shape[1]).requires_grad_(False)
        # Random init and fix parameters of input layer
        if 0:
            nn.init.normal_(self.f_in.weight, mean=0, std=1)
            nn.init.normal_(self.f_in.bias, mean=0, std=1)
        else:
            l = 1
            nn.init.uniform_(self.f_in.weight, a=-l, b=l)
            nn.init.uniform_(self.f_in.bias, a=-l, b=l)
              
        self.f_out = nn.Linear(shape[1], shape[2], bias=False) # No bias term as in [1]
            #self.f_out.weight.data = torch.zeros((shape[1], shape[2]))
        

    def forward(self, x : torch.tensor) -> torch.tensor:
        x = self.f_in(x)
        x = self.afun_hidden(x)
        x = self.f_out(x)
        return x
    
    def forward2hidden(self, x : torch.tensor) -> torch.tensor:
        x = self.f_in(x)
        x = self.afun_hidden(x)
        return x.data
    
    def trainLeastSquares(self, X: torch.tensor, Y: torch.tensor, jitter=1e-6):
        '''Minimizes the quadric error cost function to update the output weights via the pseudoinverse (scales with the number of neurons) '''
        self.train()
        with torch.no_grad():
            H = self.forward2hidden(X)
            H_T = torch.transpose(H, 0, 1)  # Transpose 2D-Matrix
            # Solve linear system via pseudoinverse
            weights = torch.linalg.solve(torch.matmul(H_T, H) + jitter* torch.eye(self.shape[1], self.shape[1]), torch.matmul(H_T, Y))

            # Update weights of layer f_out
            for name, param in self.named_parameters(): # https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/2
                if 'f_out' in name:
                    param.copy_(torch.transpose(weights, 0, 1)) 
                    param.requires_grad_(False)

                
    def trainRecursiveLeastSquares(self, X: torch.tensor, Y: torch.tensor, jitter=1e-6):
        '''Minimizes the quadric error cost function to update the output weights via the pseudoinverse (scales with the number of neurons) '''
        self.train()
        N, _ = X.size()
        M = self.shape[1] # number of neureon
        
        alpha = 1e4
        
        with torch.no_grad():
            # Init 
            P = alpha * torch.eye(M, M)
            weights = torch.zeros(M, 1)

            for t in range(N):
                h_T = self.forward2hidden(X[t,:]).reshape(1, M)
                h = torch.transpose(h_T, 0, 1) 

                k = P @ h / (1 + h_T @  P @ h)
                P -= k @ h_T @ P
                weights += k @ (Y[t,:] - h_T @ weights )


            # Update weights of layer f_out    
            for name, param in self.named_parameters():
                if 'f_out' in name:
                    param.copy_(torch.transpose(weights, 0, 1)) 
                    param.requires_grad_(False)








        
