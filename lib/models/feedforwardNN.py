import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np

import torch.autograd.functional as F

from lib.models.stepEKF import stepEKF as EKF


# https://discuss.pytorch.org/t/when-should-i-use-nn-modulelist-and-when-should-i-use-nn-sequential/5463
# https://discuss.pytorch.org/t/how-to-create-mlp-model-with-arbitrary-number-of-hidden-layers/13124/6

class FFNN(nn.Module):
    """Define a FFNN with aribtrary network shape and activation function. The output layer is a linear."""
    def __init__(self, shape=[1, 10, 1], activation_fun_hidden=nn.LeakyReLU, activation_fun_out=nn.Linear, bias=True ):
        """Constructor
        Keyword arguments:
        shape -- shape of network [input, hidden, output] 
        activation_fun_hidden -- activation function of hidden layers, passed as nn.ReLu()
        activation_fun_out -- activation function of output layer,  passed as nn.ReLu(). For linear layer nn.Linear
        bias -- Set to True to add bias to each linear layer
        """
        super().__init__()
        self.shape = shape
        self.afun_hidden = activation_fun_hidden 
        self.afun_out = activation_fun_out 
               
        self.layers = nn.ModuleList()
                    
        for i in range(len(self.shape)-2): 
            self.layers.append(nn.Linear(shape[i], shape[i+1], bias=bias))  
            if not activation_fun_hidden == nn.Linear:
                self.layers.append(activation_fun_hidden)
        self.layers.append(nn.Linear(shape[-2], shape[-1], bias=bias)) # Linear output layer 
        
        if not activation_fun_out == nn.Linear:
            self.layers.append(activation_fun_out)


    def forward(self, x : torch.tensor) -> torch.tensor:
        for layer in self.layers:
            x = layer(x)
        return x
    
    def printParams(self):
        for n,p in self.named_parameters():
            print(n)
            print(p)

    def jacobian_wrt_params(self, x):
        ''' Computes the Jacobian Matrix of nn.Module wrt to all parameter at input x
            J = [df_i(par) / dpar_j ]_ij |_x  ... n_out x n_par Matrix
            Ref: 
            https://github.com/pytorch/pytorch/issues/49171#issuecomment-933814662
        '''
        n_par = sum(p.numel() for p in self.parameters())
        n_out = self.shape[-1]

        names = list(n for n, _ in self.named_parameters())
        fun = lambda *params: torch.func.functional_call(self, {n: p for n, p in zip(names, params)}, x)
        # Compute jacobians wrt to parameters
        jacobian_list = F.jacobian(fun , tuple(self.parameters()))

        newlist = list()
        for jac_part in jacobian_list:
            # Resahpe 3D Tensor to 2D
            if len(jac_part.size()) == 3:
                jac_part = torch.flatten(jac_part, start_dim=1)
            newlist.append(jac_part)
        jacobian = torch.cat(newlist, 1) # Concatenate 

        assert jacobian.shape == (n_out, n_par)

        if 0:
            self.printParams()
            print('Jacobian List')
            print(jacobian_list)
            print('Jacobian Matrix')
            print(jacobian)

        return jacobian

    def trainEpochs(self, train_loader=DataLoader, num_epochs=100, optimizer=optim.Adam, loss_function=nn.MSELoss(), test_loader=DataLoader):
        
        train_loss = np.zeros(num_epochs)
        test_loss = np.zeros(num_epochs)
        
        for X, y in test_loader:
            X_test = X
            Y_test = y   
        
        for epoch in range(num_epochs):
            # Train set
            for X, y in train_loader:
                optimizer.zero_grad()
                pred = self.forward(X)
                loss = loss_function(pred, y)
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss[epoch] = loss
            test_loss[epoch] = loss_function(self.forward(X_test), Y_test)

        return train_loss, test_loss

    def trainEpochsRegularized(self, train_loader=DataLoader, num_epochs=100,  optimizer=optim.Adam, loss_function=nn.MSELoss(), test_loader=DataLoader, regNorm=1, regPar=1e-3):
        
        train_loss = np.zeros(num_epochs)
        test_loss = np.zeros(num_epochs)
        
        for X, y in test_loader:
            X_test = X
            Y_test = y   
        
        for epoch in range(num_epochs):
        # Train set
            for X, y in train_loader:
                optimizer.zero_grad()
                pred = self.forward(X)
                loss = loss_function(pred, y)
                
                reg = torch.tensor(0.)
                for name, param in self.named_parameters():
                    if 'bias' not in name:
                        reg += torch.norm(param, p=regNorm)
                loss += regPar*reg
                
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            train_loss[epoch] = loss
            test_loss[epoch] = loss_function(self.forward(X_test), Y_test)

        return train_loss, test_loss
    
    


    def trainEKF(self, X, Y, num_epochs=100, p=1e4, q=1e-4, r=1e-1):
        """ Trains the NN using the Extended Kalman Filter (EKF)
            References: Haykin Simon, "Kalman filtering and neural networks", 2004 -> Section 2.4

            Keyword arguments:
            p -- Error Covariance scale P = p * eye
            q -- Process Covariance scale Q = q * eye
            r -- Measurement Covariance scale R = r * eye

            Nonlinear state-space model 
            W[k+1] = W[k] + w[k]
            y[k]   = h(w[k], x[k]) + v[k]

            h ... Neural network
            W ... NN parameters
            x ... inputs
            y ... observations
            w ... zero mean Gaussian process noise
            v ... zero mean Gaussian measuremnt noise            
        """
        self.train()
        N, _ = X.size()
        n_out = self.shape[-1]
        train_loss = np.zeros(num_epochs)
        test_loss = np.zeros(num_epochs)

        # Pack model parameters to vector
        W = nn.utils.convert_parameters.parameters_to_vector(self.parameters())
        N_W = W.size()[0]
        W = W.reshape(N_W, 1)

        # Initialize  estiation Error Covariance 
        P = p * torch.eye(N_W, N_W)
        #W = torch.zeros(N_W, 1, requires_grad=False)

        # Measurement Covariance Matrix
        R = r * torch.eye(n_out, n_out)
        # Process Covariance Matrix
        Q = q * torch.eye(N_W, N_W)

       
        
        for epoch in range(num_epochs):                              
            for t in range(N):
                
                with torch.no_grad():
                    x = X[t, :] 
                    y = Y[t, :]
                    
                    # Predict
                    y_pred = self(x)
                    e = y - y_pred
                    e = e.reshape(n_out, 1)

                    # Liniearize NN wrt params at x 
                    self.zero_grad()
                    H = self.jacobian_wrt_params(x)
                    
                    W, P = EKF(W, P, H, e, Q, R)


                    # Unpack vector to model parameters
                    nn.utils.convert_parameters.vector_to_parameters(W, self.parameters())
                  




        #return train_loss, test_loss