import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from lib.models.feedforwardNN import FFNN
import numpy as np


class AENN(nn.Module):
    """Define a deep autoencoder with aribtrary encode/decoder shape and activation function. The output layer is a linear."""
    def __init__(self, shape_encoder=[1, 10, 1],  activation_fun_encoder=nn.LeakyReLU, shape_decoder=[1, 10, 1],  activation_fun_decoder=nn.LeakyReLU, activation_fun_out=nn.Linear):
        """Constructor
        Keyword arguments:
        shape_encoder -- shape of enocder network [input, hidden, output] 
        activation_fun_encoder -- activation function of encoder hidden layers
        shape_decoder -- shape of enocder network [input, hidden, output] 
        activation_fun_decoder -- activation function of encoder hidden layers
        activation_fun_out -- activation function of output layer
        """
        super().__init__()
        self.shape_enc = shape_encoder
        self.afun_enc = activation_fun_encoder 
        self.shape_dec = shape_decoder
        self.afun_dec = activation_fun_decoder 
        self.afun_out = activation_fun_out
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)

        self.encoder = FFNN(self.shape_enc, self.afun_enc, self.afun_out)
        self.decoder = FFNN(self.shape_dec, self.afun_dec, self.afun_out)

    def forward(self, x : torch.tensor) -> torch.tensor:
        y = self.encoder(x)
        z = self.decoder(y)
        return z
    

    def trainEpochs(self, train_loader=DataLoader, num_epochs=100, optimizer=optim.Adam, loss_function=nn.MSELoss(), test_loader=DataLoader):
        
        train_loss = np.zeros(num_epochs)
        test_loss = np.zeros(num_epochs)
        
        for X, y in test_loader:
            X_test = X        
                

        for epoch in range(num_epochs):
            # Train set
            for X, y in train_loader:
                optimizer.zero_grad()
                pred = self.forward(X)
                loss = loss_function(pred, X)
                #optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_loss[epoch] = loss
            test_loss[epoch] = loss_function(self.forward(X_test), X_test)
        
        return train_loss, test_loss




