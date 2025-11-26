import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from lib.models.feedforwardNN import FFNN
import numpy as np

from torch.autograd import Variable
from IPython.core.debugger import set_trace

# https://github.com/patrickloeber/pytorch-examples/blob/master/rnn-name-classification/rnn.py
# https://cnvrg.io/pytorch-lstm/
# https://machinelearningmastery.com/how-to-avoid-exploding-gradients-in-neural-networks-with-gradient-clipping/

class RNN(nn.Module):
    """Define a recurrent NN with aribtrary network shape and activation function. The output layer is a linear."""
    def __init__(self, shape=[1, 10, 1], n_hidden_states = 10, activation_fun=nn.LeakyReLU, activation_fun_hidden=nn.Tanh):
        """Constructor
        Keyword arguments:
        shape -- shape of the network [input, hidden, output] 
        num_hidden_states -- number of hidden states in the network
        activation_function -- activation function of hidden layers
        """
        super().__init__()
        self.shape = shape
        self.afun = activation_fun 
        self.num_hidden = n_hidden_states
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        self.to(self.device)

        shape_hidden_FFNN = shape.copy()
        shape_hidden_FFNN[0]  += n_hidden_states
        shape_hidden_FFNN[-1] = n_hidden_states

        self.hidden_FFNN = FFNN(shape_hidden_FFNN, activation_fun, activation_fun_hidden)
        self.out_linear = nn.Linear(n_hidden_states, shape[-1])

               

    def forward(self, x:torch.tensor, h:torch.tensor) -> (torch.tensor, torch.tensor):
        # Forward propagate a single step
        input = torch.cat((x, h), 1) # Concatenate inputs x and hidden to input vector     
        hidden  = self.hidden_FFNN(input)
        out = self.out_linear(hidden)
        
        return out, hidden
       
    
    def predict(self, x:torch.tensor, h:torch.tensor,):
        
        N = x.size(dim=0)
        ouput = torch.zeros(N, self.shape[-1])
        for i in range(N):
            X = torch.reshape(x[i,:], (1, self.shape[0])) 
            out, h = self.forward(X, h)
            ouput[i, :] = out
        return ouput, h
    
    
        
    def init_hidden(self):
        return torch.zeros(1, self.num_hidden, requires_grad=False)
    
    #
    

    def trainTBPTT(self, X=torch.tensor, Y=torch.tensor, num_epochs=int(100),  optimizer=optim.Adam, loss_function=nn.MSELoss(), k1=int(5), max_grad_norm=int(1), reg_lambda=float(0)):
        """Truncated Backprobagation trough time training
        Keyword arguments:
        X -- input tensor with dimension [N_samples, features]
        Y -- output tensor with dimension [N_samples, features]
        num_epochs -- number of training epochs (has to be large for good result!!)
        k1 -- number of forward passes
        optimizer -- optimizer as torch module
        loss_function -- loss function as torch module
        max_grad_norm -- Maximum Norm of gradient for backward pass to avoid exploding gradients
        reg_lambda --
        """
        # Implementation reference
        # https://stackoverflow.com/questions/62901561/truncated-backpropagation-in-pytorch-code-check

        N_samples = X.size(dim=0)
        N = int((N_samples))
         
        train_loss = np.zeros(N)  # Loss per epoch
        epoch_loss = np.zeros(num_epochs) # Loss over epochs
        
        for n in range(num_epochs):
            
            hidden = self.init_hidden()
            total_loss = 0
            loss = 0.

            for t in range(0, N, k1):
                
                # Get chunk of training data points of size k1
                if t+k1 < N:
                    x = torch.reshape(X[t:t+k1, :], (k1, self.shape[0]))
                    y = torch.reshape(Y[t:t+k1, :], (k1 ,self.shape[-1]))   
                else:
                    x = torch.reshape(X[t:N, :], (N-t, self.shape[0]))
                    y = torch.reshape(Y[t:N, :], (N-t ,self.shape[-1]))   
                                  
                # k1 Forward passes
                pred, _ = self.predict(x, hidden)
                loss = loss_function(pred, y) 
                
                # update learing metrics
                train_loss[t] = loss
                total_loss += loss

                # Regularization
                if not reg_lambda==0.:
                    reg = torch.tensor(0.)
                    for name, param in self.named_parameters():
                        if 'bias' not in name:
                            reg += torch.norm(param, p=2)
                    loss += reg_lambda*reg

                # k1 backward passes                   
                loss.backward()
                with torch.no_grad():
                    # Avoid exploding gradients    
                    nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()  
                    
                    loss = 0. # Reset loss
                    _, hidden = self.predict(x, hidden) # Update hidden state
                    hidden = hidden.detach() # Detach hidden state to break gradient chains
            
            epoch_loss[n] = total_loss

        return hidden, epoch_loss, train_loss
    
    
    