import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.autograd as autograd
from torch.nn.parameter import Parameter

def grad(function,var):
    return autograd.grad(function, var, grad_outputs=torch.ones_like(function), create_graph=True)[0]

# Parameters for equation
a = 2

class PINN(nn.module):

    def __init__(self,num_features,num_hidden,num_output, learning_rate):

        self.loss_function = nn.MSELoss(reduction ='mean')
        self.hidden = nn.Linear(num_features, num_hidden)
        self.out = nn.Linear(num_hidden, num_output)

        # training

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)

    # We are using a simple exponential equation to test the PINN
    def LossPDE(self,t):

        u = self.forward(t)
        u_t = grad(u,t)

        eq1 = u_t - a*u

        loss_PDE = self.loss_function(eq1,torch.zeros_like(eq1))
        
    
    def closure(self):
        
        
        loss = self.LossPDE()
        
        loss.backward()
                
        self.iter += 1
        
        if self.iter % 100 == 0:

            error_vec, _ = PINN.test()
        
            print(loss,error_vec)

        return loss
    # self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

decay_rate = 10**(-4)
#decay_rate = 0
learning_rate =2*10**(-4)
#learning_rate = 0.005
optimizer = optim.Adam(net.parameters(), lr=learning_rate,weight_decay= decay_rate)
momentum_rate = 0.5
#optimizer = optim.SGD(net.parameters(), lr=0.005,weight_decay= decay_rate,momentum=momentum_rate)
criterion = nn.CrossEntropyLoss()