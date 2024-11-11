import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.autograd as autograd
from torch.nn.parameter import Parameter


class PINN(nn.module):

    def __init__(self,num_features,num_hidden,num_output, learning_rate):

        self.hidden = nn.Linear(num_features, num_hidden)
        self.out = nn.Linear(num_hidden, num_output)
        # training
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = self.hidden(x)
        x = F.relu(x)
        x = self.out(x)
        return F.softmax(x, dim=1)

    def LossPDE(self,X_train_Nu):
        

