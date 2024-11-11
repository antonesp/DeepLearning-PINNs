import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.autograd as autograd
from torch.nn.parameter import Parameter


class PINN(nn.module):


    def LossPDE(self,X_train_Nu):

        