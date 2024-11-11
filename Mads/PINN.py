import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.autograd as autograd
from torch.nn.parameter import Parameter


class PINN(nn.Module):
    
    def __init__(self, num_features, num_hidden, num_output):
        super(PINN, self).__init__()
        
        
        
        # Define activation function
        self.activation = nn.ReLU()
        
    def forward(self, x):
        
        return x


    def MVP(self, x, u, d, pf):
        """
        Input:
            x: Is the state tensor 
        """
        
        
        # Meal system
        D_1 = Parameter()
        D_2 = Parameter()
        
        # Insulin system
        I_sc = Parameter()
        I_p = Parameter()
        I_eff = Parameter()
        
        # Glucose system        
        G = Parameter()
        G_sc = Parameter()
        
        # Our unknown patient parameter
        S_I = Parameter() # [(dL/mU)/min]
        
        # Known patient paramter
        tau_1 = pf[0]49      # [min]
        tau_2 = 47      # [min]
        C_I = 20.1      # [dL/min]
        p_2 = 0.0106    # [min^(-1)]
        GEZI = 0.0022   # [min^(-1)]
        EGP_0 = 1.33    # [(mg/dL)/min]
        V_G = 253       # [dL]
        tau_m = 47      # [min]
        tau_sc = 5      # [min]
        
        # Define gradients needed
        D_1_t = D_1.grad
        D_2_t = D_2.grad
        I_sc_t = I_sc.grad
        I_p_t = I_p.grad
        I_eff_t = I_eff.grad
        G_t = G.grad
        G_sc_t = G_sc.grad
        
        Gradients = torch.tensor([D_1_t, D_2_t, I_sc_t, I_p_t, I_eff_t, G_t, G_sc_t])
        
        # Define our ODEs
        Meal_1 = D_1_t - d + (D_1 / tau_m)
        Meal_2 = D_2_t - (D_1 / tau_m) + (D_2 / tau_m)
        
        Insulin1 = I_sc_t - (u / (tau_1 * C_I)) - (I_sc / tau_1)
        Insulin2 = (I_sc - I_p) / tau_2
        Insulin3 = -p_2 * I_eff + p_2 * S_I * I_p
        
        Glucose1 = -(GEZI + I_eff) * G + EGP_0 + ((1000 * D_2) / (V_G * tau_m))
        Glucose2 = (G - G_sc) / tau_sc
        
        ODE = torch.tensor([Meal_1, Meal_2, Insulin1, Insulin2, Insulin3, Glucose1, Glucose2])
        
        return Gradients, ODE