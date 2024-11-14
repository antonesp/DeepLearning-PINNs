import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from Load_data2 import custom_csv_parser



def grad(func, var):
    '''
    Computes the gradient of a function with respect to a variable.
    Written by Engsig-Karup, Allan P. (07/05/2024).

    Args:
    func (torch.Tensor): Function to differentiate
    var (torch.Tensor): Variable to differentiate with respect to

    Returns:
    torch.Tensor: Gradient of func with respect to var
    '''
    return torch.autograd.grad(func, var, grad_outputs=torch.ones_like(func), create_graph=True, retain_graph=True)[0]    

class PINN(nn.Module):
    
    def __init__(self): # num_features, num_hidden, num_output
        super(PINN, self).__init__()
        
        self.input = nn.Linear(in_features=1,  out_features=128)
        self.hidden = nn.Linear(in_features=128, out_features=128)
        self.output = nn.Linear(in_features=128, out_features=7)
        
        # Define activation function
        self.activation = nn.SiLU()

        # Define patient parameters
        self.tau_1 = torch.tensor([49.0], requires_grad=True, device=device)       # [min]
        self.tau_2 = torch.tensor([47.0], requires_grad=True, device=device)       # [min]
        self.C_I = torch.tensor([20.1], requires_grad=True, device=device)         # [dL/min]
        self.p_2 = torch.tensor([0.0106], requires_grad=True, device=device)       # [min^(-1)]
        self.GEZI = torch.tensor([0.0022], requires_grad=True, device=device)      # [min^(-1)]
        self.EGP_0 = torch.tensor([1.33], requires_grad=True, device=device)       # [(mg/dL)/min]
        self.V_G = torch.tensor([253.0], requires_grad=True, device=device)        # [dL]
        self.tau_m = torch.tensor([47.0], requires_grad=True, device=device)       # [min]
        self.tau_sc = torch.tensor([5.0], requires_grad=True, device=device)       # [min]
        self.S_I = torch.tensor([0.0081], requires_grad=True, device=device)
        
    def forward(self, t):
        u = self.activation(self.input(t))
        u = self.activation(self.hidden(u))
        u = self.activation(self.hidden(u))
        u = self.activation(self.hidden(u))
        u = self.activation(self.hidden(u))
        u = self.activation(self.hidden(u))
        u = self.activation(self.hidden(u))
        u = self.output(u)

        return u

    def MVP(self, t, u, d):
        '''
        Input:
            x: Is the state tensor 
        '''
        
        # Calculate the state vector
        X = self.forward(t)

        # Meal system
        D_1 = X[:, 0]
        D_2 = X[:, 1]
        
        # Insulin system
        I_sc = X[:, 2]
        I_p = X[:, 3]
        I_eff = X[:, 4]
        
        # Glucose system        
        G = X[:, 5]
        G_sc = X[:, 6]
                
        tau_1 = self.tau_1
        tau_2 = self.tau_2
        C_I = self.C_I
        p_2 = self.p_2
        GEZI = self.GEZI
        EGP_0 = self.EGP_0
        V_G = self.V_G
        tau_m = self.tau_m
        tau_sc = self.tau_sc
        # S_I = self.S_I
        
        # Define gradients needed
        D_1_t = grad(D_1, t)
        D_2_t = grad(D_2, t)
        I_sc_t = grad(I_sc, t)
        I_p_t = grad(I_p, t)
        I_eff_t = grad(I_eff, t)
        G_t = grad(G, t)
        G_sc_t = grad(G_sc, t)
                
        # Define our ODEs
        Meal_1 = D_1_t - d + (D_1 / tau_m)
        Meal_2 = D_2_t - (D_1 / tau_m) + (D_2 / tau_m)
        
        Insulin1 = I_sc_t - (u / (tau_1 * C_I)) + (I_sc / tau_1)
        Insulin2 = I_p_t - ((I_sc - I_p) / tau_2)
        Insulin3 = I_eff_t - (-p_2 * I_eff + p_2 * S_I * I_p)
        
        Glucose1 = G_t - (-(GEZI + I_eff) * G + EGP_0 + ((1000 * D_2) / (V_G * tau_m)))
        Glucose2 = G_sc_t - ((G - G_sc) / tau_sc)
        
        # ODE = torch.stack([Meal_1, Meal_2, Insulin1, Insulin2, Insulin3, Glucose1, Glucose2], dim=1)
        
        loss_ode = torch.mean(Meal_1**2) + torch.mean(Meal_2**2) + torch.mean(Insulin1**2) + torch.mean(Insulin2**2) + torch.mean(Insulin3**2) + torch.mean(Glucose1**2 )+ torch.mean(Glucose2**2)

        return loss_ode

    def data_loss(self, t, data):
        X = self.forward(t)

        # Meal system
        D_1 = X[:, 0]
        D_2 = X[:, 1]
        
        # Insulin system
        I_sc = X[:, 2]
        I_p = X[:, 3]
        I_eff = X[:, 4]
        
        # Glucose system        
        G = X[:, 5]
        G_sc = X[:, 6]

        # Convert data to tensors
        D1_data = torch.tensor(data["D1"], device=t.device)
        D2_data = torch.tensor(data["D2"], device=t.device)
        I_sc_data = torch.tensor(data["I_sc"], device=t.device)
        I_p_data = torch.tensor(data["I_p"], device=t.device)
        I_eff_data = torch.tensor(data["I_eff"], device=t.device)
        G_data = torch.tensor(data["G"], device=t.device)
        G_sc_data = torch.tensor(data["G_sc"], device=t.device)

        data_1 = torch.mean((D_1 - D1_data)**2)
        data_2 = torch.mean((D_2 - D2_data)**2)
        data_3 = torch.mean((I_sc - I_sc_data)**2)
        data_4 = torch.mean((I_p - I_p_data)**2)
        data_5 = torch.mean((I_eff - I_eff_data)**2)
        data_6 = torch.mean((G - G_data)**2)
        data_7 = torch.mean((G_sc - G_sc_data)**2)

        data_loss = data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7

        return data_loss

    def loss(self, t_train, t_data, u, d, data):
        loss1 = self.MVP(t_train, u, d)
        loss2 = self.data_loss(t_data, data)

        return loss1 + loss2

if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our model
    model = PINN().to(device)

    # Define our parameter we want to estimate with initial guess
    S_I = torch.tensor([0.2], requires_grad=True, device=device)
    # The true value of S_I = 0.0081 [(dL/mU)/min]

    # Add the parameter to our optimizer
    # optimizer = torch.optim.Adam(list(model.parameters()) + [S_I], lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Define the training t data, collocation points    
    T = 300
    num_train_col = 1000

    # t_train = torch.linspace(0, T, 300, requires_grad=True, device=device).reshape(-1, 1)
    t_train_data = torch.linspace(0, T, 300, requires_grad=True, device=device).reshape(-1, 1)
    # t_train = torch.linspace(0, T, num_train_col, requires_grad=True, device=device).reshape(-1, 1)
    t_train = T * torch.rand(num_train_col, 1, requires_grad=True, device=device)
    d_train = torch.zeros(num_train_col, requires_grad=True, device=device)

    # Add a meal
    d_train = d_train.clone()
    d_train[0] = 20
    u_train = 25.04 * torch.ones(num_train_col, requires_grad=True, device=device)
    u_train = u_train.clone()
    u_train[0] += 100

    data = custom_csv_parser('Patient2.csv')


    # Define number of epoch
    num_epoch = 20000

    # Enable interactive mode for live plotting


    
    t_test = torch.linspace(0, T, 2500, device=device).reshape(-1, 1)
    t_data = torch.linspace(0, T, 300, device=device)

    # Begin training our model
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        loss = model.loss(t_train, t_train_data, u_train, d_train, data)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # Create the console output and plot
        if epoch%(num_epoch/10) == 0:
            # print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")#, Estimated a: {S_I.item():.6f}")

            # ax.clear()
            # X_pred = model(t_test)
            # G_pred = X_pred[:, 5]
            # # Plot G_pred over t_test
            # ax.plot(t_test.detach().cpu().numpy(), G_pred.detach().cpu().numpy(), label='Predicted Glucose (G)')
            # ax.plot(t_data.detach().cpu().numpy(), data["G"], label='True Glucose (G)')
            # ax.set_xlabel('Time (t)')
            # ax.set_ylabel('Predicted Glucose (G)')
            # # plt.title(f'Predicted Glucose (G) vs Time')
            # ax.set_title(f'Predicted Glucose (G) vs Time\nEstimated S_I: {S_I.item():.4f}, Epoch {epoch}')
            # ax.legend()
            # ax.grid(True)
            # plt.pause(0.1)

    

    # plt.ioff()
    # plt.show()
    X_pred = model(t_test)
    G_pred = X_pred[:, 5]
    plt.plot(t_test.detach().cpu().numpy(), G_pred.detach().cpu().numpy(), label='Predicted Glucose (G)')
    plt.plot(t_data.detach().cpu().numpy(), data["G"], label='True Glucose (G)')
    plt.xlabel('Time (t)')
    plt.ylabel('Predicted Glucose (G)')
    # plt.title(f'Predicted Glucose (G) vs Time')
    plt.title(f'Predicted Glucose (G) vs Time')#\nEstimated S_I: {S_I.item():.6f}')
    plt.legend()
    plt.show()