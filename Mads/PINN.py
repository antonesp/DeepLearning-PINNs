import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
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
    
    def __init__(self, input_dim=1, hidden_dim=20, output_dim=7, num_hidden=1):
        super().__init__()               
        self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim)

        self.hidden = nn.ModuleList()
        for _ in range(num_hidden):
            self.hidden.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
        
        
        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        
        self._initialize_weights()

        # Define activation function
        self.activation = nn.SiLU()

        # Define patient parameters
        self.tau_1 = torch.tensor([49.0], requires_grad=True, device=device)       # [min]
        self.tau_2 = torch.tensor([47.0], requires_grad=True, device=device)       # [min]
        self.C_I = torch.tensor([20.1], requires_grad=True, device=device)         # [dL/min]
        self.p_2 = torch.tensor([0.0106], requires_grad=True, device=device)       # [min^(-1)]
        # self.GEZI = torch.tensor([0.0022], requires_grad=True, device=device)      # [min^(-1)]
        self.EGP_0 = torch.tensor([1.33], requires_grad=True, device=device)       # [(mg/dL)/min]
        self.V_G = torch.tensor([253.0], requires_grad=True, device=device)        # [dL]
        self.tau_m = torch.tensor([47.0], requires_grad=True, device=device)       # [min]
        self.tau_sc = torch.tensor([5.0], requires_grad=True, device=device)       # [min]
        # self.S_I = torch.tensor([0.0081], requires_grad=True, device=device)
        
    def forward(self, t):
        u = self.activation(self.input(t))

        for hidden_layer in self.hidden:
            u = self.activation(hidden_layer(u))

        u = self.output(u)
        return u

    def _initialize_weights(self):
        # Initialize weights using Xavier initialization and biases to zero
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def MVP(self, t, u, d, scaling_mean, scaling_std):
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
        # GEZI = self.GEZI
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

        Insulin1 = I_sc_t - (u/(tau_1*C_I)) + (I_sc / tau_1)
        Insulin2 = I_p_t - (I_sc / tau_2) + (I_p / tau_2)
        Insulin3 = I_eff_t + p_2 * I_eff - p_2 * S_I * I_p

        Glucose1 = G_t + (GEZI + I_eff) * G - EGP_0 - ((1000 * D_2) / (V_G * tau_m))
        Glucose2 = G_sc_t - (G / tau_sc) + (G_sc / tau_sc)
                
        loss_ode = (1/scaling_mean[0]*torch.mean((Meal_1)**2)   + 
                    1/scaling_mean[1]*torch.mean(Meal_2**2)     + 
                    1/scaling_mean[2]*torch.mean(Insulin1**2)   + 
                    1/scaling_mean[3]*torch.mean(Insulin2**2)   + 
                    1/scaling_mean[4]*torch.mean(Insulin3**2)   + 
                    1/scaling_mean[5]*torch.mean(Glucose1**2 )  + 
                    1/scaling_mean[6]*torch.mean(Glucose2**2))
        return loss_ode
        # return Meal_1, Meal_2, Insulin1, Insulin2, Insulin3, Glucose1, Glucose2

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
        # D1_data = torch.tensor(data["D1"], device=t.device)
        D1_data = data["D1"]
        D2_data = data["D2"]
        I_sc_data = data["I_sc"]
        I_p_data = data["I_p"]
        I_eff_data = data["I_eff"]
        G_data = data["G"]
        G_sc_data = data["G_sc"]

        data_1 = torch.mean((D_1 - D1_data)**2)
        data_2 = torch.mean((D_2 - D2_data)**2)
        data_3 = torch.mean((I_sc - I_sc_data)**2)
        data_4 = torch.mean((I_p - I_p_data)**2)
        data_5 = torch.mean((I_eff - I_eff_data)**2)
        data_6 = torch.mean((G - G_data)**2)
        data_7 = torch.mean((G_sc - G_sc_data)**2)

        data_loss = data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7

        return data_loss

    def loss(self, t_train, t_data, u, d, data, scaling_mean, scaling_std):
        loss_ode = self.MVP(t_train, u, d, scaling_mean, scaling_std)
        
        # loss_ODE_std = [res.detach().norm() + 1e-6 for res in loss_ODE]  # Add a small value to avoid division by zero
        # for val in loss_ODE:
        #     print(val.max())
        # return 0

     # Normalize residuals and compute ODE loss
        # loss_ode = sum(
        #     torch.mean((res / std) ** 2)
        #     for res, std in zip(loss_ODE, loss_ODE_std)
        # )
        
        loss_data = self.data_loss(t_data, data)

        return loss_ode, loss_data

if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our model parameters
    hidden_dim = 128
    num_hidden_layers = 3

    # Load data and pre-process
    data = custom_csv_parser('../Patient2.csv')
    n_data = len(data["G"])

    # Split data into training and validation
    torch.manual_seed(42)

    indices = torch.randperm(n_data)


    n_train = int(n_data * 0.1)   # 80% training data

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Split the data dictionary 
    data_train = {}
    data_val = {}

    for key in data.keys():
        data_tensor = torch.tensor(data[key], requires_grad=True, device=device)          # Ensure data is a tensor
        data_train[key] = data_tensor[train_indices]
        data_val[key] = data_tensor[val_indices]


    # Define our model
    model = PINN(hidden_dim=hidden_dim, num_hidden=num_hidden_layers).to(device)

    # Define the optimizer and scheduler
    
    S_I = torch.tensor([0.0], requires_grad=True, device=device)
    GEZI = torch.tensor([0.0], requires_grad=True, device=device)      # [min^(-1)]

    # optimizer = torch.optim.Adam(list(model.parameters()) + [GEZI, S_I], lr=1e-3, weight_decay=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    scheduler = StepLR(optimizer, step_size=10000, gamma=0.95)
   
    # Define number of epoch
    num_epoch = 30000

    # Collocation points
    ### Options for improvement, try and extrend the collocation points after a large sum of training epochs, T + 5 or something
    
    d_train = data_train["Meal"]
    d_val = data_val["Meal"]
    
    u_train = data_train["Insulin"]
    u_val = data_val["Insulin"]
    
    t_train_data = data_train["t"].reshape(-1, 1)
    t_val_data = data_val["t"].reshape(-1, 1)
    
    T = data["t"][-1]
    t_train = torch.linspace(0, T, n_train, requires_grad=True, device=device).reshape(-1, 1)
    
    
    # Setup arrays for saving the losses
    train_losses = []
    val_losses = []
    learning_rates = []

    # Begin training our model
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        loss_ode, loss_data = model.loss(t_train, t_train_data, u_train, d_train, data_train, scaling_mean, scaling_std)
        loss = loss_ode + loss_data
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Create the console output and plot
        if epoch % 100 == 0:
            with torch.no_grad():
                model.eval()
                val_loss = model.data_loss(t_val_data, data_val)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            learning_rates.append(current_lr)

            # Print training and validation loss
        if epoch % 1000 == 0:
            # print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
            # print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, S_I: {S_I.item():.6f}")
            # print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, GEZI: {GEZI.item():.6f}, S_I: {S_I.item():.6f}")
            print(f"Epoch {epoch}, Loss ODE: {loss_ode.item():.6f}, Loss data: {loss_data.item():.6f}")#, S_I: {S_I.item():.6f}")

    

    # Plot training and validation losses and learning rate
    plt.figure(figsize=(18, 5))

    # First subplot for losses
    plt.subplot(1, 3, 1)
    epochs = range(0, num_epoch, 100)  # Since we record losses every 100 epochs
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    # Second subplot for learning rate
    plt.subplot(1, 3, 2)
    plt.plot(epochs, learning_rates, label='Learning Rate', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    # Third subplot for glucose predictions
    plt.subplot(1, 3, 3)
    t_test = torch.linspace(0, T + 150, 2500, device=device).reshape(-1, 1)
    X_pred = model(t_test)
    G_pred = X_pred[:, 5].detach().cpu().numpy()

    plt.plot(t_test.cpu().numpy(), G_pred, label='Predicted Glucose (G)')
    plt.plot(data["t"], data["G"], label='True Glucose (G)')
    plt.xlabel('Time (t)')
    plt.ylabel('Glucose Level')
    plt.title('Predicted vs True Glucose Levels')
    plt.legend()

    plt.tight_layout()
    plt.show()