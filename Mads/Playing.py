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
from softadapt import SoftAdapt, LossWeightedSoftAdapt, NormalizedSoftAdapt

from prettytable import PrettyTable



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

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Define activation function
        self.activation = nn.Tanh()

        # Define patient parameters
        self.tau_1 = (torch.tensor([49.0], requires_grad=True, device=device))       # [min]
        self.tau_2 = (torch.tensor([47.0], requires_grad=True, device=device))       # [min]
        self.C_I = (torch.tensor([20.1], requires_grad=True, device=device))         # [dL/min]
        self.p_2 = (torch.tensor([0.0106], requires_grad=True, device=device))       # [min^(-1)]
        self.GEZI = Parameter(torch.tensor([0.0], requires_grad=True, device=device))      # [min^(-1)]
        self.EGP_0 = Parameter(torch.tensor([0.0], requires_grad=True, device=device))       # [(mg/dL)/min]
        self.V_G = (torch.tensor([253.0], requires_grad=True, device=device))        # [dL]
        self.tau_m = (torch.tensor([47.0], requires_grad=True, device=device))       # [min]
        self.tau_sc = (torch.tensor([5.0], requires_grad=True, device=device))       # [min]
        self.S_I = (torch.tensor([0.0081], requires_grad=True, device=device))
        
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

    def MVP(self, t, u, d, scaling_mean):
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
        S_I = self.S_I
        
        # Define gradients needed
        D_1_t = grad(D_1, t)
        D_2_t = grad(D_2, t)
        I_sc_t = grad(I_sc, t)
        I_p_t = grad(I_p, t)
        I_eff_t = grad(I_eff, t)
        G_t = grad(G, t)
        G_sc_t = grad(G_sc, t)
                
        # Define our ODEs
        Meal1 = D_1_t - d + (D_1 / tau_m)
        Meal2 = D_2_t - (D_1 / tau_m) + (D_2 / tau_m)

        Insulin1 = I_sc_t - (u/(tau_1*C_I)) + (I_sc / tau_1)
        Insulin2 = I_p_t - (I_sc / tau_2) + (I_p / tau_2)
        Insulin3 = I_eff_t + p_2 * I_eff - p_2 * S_I * I_p

        Glucose1 = G_t + (GEZI + I_eff) * G - EGP_0 - ((1000 * D_2) / (V_G * tau_m))
        Glucose2 = G_sc_t - (G / tau_sc) + (G_sc / tau_sc)
                
        loss_ode = (1/scaling_mean[0]*torch.mean((Meal1)**2)   + 
                    1/scaling_mean[1]*torch.mean(Meal2**2)     + 
                    1/scaling_mean[2]*torch.mean(Insulin1**2)   + 
                    1/scaling_mean[3]*torch.mean(Insulin2**2)   + 
                    1/scaling_mean[4]*torch.mean(Insulin3**2)   + 
                    1/scaling_mean[5]*torch.mean(Glucose1**2 )  + 
                    1/scaling_mean[6]*torch.mean(Glucose2**2))

        mvp = torch.stack([Meal1, Meal2, Insulin1, Insulin2, Insulin3, Glucose1, Glucose2], dim=1)

        return mvp

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

        data_1 = self.loss_fn(D_1, D1_data)
        data_2 = self.loss_fn(D_2, D2_data)
        data_3 = self.loss_fn(I_sc, I_sc_data)
        data_4 = self.loss_fn(I_p, I_p_data)
        data_5 = self.loss_fn(I_eff, I_eff_data)
        data_6 = self.loss_fn(G, G_data)
        data_7 = self.loss_fn(G_sc, G_sc_data)

        data_loss = data_1 + data_2 + data_3 + data_4 + data_5 + data_6 + data_7
        # data_loss = data_6

        return data_loss

    def loss(self, t_train, t_data, u, d, data, scaling_mean):
        ODE = self.MVP(t_train, u, d, scaling_mean)

        True_ODE = torch.zeros_like(ODE[:, 0], device=device)
        loss_d1 = self.loss_fn(ODE[:, 0] / scaling_mean[0], True_ODE)
        loss_d2 = self.loss_fn(ODE[:, 1] / scaling_mean[1], True_ODE)
        loss_isc = self.loss_fn(ODE[:, 2] / scaling_mean[2], True_ODE)
        loss_ip = self.loss_fn(ODE[:, 3] / scaling_mean[3], True_ODE)
        loss_ieff = self.loss_fn(ODE[:, 4] / scaling_mean[4], True_ODE)
        loss_g = self.loss_fn(ODE[:, 5] / scaling_mean[5], True_ODE)
        loss_gsc = self.loss_fn(ODE[:, 6] / scaling_mean[6], True_ODE)

        
        loss_data = self.data_loss(t_data, data)

        loss = [loss_d1, loss_d2, loss_isc, loss_ip, loss_ieff, loss_g, loss_gsc, loss_data]

        return loss

if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our model parameters
    hidden_dim = 128
    num_hidden_layers = 3

    # Load data and pre-process
    data = custom_csv_parser('Patient3.csv')
    n_data = len(data["G"])

    # Split data into training and validation
    torch.manual_seed(42)

    indices = torch.randperm(n_data)


    n_train = int(n_data * 0.8)   # 80% training data

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Define  
    T = data["t"][-1]

    t_data = torch.linspace(0, T, n_data, device=device)
    t_train_data = t_data[train_indices].reshape(-1, 1)
    t_val_data = t_data[val_indices].reshape(-1, 1)

    # Split the data dictionary 
    data_train = {}
    data_val = {}

    for key in data.keys():
        data_tensor = torch.tensor(data[key], device=device)          # Ensure data is a tensor
        data_train[key] = data_tensor[train_indices]
        data_val[key] = data_tensor[val_indices]


    # Define our model
    model = PINN(hidden_dim=hidden_dim, num_hidden=num_hidden_layers).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

   
    # Define number of epoch
    num_epoch = 15000

    # Collocation points
    ### Options for improvement, try and extrend the collocation points after a large sum of training epochs, T + 5 or something

    d_train = data_train["Meal"].clone()
    u_train = data_train["Insulin"].clone()
    t_train = data_train["t"].clone().requires_grad_(True).reshape(-1, 1)



    # Try naive scaling
    # 'D1', 'D2', 'I_sc', 'I_p', 'I_eff', 'G', 'G_sc'
    scaling_mean = torch.Tensor([data_train["D1"].mean(), data_train["D2"].mean(), data_train["I_sc"].mean(), data_train["I_p"].mean(), data_train["I_eff"].mean(), data_train["G"].mean(), data_train["G_sc"].mean()])

    # Setup arrays for saving the losses
    train_losses = []
    val_losses = []
    learning_rates = []

    # Setup SoftAdapt
    softadapt_object = SoftAdapt(beta=0.1)
    epochs_to_make_updates = 5
    ODE_loss1 = []
    ODE_loss2 = []
    ODE_loss3 = []
    ODE_loss4 = []
    ODE_loss5 = []
    ODE_loss6 = []
    ODE_loss7 = []
    data_loss = []
    adapt_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


    # Begin training our model
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        loss = model.loss(t_train, t_train_data, u_train, d_train, data_train, scaling_mean)

        loss_ode1 = loss[0]
        loss_ode2 = loss[1]
        loss_ode3 = loss[2]
        loss_ode4 = loss[3]
        loss_ode5 = loss[4]
        loss_ode6 = loss[5]
        loss_ode7 = loss[6]

        loss_data = loss[7]
        loss = adapt_weight[0] * loss_ode1 + adapt_weight[1] * loss_ode2 + adapt_weight[2] * loss_ode3 + adapt_weight[3] * loss_ode4 + adapt_weight[4] * loss_ode5 + adapt_weight[5] * loss_ode6 + adapt_weight[6] * loss_ode7 + adapt_weight[7] * loss_data
        loss.backward()
        optimizer.step()
        
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
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, GEZI: {model.GEZI.item():.6f}, S_I: {model.S_I.item():.6f}")

    parameter_info = [
        ('tau_1', 49.0),
        ('tau_2', 47.0),
        ('C_I', 20.1),
        ('p_2', 0.0106),
        ('GEZI', 0.0022),
        ('EGP_0', 1.33),
        ('V_G', 253.0),
        ('tau_m', 47.0),
        ('tau_sc', 5.0),
        ('S_I', 0.0081)
    ]

    estimated_values = []
    true_values = []
    relative_errors = []

    # Step 3: Compute estimated values and relative errors
    with torch.no_grad():
        for name, true_value in parameter_info:
            raw_param = getattr(model, f'{name}')
            est_value = raw_param.item()
            rel_error = abs((est_value - true_value) / true_value)
            estimated_values.append(est_value)
            true_values.append(true_value)
            relative_errors.append(rel_error)

    # Step 4: Create and populate the PrettyTable
    table = PrettyTable()
    table.field_names = ["Parameter", "Estimated Value", "True Value", "Relative Error"]

    for name, est_value, true_value, rel_error in zip(
            [p[0] for p in parameter_info], estimated_values, true_values, relative_errors):
        # Determine formatting based on the parameter's magnitude
        if true_value >= 1.0:
            est_str = f"{est_value:.4f}"
            true_str = f"{true_value:.4f}"
        else:
            est_str = f"{est_value:.6f}"
            true_str = f"{true_value:.6f}"
        rel_err_str = f"{rel_error:.2f}"
        table.add_row([name, est_str, true_str, rel_err_str])

    # Align the columns
    table.align["Parameter"] = "l"
    table.align["Estimated Value"] = "r"
    table.align["True Value"] = "r"
    table.align["Relative Error"] = "r"

    # Display the table
    print(table)
    t_test = torch.linspace(0, T+100, 2500, device=device).reshape(-1, 1)
    X_pred = model(t_test)
    G_pred = X_pred[:, 5].detach().cpu().numpy()

    plt.plot(t_test.cpu().numpy(), G_pred, label='Predicted Glucose (G)')
    plt.plot(data["t"], data["G"], label='True Glucose (G)')
    plt.xlabel('Time (t)')
    plt.ylabel('Glucose Level')
    plt.title('Predicted vs True Glucose Levels')
    plt.legend()
    plt.show()