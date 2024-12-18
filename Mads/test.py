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
from softadapt import SoftAdapt, LossWeightedSoftAdapt

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

        # Define the layers
        self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.hidden = nn.ModuleList()
        for _ in range(num_hidden):
            self.hidden.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim)) 
        
        # Initialize weights
        self._initialize_weights()

        # Define activation function
        self.activation = nn.Tanh()

        # Define loss function
        self.loss_fn = nn.MSELoss()

        # Define softplus as to not get negative values
        self.softplus = nn.Softplus()

        # Define patient parameters 
        self.tau_1 =    Parameter(torch.tensor([47.0], requires_grad=True, device=device))       # [min]
        self.tau_2 =    Parameter(torch.tensor([47.0], requires_grad=True, device=device))       # [min]
        self.C_I =      Parameter(torch.tensor([18.0], requires_grad=True, device=device))         # [dL/min]
        self.p_2 =      Parameter(torch.tensor([0.0005], requires_grad=True, device=device))       # [min^(-1)]
        self.GEZI =     Parameter(torch.tensor([0.0005], requires_grad=True, device=device))      # [min^(-1)]
        self.EGP_0 =    Parameter(torch.tensor([2.0], requires_grad=True, device=device))       # [(mg/dL)/min]
        self.V_G =      Parameter(torch.tensor([300.0], requires_grad=True, device=device))        # [dL]
        self.tau_m =    Parameter(torch.tensor([50.0], requires_grad=True, device=device))       # [min]
        self.tau_sc =   Parameter(torch.tensor([6.0], requires_grad=True, device=device))       # [min]
        self.S_I =      Parameter(torch.tensor([0.01], requires_grad=True, device=device))
    

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
                
        # Save all the ODEs in a tensor
        mvp = torch.stack([Meal1, Meal2, Insulin1, Insulin2, Insulin3, Glucose1, Glucose2], dim=1)

        return mvp

    def data_loss(self, t, data):
        
        
        # Calculate the state vector
        X = self.forward(t)
        
        # Glucose system        
        G = X[:, 5]

        # Scale the data
        G_data = data["G"].clone()

        # Calculate the loss
        loss_data = self.loss_fn(G, G_data)

        return loss_data

    def loss(self, t_train, u, d, data):
        ODE = self.MVP(t_train, u, d)
        True_ODE = torch.zeros_like(ODE)
        loss_ode = self.loss_fn(ODE, True_ODE)

        
        loss_data = self.data_loss(t_train, data)

        loss = loss_ode + loss_data

        return loss, loss_ode, loss_data

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
    n_train = int(n_data * 0.8)   # 80% training data

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Define  
    T = data["t"][-1]

    # Split the data dictionary 
    data_train = {}
    data_val = {}

    for key in data.keys():
        data_tensor = torch.tensor(data[key], device=device)          # Ensure data is a tensor
        data_train[key] = data_tensor[train_indices]
        data_val[key] = data_tensor[val_indices]

    d_train = data_train["Meal"].clone()
    u_train = data_train["Insulin"].clone()
    t_train = data_train["t"].clone().requires_grad_(True).reshape(-1, 1)
    t_val = data_val["t"].clone().requires_grad_(True).reshape(-1, 1)

    # Define our model
    model = PINN(hidden_dim=hidden_dim, num_hidden=num_hidden_layers).to(device)

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
   
    # Define number of epoch
    num_epoch = 30000

    # Setup arrays for saving the losses
    train_losses = []
    val_losses = []
    learning_rates = []


    # Setup SoftAdapt
    softadapt_object = SoftAdapt(beta=0.1)
    epochs_to_make_updates = 5
    ODE_loss = []
    data_loss = []
    adapt_weight = [1.0, 1.0]   

    # Define the true parameters
    tau1_true = 49.0
    tau2_true = 47.0
    Ci_true = 20.1
    p2_true = 0.0106
    GEZI_true = 0.0022
    EGP0_true = 1.33
    Vg_true = 253.0
    taum_true = 47.0
    tausc_true = 5.0
    Si_true = 0.0081

    relative_errs = []

    # Begin training our model
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        _, loss_ode, loss_data = model.loss(t_train, u_train, d_train, data_train)
        loss = adapt_weight[0] * loss_ode + adapt_weight[1] * loss_data
        loss.backward()
        optimizer.step()
        # Calculate softadapt weights
        ODE_loss.append(loss_ode)
        data_loss.append(loss_data)
        if epoch % epochs_to_make_updates == 0 and epoch != 0:
            adapt_weight = softadapt_object.get_component_weights(torch.tensor(ODE_loss), 
                                                                torch.tensor(data_loss),
                                                                verbose=False,
                                                                )
            
            ODE_loss = []
            data_loss = []


        # Create the console output and plot
        if epoch % 100 == 0:
            with torch.no_grad():
                model.eval()
                val_loss = model.data_loss(t_val, data_val)
                sum_rel_errors = (abs(Si_true - model.S_I.item()) / Si_true 
                                + abs(tau1_true - model.tau_1.item()) / tau1_true 
                                + abs(tau2_true - model.tau_2.item()) / tau2_true 
                                + abs(Ci_true - model.C_I.item()) / Ci_true 
                                + abs(p2_true - model.p_2.item()) / p2_true 
                                + abs(GEZI_true - model.GEZI.item()) / GEZI_true 
                                + abs(EGP0_true - model.EGP_0.item()) / EGP0_true 
                                + abs(Vg_true - model.V_G.item()) / Vg_true 
                                + abs(taum_true - model.tau_m.item()) / taum_true 
                                + abs(tausc_true - model.tau_sc.item()) / tausc_true)
                relative_errs.append(sum_rel_errors)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())
            # learning_rates.append(current_lr)

            # Print training and validation loss
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")
            # print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, S_I: {S_I.item():.6f}")
            # print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, GEZI: {GEZI.item():.6f}, S_I: {S_I.item():.6f}")
            # print(f"Epoch {epoch}, Loss ODE: {loss_ode.item():.6f}, Loss data: {loss_data.item():.6f}")#, S_I: {S_I.item():.6f}")
    
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
    print(relative_errs[-1])






    # Plot training and validation losses and learning rate
    plt.figure(figsize=(18, 5))

    # # First subplot for losses
    plt.subplot(2, 4, 1)
    epochs = range(0, num_epoch, 100)  # Since we record losses every 100 epochs
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    # Third subplot for glucose predictions
    plt.subplot(2, 4, 2)
    t_test = torch.linspace(0, T, 2500, device=device).reshape(-1, 1)
    X_pred = model(t_test)
    G_pred = X_pred[:, 5].detach().cpu().numpy()

    plt.plot(t_test.cpu().numpy(), G_pred, label='Predicted Glucose (G)')
    plt.plot(data["t"], data["G"], label='True Glucose (G)')
    plt.xlabel('Time (t)')
    plt.ylabel('Glucose Level')
    plt.title('Predicted vs True Glucose Levels')
    plt.legend()

    plt.subplot(2, 4, 3)
    Gsc_pred = X_pred[:, 6].detach().cpu().numpy()
    plt.plot(t_test.cpu().numpy(), Gsc_pred , label='Predicted (Gsc)')
    plt.plot(data["t"], data["G_sc"], label='True (Gsc)')
    plt.legend()

    plt.subplot(2, 4, 4)
    D1_pred = X_pred[:, 0].detach().cpu().numpy()
    plt.plot(t_test.cpu().numpy(), D1_pred, label='Predicted (D1)')
    plt.plot(data["t"], data["D1"], label='True Glucose (D1)')
    plt.legend()

    plt.subplot(2, 4, 5)
    D2_pred = X_pred[:, 1].detach().cpu().numpy()
    plt.plot(t_test.cpu().numpy(), D2_pred, label='Predicted Glucose (D2)')
    plt.plot(data["t"], data["D2"], label='True Glucose (D2)')
    plt.legend()

    plt.subplot(2, 4, 6)
    Isc_pred = X_pred[:, 2].detach().cpu().numpy()
    plt.plot(t_test.cpu().numpy(), Isc_pred, label='Predicted (Isc)')
    plt.plot(data["t"], data["I_sc"], label='True Glucose (Isc)')
    plt.legend()

    plt.subplot(2, 4, 7)
    Ip_pred = X_pred[:, 3].detach().cpu().numpy()
    plt.plot(t_test.cpu().numpy(), Ip_pred, label='Predicted (Ip)')
    plt.plot(data["t"], data["I_p"], label='True (Ip)')
    plt.legend()

    plt.subplot(2, 4, 8)
    Ieff_pred = X_pred[:, 4].detach().cpu().numpy()
    plt.plot(t_test.cpu().numpy(), Ieff_pred, label='Predicted (Ieff)')
    plt.plot(data["t"], data["I_eff"], label='True (Ieff)')
    plt.legend()

    plt.tight_layout()
    plt.show()