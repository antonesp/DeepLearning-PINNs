import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import torch.autograd as autograd
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import numpy as np

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
    
    def __init__(self):
        super().__init__()       
        self.activation = nn.Tanh()
        
        self.input = nn.Linear(in_features=1, out_features=20)
        self.hidden = nn.Linear(in_features=20, out_features=20)
        self.output = nn.Linear(in_features=20, out_features=1)

    def forward(self, t):
        u = self.activation(self.input(t))
        u = self.activation(self.hidden(u))
        u = self.output(u)
        return u
        
    
    def loss(self, t, true_coef=2):
        
        # Define the PDE loss
        u = self.forward(t)
        u_t = grad(u, t)
        ode = u_t - a * u
        loss_ode = torch.mean(ode**2)

        # Define the initial condition loss
        initial_t = torch.zeros(100, device=t.device).reshape(-1, 1)
        initial_condition = self.forward(initial_t) - 1.0
        loss_ics = torch.mean(initial_condition**2)
        
        # Define the data loss
        t_observed = torch.linspace(0, 1, 10, device=t.device).reshape(-1, 1)
        u_observed = torch.exp(true_coef * t_observed)  # Expected y values for given t points
        u_pred = self.forward(t_observed)
        loss_data = torch.mean((u_pred - u_observed)**2)
                
        # Combine the loss function
        loss = loss_ode + loss_ics + loss_data
        
        return loss
    
if __name__ == "__main__":
    # Check for CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define our model
    model = PINN().to(device)

    # Define our parameter we want to estimate with initial guess
    a = torch.tensor([1.0], requires_grad=True, device=device)
    true_a = torch.tensor([3.0], requires_grad=True, device=device)

    # Add the parameter to our optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + [a], lr=0.001)
    
    # Define the training t data, collocation points
    # t_train = torch.linspace(0, 1, 100, requires_grad=True, device=device).reshape(-1, 1)
    
    # Can also be done with random numbers
    t_train = torch.rand(100, 1, requires_grad=True, device=device)

    # Define number of epoch
    num_epoch = 10000

    # Enable interactive mode for live plotting
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create the testing
    t_test = torch.linspace(0, 1, 100, device=device).reshape(-1, 1)
    y_pred = model(t_test).detach().cpu().numpy()
    
    t_vals = t_test.detach().cpu().numpy()
    
    y_exact = np.exp(true_a.detach().cpu().numpy() * t_vals)

    # Begin training our model
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        loss = model.loss(t_train, true_coef=true_a)
        loss.backward()
        optimizer.step()
        
        # Create the console output and plot
        if epoch%(num_epoch/50) == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}, Estimated a: {a.item():.6f}")
        
            # Update the plot every (num_epoch // 10) epochs
            ax.clear()  # Clear previous plots
            y_pred = model(t_test).detach().cpu().numpy()
            ax.plot(t_vals, y_exact, label="Exact Solution $y(t) = e^{3t}$", linestyle="--")
            ax.plot(t_vals, y_pred, label="PINN Approximation", linestyle="-")
            ax.set_xlabel("Time $t$")
            ax.set_ylabel("$y(t)$")
            ax.set_title(f"PINN Solution vs Exact Solution\nEstimated a: {a.item():.4f}, Epoch: {epoch}")
            ax.legend()
            ax.grid(True)
            plt.pause(0.01)  # Pause to update the figure

    # Turn off interactive mode and show the final plot
    plt.ioff()
    plt.show()