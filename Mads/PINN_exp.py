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
        
        self.a = Parameter(data=torch.tensor(0.5), requires_grad=True)
        
        self.activation = F.relu
        
        self.input = nn.Linear(in_features=1, out_features=100)
        
        self.hidden = nn.Linear(in_features=100, out_features=100)
        
        self.output = nn.Linear(in_features=100, out_features=1)

    def forward(self, t):
        u = self.input(t)
        u = self.activation(u)
        u = self.hidden(u)
        u = self.activation(u)
        u = self.hidden(u)
        u = self.activation(u)
        u = self.hidden(u)
        u = self.activation(u)
        u = self.output(u)
        return u
        
    
    def loss(self, t):
        u = self.forward(t)
        u_t = grad(u, t)
        
        ode = u_t - self.a * u
        initial_condition = self.forward(torch.ones_like(t) * 0) - 1
        
        # Observed points, e.g., (t, y) pairs from y = exp(2 * t)
        t_observed = torch.tensor([[0.5], [1.0]], requires_grad=True)
        u_observed = torch.exp(2 * t_observed)  # Expected y values for given t points
        u_pred = model(t_observed)
        
        # Compute MSE
        loss_ode = torch.mean(ode**2)
        loss_ics = torch.mean(initial_condition**2)
        loss_data = torch.mean((u_pred - u_observed)**2)
        
        
        
        loss = loss_ode + 10*loss_ics + 10*loss_data
        
        return loss
    
    def data_loss(self):
        # Observed points, e.g., (t, y) pairs from y = exp(2 * t)
        t_observed = torch.tensor([[0.5], [1.0]], requires_grad=True)
        y_observed = torch.exp(2 * t_observed)  # Expected y values for given t points
        y_pred = model(t_observed)
        return torch.mean((y_pred - y_observed)**2)

if __name__ == "__main__":
    
    model = PINN()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    t_train = torch.rand(100, 1)
    t_train.requires_grad = True
    
    for i in range(100000):
        loss = model.loss(t_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i%(100000/10) == 0:
            print("training: ", loss)
            
            
    # Testing
    t_test = torch.linspace(0, 1, 100).reshape(-1, 1)
    
    y_exact = np.exp(2 * t_test.detach().numpy())
    y_pred = model(t_test).detach().numpy()
    
    t_vals = t_test.detach().numpy()
    # print(model(t_test))
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(t_vals, y_exact, label="Exact Solution $y(t) = e^t$", linestyle="--")
    plt.plot(t_vals, y_pred, label="PINN Approximation", linestyle="-")
    plt.xlabel("Time $t$")
    plt.ylabel("$y(t)$")
    plt.title("PINN Solution vs Exact Solution for $y_t = y$, $y(0) = 1$")
    plt.legend()
    plt.grid(True)
    plt.show()