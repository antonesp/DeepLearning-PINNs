import torch
import torch.autograd as autograd
import torch.nn as nn
from torch import Tensor
import torch.optim as optim


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
import scipy.io
from prettytable import PrettyTable
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
import os
import wandb

torch.set_default_dtype(torch.float)

# Random number generators in other libraries
np.random.seed(28)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

# Lucas/this_script.py
import sys
import os
current_dir = os.getcwd()
print(current_dir)
# Add the 'functions' folder to the Python path
functions_dir = os.path.join(current_dir,'..', 'functions')
sys.path.append(functions_dir)
functions_dir = os.path.join(current_dir,'..', 'Mads')
sys.path.append(functions_dir)

# Now you can import the function
from Load_data import custom_csv_parser2, custom_csv_parser
from Load_data import data_split

pde_keys = ['D1', 'D2', 'I_sc','I_p', 'I_eff', 'G', 'G_sc']
patient_keys = ['tau1', 'tau2', 'Ci', 'p2', 'GEZI', 'EGP0', 'Vg', 'taum', 'tausc']
patient_keys_si = ['tau1', 'tau2', 'Ci', 'p2', 'Si', 'GEZI', 'EGP0', 'Vg', 'taum', 'tausc']

def dict_to_vec(X, t):
    
    t  = int(t)
    print(t)
    X_vec = torch.tensor([X['D1'][t], X['D2'][t], X['I_sc'][t], X['I_p'][t], X['I_eff'][t], X['G'][t], X['G_sc'][t]], device=device, dtype=torch.float32)

    return X_vec.to(device).reshape(1, -1)

scale_vec = torch.tensor([1.56, 1.54, 1.24, 1.24, 0.01, 117.56, 117.48], device=device, dtype=torch.float32)
# Use the function
data = custom_csv_parser2('../Patient2.csv')
parms = custom_csv_parser('../Patient.csv')


Si_true = parms['Si']
tau1_true = parms['tau1']
tau2_true = parms['tau2']
Ci_true = parms['Ci']
p2_true = parms['p2']
GEZI_true = parms['GEZI'][0]
EGP0_true = parms['EGP0']
Vg_true = parms['Vg']
taum_true = parms['taum']
tausc_true = parms['tausc']

# Set true parameters, since we're not estimating them here anyway
taum = taum_true
tau1 = tau1_true
Ci = Ci_true
tau2 = tau2_true
p2 = p2_true
GEZI = GEZI_true
EGP0 = EGP0_true
Vg = Vg_true
tausc = tausc_true

# Load and prepare data for torch
X_train, X_test, ts_train, ts_test, ts, X_train_arr, X_test_arr = data_split(data , train_frac = 0.55)
G_true = X_train["G"]
ts_train = torch.tensor(ts_train, device=device, dtype=torch.float32).unsqueeze(1)
ts_test = torch.tensor(ts_test, device=device, dtype=torch.float32).unsqueeze(1)
X_train_arr = torch.tensor(X_train_arr, device=device, dtype=torch.float32)
X_test_arr = torch.tensor(X_test_arr, device=device, dtype=torch.float32)

collacation_points = 20000
t_max = len(data['t']) + 1000
ts_collacation = torch.linspace(0, t_max, collacation_points).to(device).reshape(-1, 1)

class DNN(nn.Module):
    '''Network'''
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1] ) for i in range(len(layers)-1)])
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(layers[i]) for i in range(1,len(layers))])
        self.activation = nn.Tanh()

        'Initialization'
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight, gain=1.0)
            nn.init.zeros_(self.linears[i].bias)

    def forward(self, x):
        for i in range(0, len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.batchnorm[i](x)
        x = self.linears[-1](x)
        x = torch.nn.Softplus()(x)
        return x

def pdes(u, nn_dt, ts):
    pdes = torch.zeros_like(u, device=device)
    D1 = u[:, 0]  
    D2 = u[:, 1]  
    I_sc = u[:, 2]  
    I_p = u[:, 3]  
    I_eff = u[:, 4]  
    G = u[:, 5]  
    G_sc = u[:, 6]  

    ds = torch.zeros_like(u[:,0], device=device)
    us = torch.ones_like(u[:,0], device=device) * 25.04
    zero_indices = (ts == 0).nonzero(as_tuple=True)[0]
    us[zero_indices] = data['Insulin'][0]
    ds[zero_indices] = data['Meal'][0]

    # Example assignment for all rows of pdes
    pdes[:, 0] = nn_dt[:, 0] + D1 / taum - ds
    pdes[:, 1] = nn_dt[:, 1] - (D1 - D2) / taum
    pdes[:, 2] = nn_dt[:, 2] + I_sc / tau1 - us / (tau1 * Ci)
    pdes[:, 3] = nn_dt[:, 3] - (I_sc - I_p) / tau2
    pdes[:, 4] = nn_dt[:, 4] + p2 * I_eff - p2 * Si * I_p
    pdes[:, 5] = nn_dt[:, 5] + (GEZI + I_eff) * G - EGP0 - 1000 * D2 / (Vg * taum)
    pdes[:, 6] = nn_dt[:, 6] - (G - G_sc) / tausc

    return pdes / scale_vec

class PINN():
    def __init__(self, layers):
        self.iter = 0
        self.nn = DNN(layers).to(device)

    def data_loss(self, ts, data):
        u = self.nn(ts)  # Outputs ts x 7 array
        G_pred = u
        loss = G_pred - data
        return torch.sum(loss**2)
    
    def parameter_loss(self):
        parm_loss_scale = 1000000
        zero = torch.tensor([0], device=device, dtype=torch.float32)
        parm_loss = (torch.min(torch.cat((Si, zero)))**2) * parm_loss_scale

        return parm_loss


    def ode_loss(self, ts_collacation):
        
        ts = ts_collacation.clone().detach()
        ts = ts.requires_grad_(True)
        u = self.nn(ts)        
        nn_dt = torch.zeros_like(u, device=device)

        for i in range(u.shape[1]):
            nn_dt[:, i] = autograd.grad(u[:, i], ts, grad_outputs=torch.ones_like(u[:, i]), create_graph=True)[0].squeeze()

        pd = pdes(u, nn_dt, ts)
        loss = pd**2
        loss = torch.sum(loss, dim = 0)
        return loss

    def loss(self, ts,ts_collacation, data):
        l1 = self.data_loss(ts, data)
        l2 = self.ode_loss(ts_collacation)
        l3 = self.parameter_loss()
        return l1, l2, l3
torch.manual_seed(42)
lr =  1e-3
layers = [1, 128, 128, 128, 7]
epochs = 15000

NN = PINN(layers)
Si0 = 0.01

Si = torch.tensor([Si0], requires_grad=True, device=device)

optimizer = optim.Adam(list(NN.nn.parameters()) + [Si], lr=lr, weight_decay=1e-5)

ts_train = torch.tensor(ts_train, device=device, dtype=torch.float32).reshape(-1,1)
ts_test = torch.tensor(ts_test, device=device, dtype=torch.float32).reshape(-1,1)

relative_errs = []
epochs_to_plot = []
num_epochs_to_plot = 100

NN.nn.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    l1, l2, l3 = NN.loss(ts_train, ts_collacation, X_train_arr)

    if epoch % num_epochs_to_plot == 0:
        sum_rel_errors = ((Si_true - Si.item()) / Si_true)**2
                        
        relative_errs.append(sum_rel_errors)
        epochs_to_plot.append(epoch)

    loss = l1 + l2[0] + l2[1] + l2[2] + l2[3]+ l2[4]+ l2[5]+ l2[6]+ l3
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"Epoch: {epoch}, Data loss: {l1.item():.4f}, ODE loss: {l2[0].item():.4f}, Parameter loss: {l3.item():.4f}")
        print(f"Si: {Si.item():.5f}")

array = np.array(relative_errs)
table = PrettyTable()
table.field_names = ["Parameter", "True Value", "Final Guess", "Relative Error"]
table.add_row(["Si", f"{Si_true:.4f}", f"{Si.item():.4f}", f"{abs(Si_true - Si.item()) / Si_true:.4f}"])

folder_name = "only_si"
folder_path = os.path.join("plots", folder_name)
os.makedirs(folder_path, exist_ok=True)

file_path = os.path.join(folder_path, "squared_relative_errors.txt")
np.savetxt(file_path, array, fmt='%f')

file_path = os.path.join(folder_path, "table.txt")
with open(file_path, "w") as file:
    file.write(table.get_string())

NN.nn.eval()
pred = NN.nn.forward(ts_collacation)
pred_test = NN.nn.forward(ts_test)
ts_collacation_plot = ts_collacation.cpu().numpy()

G_pred = pred[:,5].reshape(-1,1)
G_true = X_test_arr[:,5].reshape(-1,1)
G_true_2 = X_train_arr[:,5].reshape(-1,1)
ts_test_plot = ts_test.cpu().numpy()

for i in range(7):
    plt.figure()
    plt.plot(ts_collacation_plot, pred[:,i].detach().cpu().numpy(), label='Predicted')
    plt.plot(ts_test_plot, X_test_arr[:,i].cpu().numpy(), label='True')
    plt.legend()
    plt.title(pde_keys[i])
    plt.savefig(folder_path+"/"+pde_keys[i]+".png")
    file_path = os.path.join(folder_path, pde_keys[i] + ".txt")
    np.savetxt(file_path, pred[:,i].detach().cpu().numpy(), fmt='%f')
    plt.show()
plt.figure()
plt.plot(epochs_to_plot, relative_errs)
plt.title("Sum of relative errors")
plt.savefig(folder_path+"/error_in_epoch.png")
plt.show()
        