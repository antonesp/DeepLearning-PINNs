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

print("True parms", Si_true, tau1_true, tau2_true, Ci_true, p2_true, GEZI_true, EGP0_true, Vg_true, taum_true, tausc_true)

X_train, X_test, ts_train, ts_test, ts, X_train_arr, X_test_arr = data_split(data , train_frac = 0.55)
G_true = X_train["G"]
ts_train = torch.tensor(ts_train, device=device, dtype=torch.float32).unsqueeze(1)
ts_test = torch.tensor(ts_test, device=device, dtype=torch.float32).unsqueeze(1)
#print(ts_train.shape)
X_train_arr = torch.tensor(X_train_arr, device=device, dtype=torch.float32)
X_test_arr = torch.tensor(X_test_arr, device=device, dtype=torch.float32)

print(X_train_arr.shape)
print(X_test_arr.shape)

p = 0.0

class DNN(nn.Module):
    '''Network'''
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1] ) for i in range(len(layers)-1)])

        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(p=p)

        'Initialization'
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight, gain=1.0)
            nn.init.zeros_(self.linears[i].bias)

    def forward(self, x):
        for i in range(0, len(self.linears) - 1):
            x = self.linears[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        x = self.linears[-1](x)
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
    
    # Convert parameters to tensors
    #taum = torch.tensor(parms['taum'], device=device)
    #tau1 = torch.tensor(parms['tau1'], device=device)
    # tau2 = torch.tensor(parms['tau2'], device=device)
    # p2 = torch.tensor(parms['p2'], device=device)
    # Ci = torch.tensor(parms['Ci'], device=device)
    # GEZI = torch.tensor(parms['GEZI'], device=device)
    # EGP0 = torch.tensor(parms['EGP0'], device=device)
    # Vg = torch.tensor(parms['Vg'], device=device)
    # tausc = torch.tensor(parms['tausc'], device=device)

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
        loss = u[:,6] - data[:,6]
        return torch.sum(loss**2)


    def pde_loss(self, ts):
        
        ts = ts.clone().detach()
        ts = ts.requires_grad_(True)
        u = self.nn(ts)        
        nn_dt = torch.zeros_like(u, device=device)

        for i in range(u.shape[1]):
            nn_dt[:, i] = autograd.grad(u[:, i], ts, grad_outputs=torch.ones_like(u[:, i]), create_graph=True)[0].squeeze()

        pd = pdes(u, nn_dt, ts)
        loss = nn.MSELoss()(pd, torch.zeros_like(pd))
        return loss
    
    def loss(self, ts, data):
        l1 = self.data_loss(ts, data)
        l2 = self.pde_loss(ts)
        return l1 + 2000*l2, l1, 2000*l2
    
    def test(self, t):
        #t = t.unsqueeze(0)
        X_pred = self.nn.forward(t)
        #X_true = dict_to_vec(data, t)
        if len(t) == 450:
            X_true = X_train_arr
        else:
            X_true = X_test_arr
        error_vec = torch.linalg.norm((X_true-X_pred),2)/torch.linalg.norm(X_true,2)
        X_pred = X_pred.cpu().detach().numpy()
        return X_pred, error_vec
NN = PINN([1, 64, 128, 128, 7])
Si0 = 0.01
tau10 = 45.0
tau20 = 45.0
Ci0 = 17.0
p20 = 0.0005
GEZI0 = 0.0005
EGP00 = 2.0
Vg0 = 300.0
taum0 = 50.0
tausc0 = 6.0

p = 0.0
lr = 9e-4
Si = torch.tensor([Si0], requires_grad=True, device=device)
tau1 = torch.tensor([tau10], requires_grad=True, device=device)
tau2 = torch.tensor([tau20], requires_grad=True, device=device)
Ci = torch.tensor([Ci0], requires_grad=True, device=device)
p2 = torch.tensor([p20], requires_grad=True, device=device)
GEZI = torch.tensor([GEZI0], requires_grad=True, device=device)
EGP0 = torch.tensor([EGP00], requires_grad=True, device=device)
Vg = torch.tensor([Vg0], requires_grad=True, device=device)
taum = torch.tensor([taum0], requires_grad=True, device=device)
tausc = torch.tensor([tausc0], requires_grad=True, device=device)

optimizer = optim.Adam(list(NN.nn.parameters()) + [Si, tau1, tau2, Ci, p2, GEZI, EGP0, Vg, taum, tausc], lr=lr)

'''Test Network works'''
pred, err = NN.test(torch.tensor([1], device=device, dtype=torch.float32))
print(f"""Prediction: {pred} \n Error: {err}""")
print(f"Current guess for {Si.item()}: ")

pred = NN.nn.forward(ts_train)
true = X_train_arr
#plt.plot(ts_train.cpu().detach().numpy(), pred.cpu().detach().numpy()[:,5], label='Prediction')
#print(torch.mean((pred - true)**2))

start_time = time.time()
ts_train = torch.tensor(ts_train, device=device, dtype=torch.float32).reshape(-1,1)
ts_test = torch.tensor(ts_test, device=device, dtype=torch.float32).reshape(-1,1)
epochs = 30000
NN.nn.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    loss, l1, l2 = NN.loss(ts_train, X_train_arr)
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"Epoch: {epoch}, Data loss: {l1.item():.4f}, ODE loss: {l2.item():.4f}, Si: {Si.item():.5f}, tau1: {tau1.item():.5f}")
        if epoch % 2000 == 0:
            with torch.no_grad():
                NN.nn.eval()
                pred, err = NN.test(ts_test)
                print(f"Test Error: {err:.2f}")
            NN.nn.train()





table = PrettyTable()

table.field_names = ["Parameter", "True Value", "Final Guess", "Relative Error"]

# Calculate relative error and add rows with formatted values
table.add_row(["Si", f"{Si_true:.4f}", f"{Si.item():.4f}", f"{abs(Si_true - Si.item()) / Si_true:.4f}"])
table.add_row(["tau1", f"{tau1_true:.4f}", f"{tau1.item():.4f}", f"{abs(tau1_true - tau1.item()) / tau1_true:.4f}"])
table.add_row(["tau2", f"{tau2_true:.4f}", f"{tau2.item():.4f}", f"{abs(tau2_true - tau2.item()) / tau2_true:.4f}"])
table.add_row(["Ci", f"{Ci_true:.4f}", f"{Ci.item():.4f}", f"{abs(Ci_true - Ci.item()) / Ci_true:.4f}"])
table.add_row(["p2", f"{p2_true:.4f}", f"{p2.item():.4f}", f"{abs(p2_true - p2.item()) / p2_true:.4f}"])
table.add_row(["GEZI", f"{GEZI_true:.4f}", f"{GEZI.item():.4f}", f"{abs(GEZI_true - GEZI.item()) / GEZI_true:.4f}"])
table.add_row(["EGP0", f"{EGP0_true:.4f}", f"{EGP0.item():.4f}", f"{abs(EGP0_true - EGP0.item()) / EGP0_true:.4f}"])
table.add_row(["Vg", f"{Vg_true:.4f}", f"{Vg.item():.4f}", f"{abs(Vg_true - Vg.item()) / Vg_true:.4f}"])
table.add_row(["taum", f"{taum_true:.4f}", f"{taum.item():.4f}", f"{abs(taum_true - taum.item()) / taum_true:.4f}"])
table.add_row(["tausc", f"{tausc_true:.4f}", f"{tausc.item():.4f}", f"{abs(tausc_true - tausc.item()) / tausc_true:.4f}"])

#print(table)
with open("table.txt", "w") as file:
    file.write(table.get_string())

NN.nn.eval()
pred = NN.nn.forward(ts_test)


G_pred = pred[:,5].reshape(-1,1)
G_true = X_test_arr[:,5].reshape(-1,1)
G_true_2 = X_train_arr[:,5].reshape(-1,1)
ts_test_plot = ts_test.cpu().numpy()


for i in range(7):
    plt.figure()
    plt.plot(ts_test_plot, pred[:,i].detach().cpu().numpy(), label='Predicted')
    plt.plot(ts_test_plot, X_test_arr[:,i].cpu().numpy(), label='True')
    print("Error" , torch.sum((torch.tensor(pred[:,i]) - X_test_arr[:,i])**2))
    plt.legend()
    plt.title(pde_keys[i])
    plt.savefig("figs/"+pde_keys[i]+".png")
    plt.show()