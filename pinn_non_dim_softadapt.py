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


if device == 'cuda': 
    print(torch.cuda.get_device_name()) 

import sys
import os
current_dir = os.getcwd()

functions_dir = os.path.join(current_dir,'..', 'functions')
sys.path.append(functions_dir)

from Load_data import custom_csv_parser2, custom_csv_parser
from Load_data import data_split

pde_keys = ['D1', 'D2', 'I_sc','I_p', 'I_eff', 'G', 'G_sc']

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
            x = self.dropout(x)
        x = self.linears[-1](x)
        x = torch.nn.Softplus()(x)
        return x

def nonDimODE(u, nn_dt, ts):
    D_1s = 47.0
    D_2s = 47.0
    I_scs = 0.0477
    I_ps = 0.0477
    I_effs = 0.0000193
    G_s = 454.54
    G_scs = 4272.676
    t_s = 47.0

    D1 = u[:, 0]  
    D2 = u[:, 1]  
    I_sc = u[:, 2]  
    I_p = u[:, 3]  
    I_eff = u[:, 4]  
    G = u[:, 5]  
    G_sc = u[:, 6]  

    pdes = torch.zeros_like(u, device=device)
    D_1_t = nn_dt[:,0]
    D_2_t = nn_dt[:,1]
    I_sc_t = nn_dt[:,2]
    I_p_t = nn_dt[:,3]
    I_eff_t = nn_dt[:,4]
    G_t = nn_dt[:,5]
    G_sc_t = nn_dt[:,6]

    ds = torch.zeros_like(u[:,0], device=device)
    us = torch.ones_like(u[:,0], device=device) * 25.04
    zero_indices = (ts == 0).nonzero(as_tuple=True)[0]
    us[zero_indices] = data['Insulin'][0]
    ds[zero_indices] = data['Meal'][0]

    pdes[:,0] = D_1_t - (ds * t_s) / D_1s + t_s / taum * D1
    pdes[:,1] = D_2_t - (t_s * D_1s) / (taum * D_2s) * D1

    pdes[:,2] = I_sc_t - (t_s * us) / (tau1 * Ci * I_scs) + (t_s / tau1) * I_sc
    pdes[:,3] = I_p_t - (t_s * I_scs) / (tau2 * I_ps) * I_scs + (t_s / tau2) * I_p
    pdes[:,4] = I_eff_t + p2 * t_s * I_eff - ((p2 * Si * I_ps * t_s) / I_effs) * I_p
    
    pdes[:,5] = G_t + (t_s * GEZI + t_s * I_effs * I_eff) * G + (t_s * EGP0) / G_s + (1000 * t_s * D_2s) / (Vg * taum * G_s) * D2
    pdes[:,6] = G_sc_t - (G_s * t_s) / (G_scs * tausc) * G + (t_s / tausc) * G_sc
        
    return pdes

class PINN():
    def __init__(self, layers):
        self.iter = 0
        self.nn = DNN(layers).to(device)

    def data_loss(self, ts, data):
        G_s = 454.54
        u = self.nn(ts)  # Outputs ts x 7 array
        G_pred = u[:,5]# * G_s
        loss = G_pred - data[:,5]
        return torch.sum(loss**2)
    
    def parameter_loss(self):
        parm_loss_scale = 1000000
        zero = torch.tensor([0], device=device, dtype=torch.float32)
        parm_loss = (torch.min(torch.cat((Si, zero)))**2 + torch.min(torch.cat((tau1, zero)))**2 + torch.min(torch.cat((tau2, zero)))**2 + torch.min(torch.cat((Ci, zero)))**2 + torch.min(torch.cat((p2, zero)))**2 + torch.min(torch.cat((GEZI, zero)))**2 +  torch.min(torch.cat((EGP0, zero)))**2 + torch.min(torch.cat((Vg, zero)))**2 + torch.min(torch.cat((taum, zero)))**2 + torch.min(torch.cat((tausc, zero)))**2) * parm_loss_scale

        return parm_loss
    
    def non_dim_ode_loss(self, ts_collacation):
        ts = ts_collacation.clone().detach()
        ts = ts.requires_grad_(True)
        u = self.nn(ts)        
        nn_dt = torch.zeros_like(u, device=device)

        for i in range(u.shape[1]):
            nn_dt[:, i] = autograd.grad(u[:, i], ts, grad_outputs=torch.ones_like(u[:, i]), create_graph=True)[0].squeeze()

        pd = nonDimODE(u, nn_dt, ts)
        loss = pd**2
        loss = torch.sum(loss, dim = 0)
        return loss

    
    def loss(self, ts,ts_collacation, data):
        l1 = self.data_loss(ts, data)
        l2 = self.non_dim_ode_loss(ts_collacation)
        l3 = self.parameter_loss()
        return l1, l2, l3
    

lr = 0.4 * 1e-3
layers = [1, 64, 128, 64, 7]
epochs = 20000


NN = PINN(layers)
Si0 = 0.01
tau10 = 47.0
tau20 = 47.0
Ci0 = 18.0
p20 = 0.02
GEZI0 = 0.1
EGP00 = 2.0
Vg0 = 280.0
taum0 = 50.0
tausc0 = 4.0

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

ts_train = torch.tensor(ts_train, device=device, dtype=torch.float32).reshape(-1,1)
ts_test = torch.tensor(ts_test, device=device, dtype=torch.float32).reshape(-1,1)

epochs_to_make_updates = 5
softadapt_object = LossWeightedSoftAdapt(beta=0.1)
normalized_softadapt_object  = NormalizedSoftAdapt(beta=0.1)
loss_weighted_softadapt_object = LossWeightedSoftAdapt(beta=0.1)
values_of_l1 = []
values_of_l2a = []
values_of_l2b = []
values_of_l2c = []
values_of_l2d = []
values_of_l2e = []
values_of_l2f = []
values_of_l2g = []
values_of_l3 = []
adapt_weights = torch.tensor([1,1,1,1,1,1,1,1])

relative_errs = []
epochs_to_plot = []
num_epochs_to_plot = 100

NN.nn.train()
for epoch in range(epochs):
    optimizer.zero_grad()
    l1, l2, l3 = NN.loss(ts_train, ts_collacation, X_train_arr)
    #print(l2)
    values_of_l1.append(l1)
    values_of_l2a.append(l2[0])
    values_of_l2b.append(l2[1])
    values_of_l2c.append(l2[2])
    values_of_l2d.append(l2[3])
    values_of_l2e.append(l2[4])
    values_of_l2f.append(l2[5])
    values_of_l2g.append(l2[6])
    values_of_l3.append(l3)

    if epoch % num_epochs_to_plot == 0:
        sum_rel_errors = ((abs(Si_true - Si.item()) / Si_true)**2
                        + (abs(tau1_true - tau1.item()) / tau1_true )**2
                        + (abs(tau2_true - tau2.item()) / tau2_true)**2 
                        + (abs(Ci_true - Ci.item()) / Ci_true )**2
                        + (abs(p2_true - p2.item()) / p2_true )**2
                        + (abs(GEZI_true - GEZI.item()) / GEZI_true )**2
                        + (abs(EGP0_true - EGP0.item()) / EGP0_true)**2 
                        + (abs(Vg_true - Vg.item()) / Vg_true )**2
                        + (abs(taum_true - taum.item()) / taum_true)**2 
                        + (abs(tausc_true - tausc.item()) / tausc_true)**2)
        relative_errs.append(sum_rel_errors)
        epochs_to_plot.append(epoch)

    # if epoch % epochs_to_make_updates == 0 and epoch != 0:
    #     adapt_weights = normalized_softadapt_object.get_component_weights(torch.tensor(values_of_l1), 
    #                                                             torch.tensor(values_of_l2a), 
    #                                                             torch.tensor(values_of_l2b), 
    #                                                             torch.tensor(values_of_l2c), 
    #                                                             torch.tensor(values_of_l2d), 
    #                                                             torch.tensor(values_of_l2e), 
    #                                                             torch.tensor(values_of_l2f), 
    #                                                             torch.tensor(values_of_l2g), 
    #                                                             verbose=False,
    #                                                             )#torch.tensor(values_of_l3),
    #     values_of_l1 = []
    #     values_of_l2a = []
    #     values_of_l2b = []
    #     values_of_l2c = []
    #     values_of_l2d = []
    #     values_of_l2e = []
    #     values_of_l2f = []
    #     values_of_l2g = []
    #     values_of_l3 = []
    # loss = (adapt_weights[0] * l1 
    #         + adapt_weights[1] * l2[0]
    #         + adapt_weights[2] * l2[1]
    #         + adapt_weights[3] * l2[2]
    #         + adapt_weights[4] * l2[3]
    #         + adapt_weights[5] * l2[4]
    #         + adapt_weights[6] * l2[5]
    #         + adapt_weights[7] * l2[6] + l3)
            #+ adapt_weights[8] * l3)
    loss = l1 + l2[0] + l2[1] + l2[2] + l2[3]+ l2[4]+ l2[5]+ l2[6]+ l3
    loss.backward()
    optimizer.step()
    if epoch % 2000 == 0:
        print(f"Epoch: {epoch}, Data loss: {l1.item():.4f}, ODE loss: {l2[0].item():.4f}, Parameter loss: {l3.item():.4f}")
        print(f"Epoch: {epoch}, Adaptive data loss: {adapt_weights[0]*l1.item():.4f}, Adaptive ODE loss: {adapt_weights[1]*l2[0].item():.4f}, Adaptive parameter loss: {adapt_weights[2]*l3.item():.4f}")
        print(f"Si: {Si.item():.5f}, tau1: {tau1.item():.5f}, tau2: {tau2.item():.5f}, Ci: {Ci.item():.5f}, p2: {p2.item():.5f}, GEZI: {GEZI.item():.5f}, EGP0: {EGP0.item():.5f}, Vg: {Vg.item():.5f}, taum: {taum.item():.5f}, tausc: {tausc.item():.5f}")
        # if epoch % 2000 == 0:
        #     with torch.no_grad():
        #         NN.nn.eval()
        #         pred, err = NN.test(ts_test)
        #         print(f"Test Error: {err:.2f}")
        #     NN.nn.train()

array = np.array(relative_errs)
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

folder_name = "non_dim_softadapt"
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

D_1s = 47.0
D_2s = 47.0
I_scs = 0.0477
I_ps = 0.0477
I_effs = 0.0000193
G_s = 454.54
G_scs = 4272.676
scales = [D_1s, D_2s, I_scs, I_ps, I_effs, G_s, G_scs]

for i in range(7):
    plt.figure()
    plt.plot(ts_collacation_plot, pred[:,i].detach().cpu().numpy() * scales[i], label='Predicted')
    plt.plot(ts_test_plot, X_test_arr[:,i].cpu().numpy(), label='True')
    plt.legend()
    plt.title(pde_keys[i])
    plt.savefig(folder_path+"/"+pde_keys[i]+".png")
    file_path = os.path.join(folder_path, pde_keys[i] + ".txt")
    np.savetxt(file_path, pred[:,i].detach().cpu().numpy() * scales[i], fmt='%f')
    plt.show()
plt.figure()
plt.plot(epochs_to_plot, relative_errs)
plt.title("Sum of relative errors")
plt.savefig(folder_path+"/error_in_epoch.png")
plt.show()
