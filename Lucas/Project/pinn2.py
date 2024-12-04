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

print("True parms", Si_true, tau1_true, tau2_true, Ci_true, p2_true, GEZI_true, EGP0_true, Vg_true, taum_true, tausc_true)

X_train, X_test, ts_train, ts_test, ts, X_train_arr, X_test_arr = data_split(data , train_frac = 0.55)
G_true = X_train["G"]
ts_train = torch.tensor(ts_train, device=device, dtype=torch.float32).unsqueeze(1)
ts_test = torch.tensor(ts_test, device=device, dtype=torch.float32).unsqueeze(1)
#print(ts_train.shape)
X_train_arr = torch.tensor(X_train_arr, device=device, dtype=torch.float32)
X_test_arr = torch.tensor(X_test_arr, device=device, dtype=torch.float32)

collacation_points = 10000
t_max = len(data['t']) + 1000
ts_collacation = torch.linspace(0, t_max, collacation_points).to(device).reshape(-1, 1)
p = 0.0

class DNN(nn.Module):
    '''Network'''
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i+1] ) for i in range(len(layers)-1)])
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(layers[i]) for i in range(1,len(layers))])
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
            x = self.batchnorm[i](x)
            x = self.dropout(x)
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

    # Define our dimensionless ODEs
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
        G_pred = u[:,5] * G_s
        loss = G_pred - data[:,5]
        return torch.sum(loss**2)
    
    def parameter_loss(self):
        parm_loss_scale = 10000
        zero = torch.tensor([0], device=device, dtype=torch.float32)
        parm_loss = (torch.min(torch.cat((Si, zero)))**2 + torch.min(torch.cat((tau1, zero)))**2 + torch.min(torch.cat((tau2, zero)))**2 + torch.min(torch.cat((Ci, zero)))**2 + torch.min(torch.cat((p2, zero)))**2 + torch.min(torch.cat((GEZI, zero)))**2 +  torch.min(torch.cat((EGP0, zero)))**2 + torch.min(torch.cat((Vg, zero)))**2 + torch.min(torch.cat((taum, zero)))**2 + torch.min(torch.cat((tausc, zero)))**2) * parm_loss_scale

        return parm_loss


    def pde_loss(self, ts_collacation):
        
        ts = ts_collacation.clone().detach()
        ts = ts.requires_grad_(True)
        u = self.nn(ts)        
        nn_dt = torch.zeros_like(u, device=device)

        for i in range(u.shape[1]):
            nn_dt[:, i] = autograd.grad(u[:, i], ts, grad_outputs=torch.ones_like(u[:, i]), create_graph=True)[0].squeeze()

        pd = pdes(u, nn_dt, ts)
        loss = torch.sum(pd**2)
        return loss
    
    def non_dim_ode_loss(self, ts_collacation):
        ts = ts_collacation.clone().detach()
        ts = ts.requires_grad_(True)
        u = self.nn(ts)        
        nn_dt = torch.zeros_like(u, device=device)

        for i in range(u.shape[1]):
            nn_dt[:, i] = autograd.grad(u[:, i], ts, grad_outputs=torch.ones_like(u[:, i]), create_graph=True)[0].squeeze()

        pd = nonDimODE(u, nn_dt, ts)
        loss = torch.sum(pd**2)
        return loss

    
    def loss(self, ts,ts_collacation, data):
        l1 = self.data_loss(ts, data)
        l2 = self.non_dim_ode_loss(ts_collacation)
        l3 = self.parameter_loss()
        return l1 + l2 + l3, l1, l2, l3
    
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
lrs = [i*0.2*1e-3 for i in range(2,9)]
network_layers = [[1, 64, 128, 64, 7]]
epochs = 50000

wandb.init(
    # set the wandb project where this run will be logged
    project="Parameter Estimation in MVP",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

for lr in lrs:
    for layers in network_layers:
        print(layers)
        print(lr)
        NN = PINN(layers)
        Si0 = 0.01
        tau10 = 47.0
        tau20 = 47.0
        Ci0 = 18.0
        p20 = 0.0005
        GEZI0 = 0.0005
        EGP00 = 2.0
        Vg0 = 300.0
        taum0 = 50.0
        tausc0 = 6.0

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
        values_of_l2 = []
        values_of_l3 = []
        adapt_weights = torch.tensor([1,1,1])

        relative_errs = []
        epochs_to_plot = []
        num_epochs_to_plot = 100

        NN.nn.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            _, l1, l2, l3 = NN.loss(ts_train, ts_collacation, X_train_arr)
            values_of_l1.append(l1)
            values_of_l2.append(l2)
            values_of_l3.append(l3)

            if epoch % num_epochs_to_plot == 0:
                sum_rel_errors = abs(Si_true - Si.item()) / Si_true + abs(tau1_true - tau1.item()) / tau1_true + abs(tau2_true - tau2.item()) / tau2_true + abs(Ci_true - Ci.item()) / Ci_true + abs(p2_true - p2.item()) / p2_true + abs(GEZI_true - GEZI.item()) / GEZI_true + abs(EGP0_true - EGP0.item()) / EGP0_true + abs(Vg_true - Vg.item()) / Vg_true + abs(taum_true - taum.item()) / taum_true + abs(tausc_true - tausc.item()) / tausc_true
                relative_errs.append(sum_rel_errors)
                epochs_to_plot.append(epoch)

            if epoch % epochs_to_make_updates == 0 and epoch != 0:
                adapt_weights = normalized_softadapt_object.get_component_weights(torch.tensor(values_of_l1), 
                                                                        torch.tensor(values_of_l2), 
                                                                        torch.tensor(values_of_l3),
                                                                        verbose=False,
                                                                        )
                values_of_l1 = []
                values_of_l2 = []
                values_of_l3 = []
            loss = adapt_weights[0] * l1 + adapt_weights[1] * l2 + adapt_weights[2] * l3
            loss.backward()
            optimizer.step()
            if epoch % 2000 == 0:
                print(f"Epoch: {epoch}, Data loss: {l1.item():.4f}, ODE loss: {l2.item():.4f}, Parameter loss: {l3.item():.4f}")
                print(f"Epoch: {epoch}, Adaptive data loss: {adapt_weights[0]*l1.item():.4f}, Adaptive ODE loss: {adapt_weights[1]*l2.item():.4f}, Adaptive parameter loss: {adapt_weights[2]*l3.item():.4f}")
                print(f"Si: {Si.item():.5f}, tau1: {tau1.item():.5f}, tau2: {tau2.item():.5f}, Ci: {Ci.item():.5f}, p2: {p2.item():.5f}, GEZI: {GEZI.item():.5f}, EGP0: {EGP0.item():.5f}, Vg: {Vg.item():.5f}, taum: {taum.item():.5f}, tausc: {tausc.item():.5f}")
                # if epoch % 2000 == 0:
                #     with torch.no_grad():
                #         NN.nn.eval()
                #         pred, err = NN.test(ts_test)
                #         print(f"Test Error: {err:.2f}")
                #     NN.nn.train()


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


        layers_str = "_".join(map(str, layers))
        folder_name = f"network_layers_{layers_str}_lr_{lr:.5f}"
        folder_path = os.path.join("with_dropout", folder_name)
        # Create the directory if it does not exist
        os.makedirs(folder_path, exist_ok=True)
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
            plt.show()
        plt.figure()
        plt.plot(epochs_to_plot, relative_errs)
        plt.title("Sum of relative errors")
        plt.savefig(folder_path+"/error_in_epoch.png")
        plt.show()
        