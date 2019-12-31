import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.io as sio
import scipy.misc
from mu_updater import mu_updater_dict

import argparse

parser = argparse.ArgumentParser(description='Test LADMM with synthetic data')
parser.add_argument('--use-learned', action='store_true', help='use learned model')
parser.add_argument('--use-safeguard', action='store_true', help='use safeguarding')
parser.add_argument('--delta', type=float, default=-99.0, help='delta in safeguarding')
parser.add_argument('--mu-k-method', type=str, default='None', help='mu_k update method')
parser.add_argument('--mu-k-param', type=float, default=0.0, help='mu_k update parameter')
parser.add_argument('--test-file', type=str, default='syn_data_unseen.mat', help='data used for testing')
parser.add_argument('--model-file', type=str, default='DLADMMNet.pth', help='L2O model to be loaded and tested')
parser.add_argument('--layers', type=int, default=20, help='number of layers of the L2O model')
parser.add_argument('--objective', type=str, default='NMSE', help='objective for observations')

args = parser.parse_args()

use_learned = args.use_learned
use_safeguard = args.use_safeguard
num_iter = 200
alpha = 0.5
delta = args.delta
mu_k_method = args.mu_k_method
mu_k_param  = args.mu_k_param
test_file = args.test_file
model_file = args.model_file
layers = args.layers
K = layers if use_learned or use_safeguard else num_iter
objective = args.objective

## network definition
class DLADMMNet(nn.Module):
    def __init__(self, m, n, d, batch_size, A, Z0, E0, L0, layers):
        super(DLADMMNet, self).__init__()
        self.m = m
        self.n = n
        self.d = d
        self.batch_size = batch_size
        self.A = A.cuda()
        self.At = self.A.t()
        self.Z0 = Z0.cuda()
        self.E0 = E0.cuda()
        self.L0 = L0.cuda()
        self.layers = layers

        self.A_np = A.cpu().numpy()
        self.L = (np.linalg.norm(np.matmul(self.A_np.transpose(), self.A_np), ord=2) * torch.ones(1,1)).float().cuda()

        self.beta1 = nn.ParameterList()
        self.beta2 = nn.ParameterList()
        self.active_para = nn.ParameterList()
        self.active_para1 = nn.ParameterList()
        self.fc = nn.ModuleList()

        for k in range(self.layers):
            # self.beta1.append(nn.Parameter(torch.ones(self.m, self.batch_size, dtype=torch.float32)))
            # self.beta2.append(nn.Parameter(torch.ones(self.m, self.batch_size, dtype=torch.float32)))
            self.beta1.append(nn.Parameter(torch.ones(self.m, 1, dtype=torch.float32)))
            self.beta2.append(nn.Parameter(torch.ones(self.m, 1, dtype=torch.float32)))
            self.active_para.append(nn.Parameter(0.025 * torch.ones(self.d, 1, dtype=torch.float32)))
            self.active_para1.append(nn.Parameter(0.06 * torch.ones(self.m, 1, dtype=torch.float32)))
            self.fc.append(nn.Linear(self.m, self.d, bias = False))

        # self.active_para = torch.tensor(0.025, dtype=torch.float32).cuda()
        # self.active_para1 = torch.tensor(0.06, dtype=torch.float32).cuda()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                #m.weight.data.normal_(0, 1/20)
                m.weight = torch.nn.Parameter(self.A.t() + (1e-3)*torch.randn_like(self.A.t()))
                #m.weight = torch.nn.Parameter(self.A.t())


    def self_active(self, x, thershold):
        return F.relu(x - thershold) - F.relu(-1.0 * x - thershold)


    def two_norm(self, z):
        norm_array = (z ** 2).sum(dim=0).sqrt()
        return norm_array


    def KM(self, Zk, Ek, Lk, Tk, X, **kwargs):
        beta = 1.0 / self.L.sqrt()
        ss1 = 1.0 / self.L.sqrt()
        ss2 = 0.5

        Varn = Lk + beta * Tk
        Zn = self.self_active(Zk - ss1 * self.At.mm(Varn), ss1 * alpha)

        # NOTE: En step is different from D-LADMM implementation
        TTn = self.A.mm(Zn) + Ek - X
        En = self.self_active(Ek - ss2 * (Lk + beta * TTn), ss2)

        Tn = self.A.mm(Zn) + En - X
        Ln = Lk + beta * Tn

        return Varn, Zn, En, Tn, Ln


    def S(self, Zk, Ek, Lk, Tk, X, Ep, **kwargs):
        beta = 1.0 / self.L.sqrt()
        ss1 = 1.0 / self.L.sqrt()
        ss2 = 0.5

        Varn, Zn, En, Tn, Ln = self.KM(Zk, Ek, Lk, Tk, X, **kwargs)
        c1 = beta * ss2
        assert c1 < 1
        c = (c1 / (1 - c1)).sqrt()
        Sn = torch.cat([beta * Tn, c * (En - 2*Ek + Ep)])

        return Sn


    def forward(self, x):
        X = x
        T = list()
        Var = list()
        Z = list()
        E = list()
        L = list()
        T.append(self.A.mm(self.Z0) + self.E0 - X)

        if use_learned and use_safeguard:
            # NOTE: only take Tn for safegaurding
            S0 = self.S(self.Z0, self.E0, self.L0, T[-1], X, self.E0)
            # mu_k = self.two_norm(self.KM(self.Z0, self.E0, self.L0, T[-1], X)[-2])
            mu_k = self.two_norm(S0)
            mu_updater = mu_updater_dict[mu_k_method](mu_k, mu_k_param)
            sg_count = np.zeros(self.layers)

        for k in range(K):
            if k == 0 :
                # Classic algorithm
                Varn_KM, Zn_KM, En_KM, Tn_KM, Ln_KM = self.KM(self.Z0, self.E0, self.L0, T[-1], X)
                # L2O algorithm
                if use_learned:
                    # Tself.A.mm(self.Z0) + self.E0 - X)
                    Varn_L2O = self.L0 + self.beta1[k].mul(T[-1])
                    Zn_L2O = self.self_active(self.Z0 - self.fc[k](Varn_L2O.t()).t(), self.active_para[k])
                    En_L2O = self.self_active(X - self.A.mm(Zn_L2O) - self.beta2[k].mul(self.L0), self.active_para1[k])
                    Tn_L2O = self.A.mm(Zn_L2O) + En_L2O - X
                    Ln_L2O = self.L0 + self.beta1[k].mul(Tn_L2O)

            else :
                # Classic algorithm
                Varn_KM, Zn_KM, En_KM, Tn_KM, Ln_KM = self.KM(Z[-1], E[-1], L[-1], T[-1], X)
                # L2O algorithm
                if use_learned:
                    Varn_L2O = L[-1] + self.beta1[k].mul(T[-1])
                    Zn_L2O = self.self_active(Z[-1] - self.fc[k](Varn_L2O.t()).t(), self.active_para[k])
                    En_L2O = self.self_active(X - self.A.mm(Zn_L2O) - self.beta2[k].mul(L[-1]), self.active_para1[k])
                    Tn_L2O = self.A.mm(Zn_L2O) + En_L2O - X
                    Ln_L2O = L[-1] + self.beta1[k].mul(Tn_L2O)

            if use_safeguard:
                # L2O + safegaurding
                assert use_learned
                Ep         = self.E0 if k == 0 else E[-1]
                S_L2O      = self.S(Zn_L2O, En_L2O, Ln_L2O, Tn_L2O, X, Ep)
                S_L2O_norm = self.two_norm(S_L2O)
                # print(S_L2O_norm)
                # print(mu_k)
                # print(" ")
                bool_term       = (S_L2O_norm < (1.0-delta) * mu_k).float()
                mu_k            = mu_updater.step(S_L2O_norm, bool_term)
                bool_term       = bool_term.reshape(1, bool_term.shape[0])
                bool_complement = 1.0 - bool_term

                Var.append(bool_term * Varn_L2O + bool_complement * Varn_KM)
                Z.append(bool_term * Zn_L2O   + bool_complement * Zn_KM)
                E.append(bool_term * En_L2O   + bool_complement * En_KM)
                T.append(bool_term * Tn_L2O   + bool_complement * Tn_KM)
                L.append(bool_term * Ln_L2O   + bool_complement * Ln_KM)

                sg_count[k] = bool_complement.sum().cpu().item()

            elif use_learned:
                # Only L2O
                Var.append(Varn_L2O)
                Z.append(Zn_L2O)
                E.append(En_L2O)
                T.append(Tn_L2O)
                L.append(Ln_L2O)

            else:
                # Classic KL algorithm
                Var.append(Varn_KM)
                Z.append(Zn_KM)
                E.append(En_KM)
                T.append(Tn_KM)
                L.append(Ln_KM)

        if use_learned and use_safeguard:
            return Z, E, L, T, sg_count
        else:
            return Z, E, L, T


    def name(self):
        return "DLADMMNet"


# other functions
def trans2image(img):
    # img 256 x 1024
    # img = img.cpu().data.numpy()
    new_img = np.zeros([im_size, im_size])
    count = 0
    for ii in range(0, im_size, 16):
            for jj in range(0, im_size, 16):
                    new_img[ii:ii+16,jj:jj+16] = np.transpose(np.reshape(img[:, count],[16,16]))
                    count = count+1
    return new_img

def l2_normalize(inputs):
    [batch_size, dim] = inputs.shape
    inputs2 = torch.mul(inputs, inputs)
    norm2 = torch.sum(inputs2, 1)
    root_inv = torch.rsqrt(norm2)
    tmp_var1 = root_inv.expand(dim,batch_size)
    tmp_var2 = torch.t(tmp_var1)
    nml_inputs = torch.mul(inputs, tmp_var2)
    return nml_inputs

def l2_col_normalize(inputs):
    [dim1, dim2] = inputs.shape
    inputs2 = np.multiply(inputs, inputs)
    norm2 = np.sum(inputs2, 0)
    root = np.sqrt(norm2)
    root_inv = 1/root
    tmp_var1 = np.tile(root_inv,dim1)
    tmp_var2 = tmp_var1.reshape(dim1, dim2)
    nml_inputs = np.multiply(inputs, tmp_var2)
    return nml_inputs

def calc_PSNR(x1, x2):
    x1 = x1 * 255.0
    x2 = x2 * 255.0
    mse = F.mse_loss(x1, x2)
    psnr = -10 * torch.log10(mse) + torch.tensor(48.131)
    return psnr

def dual_gap(x, alpha):
    out = F.softplus(x - alpha) + F.softplus(- x - alpha)
    return out


np.random.seed(1126)
# os.environ["CUDA_VISIBLE_DEVICES"]="7"
device_ids = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))))
# m, d, n = 256, 512, 10000
m, d, n = 250, 500, 10000
# n_test = 1024 if 'lena' in test_file or 'fake' in test_file else 256
n_test = 1000
batch_size = 20

use_cuda = torch.cuda.is_available()
print('==>>> use cuda' if use_cuda else '==>>> use cpu')
print('==>>> batch size: {}'.format(batch_size))
# print('==>>> total trainning batch number: {}'.format(n//batch_size))
print('==>>> total testing batch number: {}'.format(n_test//batch_size))

syn_data = sio.loadmat(test_file)
A_ori = syn_data['A']
A_ori = A_ori.astype(np.float32) #*(1.0/18.0)

X = syn_data['train_x'].astype(np.float32)
X = X.T

Z_tr = syn_data['train_z'].astype(np.float32)
Z_tr = Z_tr.T

E_tr = syn_data['train_e'].astype(np.float32)
E_tr = E_tr.T

X_ts = syn_data['test_x'].astype(np.float32)
X_ts = X_ts.T

Z_ts = syn_data['test_z'].astype(np.float32)
Z_ts = Z_ts.T

E_ts = syn_data['test_e'].astype(np.float32)
E_ts = E_ts.T

# X_gt = syn_data['gt_x'].astype(np.float32)
# X_gt = X_gt.T


# init parameters
Z0 = 1.0 /d * torch.rand(d, batch_size, dtype=torch.float32)
E0 = torch.zeros(m, batch_size, dtype=torch.float32)
L0 = torch.zeros(m, batch_size, dtype=torch.float32)
A_tensor = torch.from_numpy(A_ori)

model = DLADMMNet(
    m=m, n=n, d=d, batch_size=batch_size, A=A_tensor,
    Z0=Z0, E0=E0, L0=L0, layers=layers)
A_tensor = A_tensor.cuda()
if use_cuda:
    model = model.cuda()
# model = nn.DataParallel(model, device_ids=device_ids)
# print(model)
print(" ")

# load pre-trained model
saved_model = torch.load(model_file)
model.load_state_dict(saved_model)

criterion = nn.MSELoss()
index_loc = np.arange(10000)
ts_index_loc = np.arange(1000)
psnr_value = 0
# best_pic = np.zeros(shape=(256,1024)) if 'lena' in test_file or 'fake' in test_file else np.zeros(shape=(256, 256))

print('---------------------------testing---------------------------')
# model.eval()
# mse_value = torch.zeros(layers).cuda()
if objective == 'NMSE':
    mse_z = torch.zeros(K).cuda()
    mse_e = torch.zeros(K).cuda()
elif objective == 'L1L1':
    l1l1_values = torch.zeros(K).cuda()
elif objective == 'S-L2':
    sl2_values = torch.zeros(K).cuda()
else:
    raise NotImplementedError('objective `{}` not supported'.format(objective))

if use_learned and use_safeguard:
    sg_count = torch.zeros(K).cuda()

for j in range(n_test//batch_size):
    input_bs = X_ts[:, j*batch_size:(j+1)*batch_size]
    input_bs = torch.from_numpy(input_bs)
    # input_bs_var = torch.autograd.Variable(input_bs.cuda())
    input_bs_var = input_bs.cuda()
    if use_learned and use_safeguard:
        Z, E, L, T, count = model(input_bs_var)
    else:
        Z, E, L, T = model(input_bs_var)

    Z_label_bs = torch.from_numpy(Z_ts[:, j*batch_size:(j+1)*batch_size]).cuda()
    E_label_bs = torch.from_numpy(E_ts[:, j*batch_size:(j+1)*batch_size]).cuda()

    for jj in range(K):
        # mse_value[jj] = mse_value[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, Z[jj]), reduction='elementwise_mean')
        # mse_value[jj] = mse_value[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, Z[jj]))
        # mse_value[jj] = mse_value[jj] + ((255 * input_gt_var.cuda() - 255 * torch.mm(A_tensor, Z[jj]))**2).mean()
        # mse_value[jj] = mse_value[jj] + (alpha * torch.mean(torch.abs(Z[jj])) + torch.mean(torch.abs(E[jj])))
        if objective == 'NMSE':
            mse_z[jj] = mse_z[jj] + torch.sum((Z_label_bs - Z[jj])**2.0)
            mse_e[jj] = mse_e[jj] + torch.sum((E_label_bs - E[jj])**2.0)
        elif objective == 'L1L1':
            l1l1_values[jj] = l1l1_values[jj] + (
                alpha * torch.sum(torch.abs(Z[jj])) +
                torch.sum(torch.abs(input_bs_var - torch.mm(A_tensor, Z[jj])))
            )
        elif objective == 'S-L2':
            # S_0 = self.S(self.Z0, self.E0, self.L0, T[-1], X, self.E0)
            Ep = model.E0 if jj == 0 else E[jj-1]
            Sjj = model.S(Z[jj], E[jj], L[jj], T[jj], input_bs_var, Ep)
            sl2_values[jj] = sl2_values[jj] + model.two_norm(Sjj).sum()
        if use_learned and use_safeguard:
            sg_count[jj] += count[jj]


if objective == 'NMSE':
    mse_z = mse_z / n_test
    mse_e = mse_e / n_test
    nmse_denom_z = torch.sum(torch.from_numpy(Z_ts).cuda() ** 2.0) / n_test
    nmse_denom_e = torch.sum(torch.from_numpy(E_ts).cuda() ** 2.0) / n_test
    nmse = 10 * torch.log10(mse_z / nmse_denom_z + mse_e / nmse_denom_e)
    nmse_value = 1000.0
    for jj in range(K):
        if(nmse[jj] < nmse_value):
            nmse_value = nmse[jj].cpu().item()
    print('NMSE values:')
    for k in range(K):
        # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        print('{:.3f}'.format(nmse[k]), end=',')
    print(" ")
    print('******Best NMSE: {:.3f}'.format(nmse_value))
    print(" ")

elif objective == 'L1L1':
    l1l1_values = l1l1_values / n_test
    mse_value = 1000.0
    for jj in range(K):
        if(l1l1_values[jj] < mse_value):
            mse_value = l1l1_values[jj].cpu().item()
    print('MSE values:')
    for k in range(K):
        # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        print('{:.3f}'.format(l1l1_values[k]), end=',')
    print(" ")
    print('******Best MSE: {:.3f}'.format(mse_value))
    print(" ")

elif objective == 'S-L2':
    # S_0 = self.S(self.Z0, self.E0, self.L0, T[-1], X, self.E0)
    # Ep = model.E0 if jj == 0 else E[-1]
    # Sjj = model.S(Z[jj], E[jj], L[jj], T[-1], Ep)
    # sl2_values[jj] = sl2_values[jj] + model.two_norm(sl2_values).mean()
    print('L2 norms of S(uk):')
    for k in range(K):
        # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        print('{:.3f}'.format(sl2_values[k]), end=',')
    print(" ")
    # print('******Best values: {:.3f}'.format(mse_value))
    print(" ")

if use_learned and use_safeguard:
    sg_pct = sg_count / float(n_test)
if use_learned and use_safeguard:
    print("Sg Pcts:")
    for k in range(K):
        sg_pct = sg_count / float(n_test)
        print('{:.3f}'.format(sg_pct[k]), end=',')
    print(" ")
    print(" ")

