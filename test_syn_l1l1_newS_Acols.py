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
from math import sqrt
from utils import setup_logger, my_str

import argparse

parser = argparse.ArgumentParser(description='Test LADMM with synthetic data')
parser.add_argument('--use-learned', action='store_true', help='use learned model')
parser.add_argument('--use-safeguard', action='store_true', help='use safeguarding')
parser.add_argument('--delta', type=float, default=-99.0, help='delta in safeguarding')
parser.add_argument('--mu-k-method', type=str, default='None', help='mu_k update method')
parser.add_argument('--mu-k-param', type=float, default=0.0, help='mu_k update parameter')
# parser.add_argument('--test-file', type=str, default=None, help='data used for testing')
parser.add_argument('--model-file', type=str, default='DLADMMNet.pth', help='L2O model to be loaded and tested')
parser.add_argument('--layers', type=int, default=20, help='number of layers of the L2O model')
parser.add_argument('--objective', type=str, default='NMSE', help='objective for observations')
parser.add_argument('--cols', type=int, default=20, help='num of changed columns in A')
parser.add_argument('-p', '--p', type=float, default=0.2, help='p in the Bernoulli distribution')
parser.add_argument('-m', '--mu', type=float, default=0.0, help='mu of Gaussian dist')
parser.add_argument('-s', '--sigma', type=float, default=2.0, help='sigma of Gaussian dist')
parser.add_argument('--num-iter', type=int, default=200, help='number of iterations for KM algorithm')
parser.add_argument('-a', '--alpha', type=float, default=0.01, help='hyper-param in the objective')
parser.add_argument('--continued', action='store_true', help='continue LSKM with KM')
parser.add_argument('--data-type', type=str, default='gaussian', help='data type')

args = parser.parse_args()

use_learned = args.use_learned
use_safeguard = args.use_safeguard
continued = args.continued
num_iter = args.num_iter
alpha = args.alpha
delta = args.delta
mu_k_method = args.mu_k_method
mu_k_param  = args.mu_k_param
# test_file = 'syn_data_p{}_s{}.mat'.format(args.p, args.sigma) if args.mu == 0.0 \
    # else 'syn_data_p{}_mu{}_s{}.mat'.format(args.p, args.mu, args.sigma)
test_file = 'syn_data_cols{}_p{}_mu{}_s{}.mat'.format(args.cols, args.p, args.mu, args.sigma) if args.data_type == 'gaussian' \
    else 'syn_data_cols{}_p{}_mu{}_s{}_{}.mat'.format(args.cols, args.p, args.mu, args.sigma, args.data_type)
print('using testing data file {}'.format(test_file))
model_file = args.model_file
layers = args.layers
K = layers if (not continued and (use_learned or use_safeguard)) else num_iter
objective = args.objective

# logger file
if not os.path.isdir('newS-test-logs'):
    os.makedirs('newS-test-logs')
log_file = os.path.join(
    'newS-test-logs',
    'scalar-{obj}-results-{alg}{continued}-cols{cols}-p{p}-mu{mu}-sigma{sigma}-{data_type}-{delta}{method}{param}.txt'.format(
        obj = objective.lower(),
        alg = 'lskm' if use_learned else 'km{}'.format(num_iter),
        continued = '-continued{}'.format(num_iter) if continued else '',
        cols=args.cols, p = args.p, mu=args.mu, sigma = args.sigma, data_type = args.data_type,
        delta = 'delta{}-'.format(delta) if use_learned and use_safeguard else '',
        method = mu_k_method,
        param = '' if mu_k_method == 'None' else mu_k_param
    )
)
print = setup_logger(log_file)

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
        self.beta3 = nn.ParameterList()
        self.ss2 = nn.ParameterList()
        self.active_para = nn.ParameterList()
        self.active_para1 = nn.ParameterList()
        self.fc = nn.ModuleList()

        for k in range(self.layers):
            # self.beta1.append(nn.Parameter(torch.ones(self.m, self.batch_size, dtype=torch.float32)))
            # self.beta2.append(nn.Parameter(torch.ones(self.m, self.batch_size, dtype=torch.float32)))
            self.beta1.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.beta2.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.beta3.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.ss2.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.active_para.append(nn.Parameter(0.2 * torch.ones(1, 1, dtype=torch.float32)))
            self.active_para1.append(nn.Parameter(0.8 * torch.ones(1, 1, dtype=torch.float32)))
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


    def two_norm(self, z, dim=0):
        norm_array = (z ** 2).sum(dim=dim).sqrt()
        return norm_array


    def KM_ZEL(self, Zk, Ek, Lk, Tk, X, **kwargs):
        beta = kwargs.get('beta', 1.0)
        ss1  = kwargs.get('ss1', 0.999 / self.L)
        ss2  = 1.0 / beta

        # NOTE: T^k = A*Z^k + E^k - X
        Varn = Lk + beta * Tk
        Zn = self.self_active(Zk - ss1 * self.At.mm(Varn), ss1 * alpha)

        En = self.self_active(X - self.A.mm(Zn) - ss2 * Lk, ss2)

        # NOTE: T^{k+1} = A*Z^{k+1} + E^{k+1} - X
        Tn = self.A.mm(Zn) + En - X
        Ln = Lk + beta * Tn

        return Varn, Zn, En, Tn, Ln


    def KM_ELZ(self, Ek, Lk, Zn, X, **kwargs):
        beta = kwargs.get('beta', 1.0)
        ss1  = kwargs.get('ss1', 0.999 / self.L)
        ss2  = 1.0 / beta

        En = self.self_active(X - self.A.mm(Zn) - ss2 * Lk, ss2)

        # NOTE: T^{k+1} = A*Z^{k+1} + E^{k+1} - X
        Tn = self.A.mm(Zn) + En - X
        Ln = Lk + beta * Tn

        # NOTE: T^k = A*Z^k + E^k - X
        Varnn = Ln + beta * Tn
        Znn = self.self_active(Zn - ss1 * self.At.mm(Varnn), ss1 * alpha)

        return En, Tn, Ln, Varnn, Znn


    def Snorm_ELZ(self, Ek, Lk, Zn, X, **kwargs):
        kwargs['beta'] = 1.0
        kwargs['ss1'] = 0.999 / self.L
        ss2 = 1.0 / kwargs['beta']

        En, Tn, Ln, Varnn, Znn = self.KM_ELZ(Ek, Lk, Zn, X, **kwargs)

        squared_norm_1 = ((self.A.mm(Znn) + En - X)**2.0).sum(dim=0)

        P2 = getattr(self, 'P2', None)
        if P2 is None:
            eye = torch.eye(self.d).float().cuda()
            P2 = eye / kwargs['beta'] / kwargs['ss1'] - self.At.mm(self.A)
            setattr(self, 'P2', P2)

        temp = P2.mm(Znn - Zn)
        squared_norm_2 = ((Znn - Zn) * temp).sum(dim=0)

        return (squared_norm_1 + squared_norm_2).sqrt()


    def forward(self, x, use_learned, use_safeguard, continued, K=K):
        X = x
        Z = list()
        E = list()
        L = list()
        return_cnt = use_learned and use_safeguard

        if use_learned and use_safeguard:
            mu_k = self.Snorm_ELZ(self.E0, self.L0, self.Z0, X)
            mu_updater = mu_updater_dict[mu_k_method](mu_k, mu_k_param)
            sg_count = np.zeros(self.layers)

        for k in range(K):
            if args.continued and k == layers:
                use_learned = False
                use_safeguard = False

            if k == 0 :
                # Classic algorithm
                T0 = self.A.mm(self.Z0) + self.E0 - X
                En_KM = self.E0
                Ln_KM = self.L0
                Varn_KM, Zn_KM, _, _, _ = self.KM_ZEL(self.Z0, self.E0, self.L0, T0, X)
                # L2O algorithm
                if use_learned:
                    En_L2O = self.E0
                    Ln_L2O = self.L0
                    # NOTE: `Varn` and `Zn` should be `Varnn` and `Znn`, to be strict
                    Varn_L2O = self.L0 + self.beta1[k].mul(T0)
                    Zn_L2O = self.self_active(self.Z0 - self.fc[k](Varn_L2O.t()).t(), self.active_para[k])

            else :
                # Classic algorithm
                # NOTE: `Varn_KM` and `Zn_KM` should be `Varnn_KM` and `Znn_KM`, to be strict
                En_KM, Tn_KM, Ln_KM, Varn_KM, Zn_KM = self.KM_ELZ(E[-1], L[-1], Z[-1], X)
                # L2O algorithm
                if use_learned:
                    # E step: NOTE: my modification
                    VVar = L[-1] + self.beta2[k-1] * (self.A.mm(Z[-1]) + E[-1] - X)
                    En_L2O = self.self_active(E[-1] - self.ss2[k-1].mul(VVar), self.active_para1[k-1])
                    # L step:
                    Tn_L2O = self.A.mm(Z[-1]) + En_L2O - X
                    Ln_L2O = L[-1] + self.beta3[k-1].mul(Tn_L2O)
                    # Z step:
                    # NOTE: `Varn` and `Zn` should be `Varnn` and `Znn`, to be strict
                    Varn_L2O = Ln_L2O + self.beta1[k].mul(Tn_L2O)
                    Zn_L2O = self.self_active(Z[-1] - self.fc[k](Varn_L2O.t()).t(), self.active_para[k])

            if use_safeguard:
                # L2O + safegaurding
                assert use_learned
                Snorm_L2O = self.Snorm_ELZ(En_L2O, Ln_L2O, Zn_L2O, X)
                # print(str(Snorm_L2O.mean()) + '\t' + str(mu_k.mean()))
                bool_term = (Snorm_L2O < (1.0-delta) * mu_k).float()

                mu_k            = mu_updater.step(Snorm_L2O, bool_term)
                bool_term       = bool_term.reshape(1, bool_term.shape[0])
                bool_complement = 1.0 - bool_term

                E.append  (bool_term * En_L2O   + bool_complement * En_KM)
                L.append  (bool_term * Ln_L2O   + bool_complement * Ln_KM)
                Z.append  (bool_term * Zn_L2O   + bool_complement * Zn_KM)

                sg_count[k] = bool_complement.sum().cpu().item()

            elif use_learned:
                # Only L2O
                E.append(En_L2O)
                L.append(Ln_L2O)
                Z.append(Zn_L2O)

            else:
                # Classic KL algorithm
                E.append(En_KM)
                L.append(Ln_KM)
                Z.append(Zn_KM)

        if return_cnt:
            return Z, E, L, sg_count
        else:
            return Z, E, L


    def name(self):
        return "DLADMMNet"

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
model.eval()
# mse_value = torch.zeros(layers).cuda()
if objective == 'NMSE':
    mse_z = torch.zeros(K).cuda()
    mse_e = torch.zeros(K).cuda()
elif objective == 'L1L1':
    l1l1_values = torch.zeros(K).cuda()
elif objective == 'S-L2':
    sl2_values = torch.zeros(K).cuda()
elif objective == 'Normalized-L1L1':
    normalized_l1l1_values = torch.zeros(K).cuda()
elif objective == 'GT':
    gt_values = torch.zeros(K).cuda()
elif objective == 'Normalized-GT':
    normalized_gt_values = torch.zeros(K).cuda()
else:
    raise NotImplementedError('objective `{}` not supported'.format(objective))

if use_learned and use_safeguard:
    sg_count = torch.zeros(layers).cuda()

for j in range(n_test//batch_size):
    input_bs = X_ts[:, j*batch_size:(j+1)*batch_size]
    input_bs_var = torch.from_numpy(input_bs).cuda()
    with torch.no_grad():
        if use_learned and use_safeguard:
            Z, E, L, count = model(input_bs_var, use_learned, use_safeguard, continued)
        else:
            Z, E, L = model(input_bs_var, use_learned, use_safeguard, continued)
        if 'normalized' in objective.lower() or 'gt' in objective.lower():
            Zp, Ep, Lp = model(input_bs_var, False, False, False, K=2000)

    Z_label_bs = torch.from_numpy(Z_ts[:, j*batch_size:(j+1)*batch_size]).cuda()
    E_label_bs = torch.from_numpy(E_ts[:, j*batch_size:(j+1)*batch_size]).cuda()

    for jj in range(K):

        with torch.no_grad():

            if objective == 'NMSE':
                mse_z[jj] = mse_z[jj] + torch.sum((Z_label_bs - Z[jj])**2.0)
                mse_e[jj] = mse_e[jj] + torch.sum((E_label_bs - E[jj])**2.0)

            elif objective == 'L1L1':
                l1l1_values[jj] = l1l1_values[jj] + (
                    alpha * torch.sum(torch.abs(Z[jj])) +
                    torch.sum(torch.abs(input_bs_var - torch.mm(A_tensor, Z[jj])))
                )

            elif objective == 'Normalized-L1L1':
                l1l1_value = (
                    alpha * torch.sum(torch.abs(Z[jj]), dim=0) +
                    torch.sum(torch.abs(input_bs_var - torch.mm(A_tensor, Z[jj])), dim=0)
                )
                gt_l1l1_value = (
                    alpha * torch.sum(torch.abs(Zp[-1]), dim=0) +
                    torch.sum(torch.abs(input_bs_var - torch.mm(A_tensor, Zp[-1])), dim=0)
                )
                # print(l1l1_value)
                # print(gt_l1l1_value)
                normalized_l1l1_values[jj] += (torch.abs(l1l1_value - gt_l1l1_value) / gt_l1l1_value).sum()

            elif objective == 'GT':
                gt_values[jj] += (
                    torch.sum((Z[jj] - Zp[-1])**2.0) +
                    torch.sum((E[jj] - Ep[-1])**2.0)
                )

            elif objective == 'Normalized-GT':
                gt_value = (
                    torch.sum((Z[jj] - Zp[-1])**2.0, dim=0) +
                    torch.sum((E[jj] - Ep[-1])**2.0, dim=0)
                )
                gt_norm = (
                    torch.sum(Zp[-1] ** 2.0, dim=0) +
                    torch.sum(Ep[-1] ** 2.0, dim=0)
                )
                normalized_gt_values[jj] += (gt_value / gt_norm).sum()

            elif objective == 'S-L2':
                Snorm = model.Snorm_ELZ(E[jj], L[jj], Z[jj], input_bs_var)
                sl2_values[jj] = sl2_values[jj] + Snorm.sum()

            elif objective == 'Normalized-DGAP':
                normalized_dgap = torch.zeros(K).cuda()

        if jj < layers and use_learned and use_safeguard:
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
    print(', '.join(map(my_str, nmse)))
    # for k in range(K):
        # # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        # print('{:.3f}'.format(nmse[k]), end=',')
    # print(" ")
    print('******Best NMSE: {:.3f}\n'.format(nmse_value))
    # print(" ")

elif objective == 'L1L1':
    l1l1_values = l1l1_values / n_test
    mse_value = 1000.0
    for jj in range(K):
        if(l1l1_values[jj] < mse_value):
            mse_value = l1l1_values[jj].cpu().item()
    print('MSE values:')
    print(', '.join(map(my_str, l1l1_values)))
    # for k in range(K):
        # # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        # print('{:.3f}'.format(l1l1_values[k]), end=',')
    # print(" ")
    print('******Best MSE: {:.3f}\n'.format(mse_value))
    # print(" ")

elif objective == 'Normalized-L1L1':
    normalized_l1l1_values /= n_test
    print('Normalized L1L1 values:')
    print(', '.join(map(my_str, normalized_l1l1_values)))
    # for k in range(K):
        # # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        # print('{:.3f}'.format(l1l1_values[k]), end=',')
    # print(" ")
    # print(" ")

elif objective == 'GT':
    gt_values /= n_test
    print('GT values:')
    print(', '.join(map(my_str, gt_values)))
    # for k in range(K):
        # # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        # print('{:.3f}'.format(l1l1_values[k]), end=',')
    # print(" ")
    # print(" ")

elif objective == 'Normalized-GT':
    normalized_gt_values /= n_test
    print('Normalized GT values:')
    print(', '.join(map(my_str, normalized_gt_values)))
    # for k in range(K):
        # # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        # print('{:.3f}'.format(l1l1_values[k]), end=',')
    # print(" ")
    # print(" ")

elif objective == 'S-L2':
    sl2_values /= n_test
    print('L2 norms of S(uk):')
    print(', '.join(map(my_str, sl2_values)))

if use_learned and use_safeguard:
    sg_pct = sg_count / float(n_test)
    print("Sg Pcts:")
    print(', '.join(map(my_str, sg_pct)))
    # for k in range(K):
        # sg_pct = sg_count / float(n_test)
        # print('{:.3f}'.format(sg_pct[k]), end=',')
    # print(" ")
    # print(" ")

