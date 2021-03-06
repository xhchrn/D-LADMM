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
from mu_updator import mu_updator_dict

num_iter = 200
alpha = 0.45
use_learned = True
# use_learned = False
use_safeguard = True
# use_safeguard = False
delta = 0.01
mu_k_method = 'EMA'
# mu_k_method = 'RT'
# mu_k_method = 'RM'
mu_k_parameter = 0.3
# mu_k_parameter = 3
# test_file = './lena_pepper_01.mat'
test_file = './lena_pepper_08.mat'
# test_file = './lena_gaussian_40.mat'
# test_file = './fake_image_pepper_02.mat'
# model_file = './DLADMMNet_chest.pth.tar'
model_file = './DLADMMNet'
D_file = './lena_pepper_01.mat'
# D_file = './D_mu5e-3.mat'
# D_file = './fake_image_pepper_02.mat'
# test_file = './brain_pepper_01.mat'
im_size = 512 if 'lena' in test_file or 'fake' in test_file else 256
layers = 15
K = layers if use_learned or use_safeguard else num_iter

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
        self.fc = nn.ModuleList()

        for k in range(self.layers):
            self.beta1.append(nn.Parameter(torch.ones(self.m, self.batch_size, dtype=torch.float32)))
            self.beta2.append(nn.Parameter(torch.ones(self.m, self.batch_size, dtype=torch.float32)))
            self.fc.append(nn.Linear(self.m, self.d, bias = False))


        self.active_para = torch.tensor(0.025, dtype=torch.float32).cuda()
        self.active_para1 = torch.tensor(0.06, dtype=torch.float32).cuda()



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
        # The following line is used in some situations for numerical
        # stability in computations that include division by the norm.
        #norm_array[norm_array < np.power(10.0,-10)] = np.power(10.0,10)
        return norm_array

    def S(self, Zk, Ek, Lk, Tk, X, **kwargs):
        beta = 1.0 / self.L.sqrt()
        ss1 = 1.0 / self.L.sqrt()
        ss2 = 0.5

        Varnew = Lk + beta * Tk
        Znew = self.self_active(Zk - ss1 * self.At.mm(Varnew), ss1 * alpha)

        # NOTE: Enew step is different from D-LADMM implementation
        TTnew = self.A.mm(Znew) + Ek - X
        Enew = self.self_active(Ek - ss2 * (Lk + beta * TTnew), ss2)

        Tnew = self.A.mm(Znew) + Enew - X
        Lnew = Lk + beta * Tnew

        return Varnew, Znew, Enew, Tnew, Lnew

    def forward(self, x):
        # x in shape (d, bs)
        X = x
        T = list()
        Var = list()
        Z = list()
        E = list()
        L = list()
        T.append(self.A.mm(self.Z0) + self.E0 - X)

        if use_learned and use_safeguard:
            # NOTE: only take Tnew for safegaurding
            mu_k = self.two_norm(self.S(self.Z0, self.E0, self.L0, T[-1], X)[-2])
            mu_updator = mu_updator_dict[mu_k_method](mu_k, mu_k_parameter)
            sg_count = np.zeros(self.layers)

        for k in range(K):
            if k == 0 :
                # Classic algorithm
                Varnew_KM, Znew_KM, Enew_KM, Tnew_KM, Lnew_KM = self.S(self.Z0, self.E0, self.L0, T[-1], X)
                # L2O algorithm
                if use_learned:
                    # Tself.A.mm(self.Z0) + self.E0 - X)
                    Varnew_L2O = self.L0 + self.beta1[k].mul(T[-1])
                    Znew_L2O = self.self_active(self.Z0 - self.fc[k](Varnew_L2O.t()).t(), self.active_para)
                    Enew_L2O = self.self_active(X - self.A.mm(Znew_L2O) - self.beta2[k].mul(self.L0), self.active_para1)
                    Tnew_L2O = self.A.mm(Znew_L2O) + Enew_L2O - X
                    Lnew_L2O = self.L0 + self.beta1[k].mul(Tnew_L2O)

            else :
                # Classic algorithm
                Varnew_KM, Znew_KM, Enew_KM, Tnew_KM, Lnew_KM = self.S(Z[-1], E[-1], L[-1], T[-1], X)
                # L2O algorithm
                if use_learned:
                    Varnew_L2O = L[-1] + self.beta1[k].mul(T[-1])
                    Znew_L2O = self.self_active(Z[-1] - self.fc[k](Varnew_L2O.t()).t(), self.active_para)
                    Enew_L2O = self.self_active(X - self.A.mm(Znew_L2O) - self.beta2[k].mul(L[-1]), self.active_para1)
                    Tnew_L2O = self.A.mm(Znew_L2O) + Enew_L2O - X
                    Lnew_L2O = L[-1] + self.beta1[k].mul(Tnew_L2O)

            if use_safeguard:
                # L2O + safegaurding
                assert use_learned
                _, _, _, Sv_L2O, _  = self.S(Znew_L2O, Enew_L2O, Lnew_L2O, Tnew_L2O, X)
                Sv_L2O_norm         = self.two_norm(Sv_L2O)
                print(Sv_L2O_norm)
                print(mu_k)
                print(" ")
                bool_term           = (Sv_L2O_norm < (1.0-delta) * mu_k).float()
                mu_k                = mu_updator.step(Sv_L2O_norm, bool_term)
                bool_term           = bool_term.reshape(1, bool_term.shape[0])
                bool_complement     = 1.0 - bool_term

                Var.append(bool_term * Varnew_L2O + bool_complement * Varnew_KM)
                Z.append(bool_term * Znew_L2O   + bool_complement * Znew_KM)
                E.append(bool_term * Enew_L2O   + bool_complement * Enew_KM)
                T.append(bool_term * Tnew_L2O   + bool_complement * Tnew_KM)
                L.append(bool_term * Lnew_L2O   + bool_complement * Lnew_KM)

                sg_count[k] = bool_complement.sum().cpu().item()

            elif use_learned:
                # Only L2O
                Var.append(Varnew_L2O)
                Z.append(Znew_L2O)
                E.append(Enew_L2O)
                T.append(Tnew_L2O)
                L.append(Lnew_L2O)

            else:
                # Classic KL algorithm
                Var.append(Varnew_KM)
                Z.append(Znew_KM)
                E.append(Enew_KM)
                T.append(Tnew_KM)
                L.append(Lnew_KM)

        if use_learned and use_safeguard:
            return Z, E, L, sg_count
        else:
            return Z, E, L


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
m, d, n = 256, 512, 10000
n_test = 1024 if 'lena' in test_file or 'fake' in test_file else 256
batch_size = 20
num_epoch = 100

use_cuda = torch.cuda.is_available()
print('==>>> use cuda' if use_cuda else '==>>> use cpu')
print('==>>> batch size: {}'.format(batch_size))
# print('==>>> total trainning batch number: {}'.format(n//batch_size))
print('==>>> total testing batch number: {}'.format(n_test//batch_size))

img_data = sio.loadmat(test_file)
D_data = sio.loadmat(D_file)
A_ori = D_data['D']
A_ori = A_ori.astype(np.float32) #*(1.0/18.0)

# X = img_data['train_x'].astype(np.float32)
# X = X.T

X_ts = img_data['test_x'].astype(np.float32)
X_ts = X_ts.T

X_gt = img_data['gt_x'].astype(np.float32)
X_gt = X_gt.T


# init parameters
Z0 = 1.0 /d * torch.rand(d, batch_size, dtype=torch.float32)
E0 = torch.zeros(m, batch_size, dtype=torch.float32)
L0 = torch.zeros(m, batch_size, dtype=torch.float32)
A_tensor = torch.from_numpy(A_ori)

model = DLADMMNet(m=m, n=n, d=d, batch_size=batch_size, A=A_tensor, Z0=Z0, E0=E0, L0=L0, layers=layers)
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
best_pic = np.zeros(shape=(256,1024)) if 'lena' in test_file or 'fake' in test_file else np.zeros(shape=(256, 256))

print('---------------------------testing---------------------------')
# model.eval()
loss = torch.zeros(K).cuda()
mse_value = torch.zeros(K).cuda()
if use_learned and use_safeguard:
    sg_count = torch.zeros(K).cuda()
for j in range(n_test//batch_size):
    input_bs = X_ts[:, j*batch_size:(j+1)*batch_size]
    input_bs = torch.from_numpy(input_bs)
    # input_bs_var = torch.autograd.Variable(input_bs.cuda())
    input_bs_var = input_bs.cuda()
    if use_learned and use_safeguard:
        Z, E, L, count = model(input_bs_var)
    else:
        [Z, E, L] = model(input_bs_var)

    input_gt = X_gt[:, j*batch_size:(j+1)*batch_size]
    input_gt = torch.from_numpy(input_gt)
    # input_gt_var = torch.autograd.Variable(input_gt.cuda())
    input_gt_var = input_gt.cuda()

    for jj in range(K):
        # loss[jj] = alpha * torch.mean(torch.abs(Z[k])) + torch.mean(torch.abs(E[k])) # + torch.mean(dual_gap(torch.mm(A_tensor.t(), L[k]), alpha)) + torch.mean(dual_gap(L[k], 1)) + torch.mean(L[k] * input_bs_var))
        loss[jj] += alpha * Z[jj].abs().sum(dim=0).mean(0) + (input_bs_var - model.A.mm(Z[jj])).abs().sum(dim=0).mean() # + torch.mean(dual_gap(torch.mm(A_tensor.t(), L[k]), alpha)) + torch.mean(dual_gap(L[k], 1)) + torch.mean(L[k] * input_bs_var))
        mse_value[jj] = mse_value[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, Z[jj]))
        if use_learned and use_safeguard:
            sg_count[jj] += count[jj]

loss = loss / (n_test // batch_size)
mse_value = mse_value / (n_test//batch_size)
psnr = -10 * torch.log10(mse_value) + torch.tensor(48.131).cuda()
if use_learned and use_safeguard:
    sg_pct = sg_count / float(n_test)
for jj in range(K):
    if(psnr_value < psnr[jj]):
        psnr_value = psnr[jj].cpu().item()
        for jjj in range(n_test//batch_size):
            input_bs = X_ts[:, jjj*batch_size:(jjj+1)*batch_size]
            input_bs = torch.from_numpy(input_bs)
            # input_bs_var = torch.autograd.Variable(input_bs.cuda())
            input_bs_var = input_bs.cuda()
            if use_learned and use_safeguard:
                Z, E, L, count = model(input_bs_var)
            else:
                [Z, E, L] = model(input_bs_var)
            best_pic[:, jjj*batch_size:(jjj+1)*batch_size] = (255* torch.mm(A_tensor, Z[jj])).cpu().data.numpy()
# print('==>>> epoch: {}, psnr1: {:.6f}'.format(epoch, psnr[0]))
# print('==>> epoch: {}'.format(epoch))
for k in range(K):
    print('Loss{}:{:.3f}'.format(k+1, loss[k]), end=' ')
print(" ")
for k in range(K):
    print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
print(" ")
if use_learned and use_safeguard:
    for k in range(K):
        sg_pct = sg_count / float(n_test)
        print('SG Pct {}:{:.3f}'.format(k+1, sg_pct[k]), end=' ')
    print(" ")
print('******Best PSNR:{:.3f}'.format(psnr_value))

# save recovered image
img = trans2image(best_pic)
scipy.misc.imsave('test_lena_01.jpg', img)

