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



## network definition
class DLADMMNet(nn.Module):
    def __init__(self, m, n, d, batch_size, A, Z0, E0, L0, layers):
        super(DLADMMNet, self).__init__()
        self.m = m
        self.n = n
        self.d = d
        self.batch_size = batch_size
        self.A = A.cuda()
        self.Z0 = Z0.cuda()
        self.E0 = E0.cuda()
        self.L0 = L0.cuda()
        self.layers = layers


        self.beta1 = nn.ParameterList()
        self.beta2 = nn.ParameterList()
        self.beta3 = nn.ParameterList()
        self.ss2 = nn.ParameterList()
        self.active_para = nn.ParameterList()
        self.active_para1 = nn.ParameterList()
        self.fc = nn.ModuleList()

        for k in range(self.layers):
            self.beta1.append(nn.Parameter(torch.ones(self.m, 1, dtype=torch.float32)))
            self.beta2.append(nn.Parameter(torch.ones(self.m, 1, dtype=torch.float32)))
            self.beta3.append(nn.Parameter(torch.ones(self.m, 1, dtype=torch.float32)))
            self.ss2.append(nn.Parameter(torch.ones(self.m, 1, dtype=torch.float32)))
            self.active_para.append(nn.Parameter(0.025 * torch.ones(self.d, 1, dtype=torch.float32)))
            self.active_para1.append(nn.Parameter(0.06 * torch.ones(self.m, 1, dtype=torch.float32)))
            self.fc.append(nn.Linear(self.m, self.d, bias = False))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                #m.weight.data.normal_(0, 1/20)
                m.weight = torch.nn.Parameter(self.A.t() + (1e-3)*torch.randn_like(self.A.t()))
                #m.weight = torch.nn.Parameter(self.A.t())


    def self_active(self, x, thershold):
        return F.relu(x - thershold) - F.relu(-1.0 * x - thershold)


    def forward(self, x):
        X = x
        T = list()
        TT = list()
        Var = list()
        Z = list()
        E = list()
        L = list()

        for k in range(self.layers):
            if k == 0 :
                # Step 1
                T.append(self.A.mm(self.Z0) + self.E0 - X)
                Var.append(self.L0 + self.beta1[k].mul(T[-1]))
                Z.append(self.self_active(self.Z0 - self.fc[k](Var[-1].t()).t(), self.active_para[k]))
                # Step 2 NOTE: my modification
                VVar = self.L0 + self.beta2[k] * (self.A.mm(Z[-1]) + self.E0 - X)
                E.append(self.self_active(self.E0 - self.ss2[k].mul(VVar), self.active_para1[k]))
                # Step 3
                T.append(self.A.mm(Z[-1]) + E[-1] - X)
                L.append(self.L0 + self.beta3[k].mul(T[-1]))

                # T1 = self.A.mm(self.Z0) + self.E0 - X
                # Var1 = self.L0 + self.beta1_1.mul(T1)
                # Z1 = self.self_active(self.Z0 - self.fc1(Var1.t()).t(), self.active_para)
                # E1 = self.self_active(X - self.A.mm(Z1) - self.beta1_2.mul(self.L0), self.active_para1)
                # T2 = self.A.mm(Z1) + E1 - X
                # L1 = self.L0 + self.beta1_1.mul(T2)

            else :
                # Step 1
                Var.append(L[-1] + self.beta1[k].mul(T[-1]))
                Z.append(self.self_active(Z[-1] - self.fc[k](Var[-1].t()).t(), self.active_para[k]))
                # Step 2 NOTE: my modification
                VVar = L[-1] + self.beta2[k] * (self.A.mm(Z[-1]) + E[-1] - X)
                E.append(self.self_active(E[-1] - self.ss2[k].mul(VVar), self.active_para1[k]))
                # Step 3
                T.append(self.A.mm(Z[-1]) + E[-1] - X)
                L.append(L[-1] + self.beta1[k].mul(T[-1]))

                # Z2 = self.self_active(Z1 - self.fc2(Var2.t()).t(), self.active_para)
                # E2 = self.self_active(X - self.A.mm(Z2) - self.beta2_2.mul(L1), self.active_para1)
                # T3 = self.A.mm(Z2) + E2 - X
                # L2 = L1 + self.beta2_1.mul(T3)



        return Z, E, L


    def name(self):
        return "DLADMMNet"


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
m, d, n = 250, 500, 10000
n_test = 1000
batch_size = 20
layers = 20
alpha = 0.5
num_epoch = 100

use_cuda = torch.cuda.is_available()
print('==>>> use cuda' if use_cuda else '==>>> use cpu')
print('==>>> batch size: {}'.format(batch_size))
print('==>>> total trainning batch number: {}'.format(n//batch_size))
print('==>>> total testing batch number: {}'.format(n_test//batch_size))

syn_data = sio.loadmat('syn_data.mat')
A_ori = syn_data['A']
A_ori = A_ori.astype(np.float32) #*(1.0/18.0)

X = syn_data['train_x'].astype(np.float32)
X = X.T

X_ts = syn_data['test_x'].astype(np.float32)
X_ts = X_ts.T

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
print(model)

criterion = nn.MSELoss()
index_loc = np.arange(10000)
ts_index_loc = np.arange(1000)
# psnr_value = 0
# best_pic = np.zeros(shape=(256,1024))
optimizer = None
# loss_start_layer = layers - 1
loss_start_layer = 0
for epoch in range(num_epoch):
    print('---------------------------training---------------------------')
    # model.train()
    learning_rate =  0.0002 * 0.5 ** (epoch // 30)
    print('learning rate of this epoch {:.8f}'.format(learning_rate))
    # del optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if epoch<20 else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    np.random.shuffle(index_loc)
    for j in range(n//batch_size):
        optimizer.zero_grad()
        address = index_loc[np.arange(j*batch_size,(j+1)*batch_size)]
        input_bs = X[:, address]
        input_bs = torch.from_numpy(input_bs)
        input_bs_var = input_bs.cuda()
        [Z, E, L] = model(input_bs_var)

        loss = list()
        total_loss = 0

        for k in range(layers):
            if k < loss_start_layer:
                loss.append(0.0)
                continue

            loss.append(
                alpha * torch.mean(torch.abs(Z[k])) +
                # torch.mean(torch.abs(E[k]))
                torch.mean(torch.abs(input_bs_var - torch.mm(A_tensor, Z[k])))
            )

            total_loss = total_loss + loss[-1]

        total_loss.backward()
        optimizer.step()
        if (j) % 100 == 0:
            # print('==>>> epoch: {},loss10: {:.6f}'.format(epoch, loss10))
            print('==>> epoch: {} [{}/{}]'.format(epoch+1, j, n//batch_size))
            for k in range(loss_start_layer, layers):
                print('loss{}:{:.3f}'.format(k + 1, loss[k]), end=' ')
            print(" ")

    # del loss, total_loss

    torch.save(model.state_dict(), model.name()+'_l1l1_ltheta.pth')

    print('---------------------------testing---------------------------')
    # model.eval()
    l1l1_values = torch.zeros(layers).cuda()
    for j in range(n_test//batch_size):
        input_bs = X_ts[:, j*batch_size:(j+1)*batch_size]
        input_bs = torch.from_numpy(input_bs)
        input_bs_var = input_bs.cuda()
        [Z, E, L] = model(input_bs_var)

        for jj in range(layers):
            ################ l1l1_values[jj] = l1l1_values[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, Z[jj]), reduction='elementwise_mean')
            ################ l1l1_values[jj] = l1l1_values[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, Z[jj]))
            ################ l1l1_values[jj] = l1l1_values[jj] + ((255 * input_gt_var.cuda() - 255 * torch.mm(A_tensor, Z[jj]))**2).mean()
            ################ l1l1_values[jj] = l1l1_values[jj] + (alpha * torch.mean(torch.abs(Z[jj])) + torch.mean(torch.abs(E[jj])))
            # NOTE: TODO: Add normalization after talking with Howard
            l1l1_values[jj] += alpha * torch.sum(torch.abs(Z[jj])) + \
                torch.sum(torch.abs(input_bs_var - torch.mm(A_tensor, Z[jj])))

    l1l1_values = l1l1_values / n_test
    print('==>> epoch: {}'.format(epoch))
    for k in range(layers):
        # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        print('Loss{}:{:.3f}'.format(k+1, l1l1_values[k]), end=' ')
    print(" ")

    torch.cuda.empty_cache()

