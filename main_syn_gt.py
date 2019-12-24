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



    def forward(self, x):
        #X = x.view(-1, 28*28)

        X = x

        T = list()
        Var = list()
        Z = list()
        E = list()
        L = list()

        for k in range(self.layers):
            if k == 0 :
                T.append(self.A.mm(self.Z0) + self.E0 - X)
                Var.append(self.L0 + self.beta1[k].mul(T[-1]))
                Z.append(self.self_active(self.Z0 - self.fc[k](Var[-1].t()).t(), self.active_para))
                E.append(self.self_active(X - self.A.mm(Z[-1]) - self.beta2[k].mul(self.L0), self.active_para1))
                T.append(self.A.mm(Z[-1]) + E[-1] - X)
                L.append(self.L0 + self.beta1[k].mul(T[-1]))

                # T1 = self.A.mm(self.Z0) + self.E0 - X
                # Var1 = self.L0 + self.beta1_1.mul(T1)
                # Z1 = self.self_active(self.Z0 - self.fc1(Var1.t()).t(), self.active_para)
                # E1 = self.self_active(X - self.A.mm(Z1) - self.beta1_2.mul(self.L0), self.active_para1)
                # T2 = self.A.mm(Z1) + E1 - X
                # L1 = self.L0 + self.beta1_1.mul(T2)

            else :
                Var.append(L[-1] + self.beta1[k].mul(T[-1]))
                Z.append(self.self_active(Z[-1] - self.fc[k](Var[-1].t()).t(), self.active_para))
                E.append(self.self_active(X - self.A.mm(Z[-1]) - self.beta2[k].mul(L[-1]), self.active_para1))
                T.append(self.A.mm(Z[-1]) + E[-1] - X)
                L.append(L[-1] + self.beta1[k].mul(T[-1]))

                # Z2 = self.self_active(Z1 - self.fc2(Var2.t()).t(), self.active_para)
                # E2 = self.self_active(X - self.A.mm(Z2) - self.beta2_2.mul(L1), self.active_para1)
                # T3 = self.A.mm(Z2) + E2 - X
                # L2 = L1 + self.beta2_1.mul(T3)



        return Z, E, L


    def name(self):
        return "DLADMMNet"


# other functions
def trans2image(img):
	# img 256 x 1024
	# img = img.cpu().data.numpy()
	new_img = np.zeros([512, 512])
	count = 0
	for ii in range(0, 512, 16):
		for jj in range(0, 512, 16):
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
    learning_rate =  0.01 * 0.5 ** (epoch // 30)
    print('learning rate of this epoch {:.8f}'.format(learning_rate))
    # del optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) if epoch<20 else optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    np.random.shuffle(index_loc)
    for j in range(n//batch_size):
        optimizer.zero_grad()
        address = index_loc[np.arange(j*batch_size,(j+1)*batch_size)]
        input_bs = X[:, address]
        input_bs = torch.from_numpy(input_bs)
        # input_bs_var = torch.autograd.Variable(input_bs.cuda())
        input_bs_var = input_bs.cuda()
        [Z, E, L] = model(input_bs_var)

        Z_label_bs = torch.from_numpy(Z_ts[:, address]).cuda()
        E_label_bs = torch.from_numpy(E_ts[:, address]).cuda()

        loss = list()
        total_loss = 0

        for k in range(layers):
            if k < loss_start_layer:
                loss.append(0.0)
                continue

            loss.append(
                # alpha * torch.mean(torch.abs(Z[k])) +
                # torch.mean(torch.abs(E[k]))
                torch.sum((Z[k] - Z_label_bs)**2.0) +
                torch.sum((E[k] - E_label_bs)**2.0)
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

    torch.save(model.state_dict(), model.name()+'_gt.pth')

    print('---------------------------testing---------------------------')
    # model.eval()
    # mse_value = torch.zeros(layers).cuda()
    mse_z = torch.zeros(layers).cuda()
    mse_e = torch.zeros(layers).cuda()
    for j in range(n_test//batch_size):
        input_bs = X_ts[:, j*batch_size:(j+1)*batch_size]
        input_bs = torch.from_numpy(input_bs)
        # input_bs_var = torch.autograd.Variable(input_bs.cuda())
        input_bs_var = input_bs.cuda()
        [Z, E, L] = model(input_bs_var)

        Z_label_bs = torch.from_numpy(Z_ts[:, j*batch_size:(j+1)*batch_size]).cuda()
        E_label_bs = torch.from_numpy(E_ts[:, j*batch_size:(j+1)*batch_size]).cuda()

        # input_gt = X_gt[:, j*batch_size:(j+1)*batch_size]
        # input_gt = torch.from_numpy(input_gt)
        # # input_gt_var = torch.autograd.Variable(input_gt.cuda())
        # input_gt_var = input_gt.cuda()

        for jj in range(layers):
            # mse_value[jj] = mse_value[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, Z[jj]), reduction='elementwise_mean')
            # mse_value[jj] = mse_value[jj] + F.mse_loss(255 * input_gt_var.cuda(), 255 * torch.mm(A_tensor, Z[jj]))
            # mse_value[jj] = mse_value[jj] + ((255 * input_gt_var.cuda() - 255 * torch.mm(A_tensor, Z[jj]))**2).mean()
            # mse_value[jj] = mse_value[jj] + (alpha * torch.mean(torch.abs(Z[jj])) + torch.mean(torch.abs(E[jj])))
            mse_z[jj] = mse_z[jj] + torch.sum((Z_label_bs - Z[jj])**2.0)
            mse_e[jj] = mse_e[jj] + torch.sum((E_label_bs - E[jj])**2.0)

    mse_z = mse_z / n_test
    mse_e = mse_e / n_test
    nmse_denom_z = torch.sum(Z_ts ** 2.0) / n_test
    nmse_denom_e = torch.sum(E_ts ** 2.0) / n_test
    nmse = 10 * torch.log10(mse_z / nmse_denom_z + mse_e / nmse_denom_e)
    nmse_value = 10.0
    # print(mse_value)
    # psnr = -10 * torch.log10(mse_value) + torch.tensor(48.131).cuda()
    for jj in range(layers):
        if(nmse[jj] < nmse_value):
            nmse_value = nmse[jj].cpu().item()
            # for jjj in range(n_test//batch_size):
                # input_bs = X_ts[:, jjj*batch_size:(jjj+1)*batch_size]
                # input_bs = torch.from_numpy(input_bs)
                # # input_bs_var = torch.autograd.Variable(input_bs.cuda())
                # input_bs_var = input_bs.cuda()
                # [Z, E, L] = model(input_bs_var)
                # best_pic[:, jjj*batch_size:(jjj+1)*batch_size] = (255* torch.mm(A_tensor, Z[jj])).cpu().data.numpy()
    # print('==>>> epoch: {}, nmse1: {:.6f}'.format(epoch, psnr[0]))
    print('==>> epoch: {}'.format(epoch))
    for k in range(layers):
        # print('PSNR{}:{:.3f}'.format(k+1, psnr[k]), end=' ')
        print('NMSE{}:{:.3f}'.format(k+1, nmse[k]), end=' ')
    print(" ")
    print('******Best NMSE:{:.3f}'.format(nmse_value))

    # save recovered image
    # img = trans2image(best_pic)
    # scipy.misc.imsave('lena_01.jpg', img)

    torch.cuda.empty_cache()

