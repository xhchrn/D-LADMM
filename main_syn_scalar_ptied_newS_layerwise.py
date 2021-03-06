import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import scipy.io as sio
import scipy.misc
from math import sqrt

import argparse

parser = argparse.ArgumentParser(description='Test LADMM with synthetic data')
parser.add_argument('-l', '--layers', type=int, default=20, help='number of layers')
parser.add_argument('-i', '--interval', type=int, default=10, help='interval of each fc weight used')
parser.add_argument('-lf', '--loss-fn', type=str, default='L1L1', help='loss function used for training')
parser.add_argument('-lr', '--learning-rate', type=float, default=0.005, help='initial learning rate')
parser.add_argument('-ld', '--layer-decay', type=float, default=0.5, help='layerwise lr decay rate')
parser.add_argument('-bs', '--batch-size', type=int, default=25, help='batch size for training')
parser.add_argument('-a', '--alpha', type=float, default=0.001, help='hyper-param in the objective')
parser.add_argument('-g', '--gamma', type=float, default=0.2, help='learning rate decay rate per 10 epochs')
# parser.add_argument('-m', '--mu', type=float, default=0.0, help='mu of Gaussian dist')
# parser.add_argument('-s', '--sigma', type=float, default=2.0, help='sigma of Gaussian dist')
# parser.add_argument('--num-iter', type=int, default=200, help='number of iterations for KM algorithm')
# parser.add_argument('-a', '--alpha', type=float, default=0.01, help='hyper-param in the objective')
# parser.add_argument('--continued', action='store_true', help='continue LSKM with KM')
# parser.add_argument('--data-type', type=str, default='gaussian', help='data type')

args = parser.parse_args()


## network definition
class DLADMMNet(nn.Module):
    def __init__(self, m, n, d, batch_size, A, Z0, E0, L0, layers, interval):
        super(DLADMMNet, self).__init__()
        self.m = m
        self.n = n
        self.d = d
        self.A = A.cuda()
        self.At = self.A.t()
        self.Z0 = Z0.cuda()
        self.E0 = E0.cuda()
        self.L0 = L0.cuda()
        self.layers = layers
        self.interval = interval

        self.A_np = A.cpu().numpy()
        self.L = (np.linalg.norm(np.matmul(self.A_np.transpose(), self.A_np), ord=2) * torch.ones(1,1)).float().cuda()

        self.beta1 = nn.ParameterList()
        self.beta2 = nn.ParameterList()
        self.beta3 = nn.ParameterList()
        self.ss1 = nn.ParameterList()
        self.ss2 = nn.ParameterList()
        self.active_para = nn.ParameterList()
        self.active_para1 = nn.ParameterList()
        self.fc = nn.ModuleList()

        for i in range(layers // self.interval):
            self.fc.append(nn.Linear(self.m, self.d, bias = False))

        for k in range(self.layers):
            self.beta1.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.beta2.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.beta3.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.ss1.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.ss2.append(nn.Parameter(torch.ones(1, 1, dtype=torch.float32)))
            self.active_para.append(nn.Parameter(0.01 * torch.ones(1, 1, dtype=torch.float32)))
            self.active_para1.append(nn.Parameter(0.01 * torch.ones(1, 1, dtype=torch.float32)))
            # self.fc.append(nn.Linear(self.m, self.d, bias = False))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out')
                #m.weight.data.normal_(0, 1/20)
                # m.weight = torch.nn.Parameter(self.A.t() + (1e-3)*torch.randn_like(self.A.t()))
                m.weight = torch.nn.Parameter((self.A.t() + (1e-3)*torch.randn_like(self.A.t())) * 0.4)
                #m.weight = torch.nn.Parameter(self.A.t())


    def self_active(self, x, thershold):
        return F.relu(x - thershold) - F.relu(-1.0 * x - thershold)


    def forward(self, x, K):
        X = x
        Z = list()
        E = list()
        L = list()

        K = min(K, self.layers)
        for k in range(K):
            fc = self.fc[k // self.interval]
            if k == 0 :
                E.append(self.E0)
                L.append(self.L0)
                # Z step:
                Tn = self.A.mm(self.Z0) + self.E0 - X
                Varn = self.L0 + self.beta1[k].mul(Tn)
                Z.append(self.self_active(
                    self.Z0 - self.ss1[k] * fc(Varn.t()).t(), self.active_para[k]))

            else :
                # E step:
                VVar = L[-1] + self.beta2[k-1] * (self.A.mm(Z[-1]) + E[-1] - X)
                E.append(self.self_active(
                    E[-1] - self.ss2[k-1].mul(VVar), self.active_para1[k-1]))
                # L step:
                Tn = self.A.mm(Z[-1]) + E[-1] - X
                L.append(L[-1] + self.beta3[k-1].mul(Tn))
                # Z step:
                Varn = L[-1] + self.beta1[k].mul(Tn)
                Z.append(self.self_active(
                    Z[-1] - self.ss1[k] * fc(Varn.t()).t(), self.active_para[k]))

        return Z, E, L

    def name(self):
        return "DLADMMNet_scalar_ptied{}_newS_layerwise".format(self.interval)


if __name__ == '__main__':
    np.random.seed(1126)
    device_ids = list(range(len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))))
    m, d, n = 250, 500, 10000
    n_test = 1000
    batch_size = args.batch_size
    layers = args.layers
    interval = args.interval
    alpha = args.alpha
    loss_fn = args.loss_fn
    learning_rate = args.learning_rate
    gamma = args.gamma
    layer_decay = args.layer_decay
    max_epochs = 10000
    num_stages = 3
    best_wait = 5
    lam = 0.0001 # lambda that reweiht the L1-L1 objective and squared L2 norm of S operator

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

    if loss_fn.lower() == 'gt':
        label = np.load(os.path.join(
            'cvx-solutions',
            'syn_data_p0.1_mu0.0_s1.0-dual-alpha{}-eps0.0001-train.npz'.format(alpha)))
        Z_label = label['Z']
        E_label = label['E']
        L_label = label['L']

    # init parameters
    Z0 = 1.0 /d * torch.rand(d, batch_size, dtype=torch.float32)
    E0 = torch.zeros(m, batch_size, dtype=torch.float32)
    L0 = torch.zeros(m, batch_size, dtype=torch.float32)
    A_tensor = torch.from_numpy(A_ori)

    model = DLADMMNet(
        m=m, n=n, d=d, batch_size=batch_size, A=A_tensor,
        Z0=Z0, E0=E0, L0=L0, layers=layers, interval=interval)
    A_tensor = A_tensor.cuda()
    if use_cuda:
        model = model.cuda()
    print(model)

    index_loc = np.arange(10000)
    ts_index_loc = np.arange(1000)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # layerwise training
    for l in range(1, layers+1):
        print('\n================ training layer {}'.format(l))

        for s in range(num_stages):
            lr = learning_rate * gamma**s
            optimizer = optim.Adam(model.parameters(), lr=lr)
            best_l1l1_value = 1000.0
            best_epoch = -1
            ld = layer_decay if s > 0 else 0.0

            print('---- training layer {} stage {} with learning rate {}'.format(l, s+1, lr))
            for epoch in range(max_epochs):
                np.random.shuffle(index_loc)
                for j in range(n//batch_size):
                    optimizer.zero_grad()
                    address = index_loc[np.arange(j*batch_size,(j+1)*batch_size)]
                    input_bs = X[:, address]
                    input_bs = torch.from_numpy(input_bs)
                    input_bs_var = input_bs.cuda()
                    Z, E, L = model(input_bs_var, l)

                    if loss_fn.lower() == 'l1l1':
                        loss = (
                            alpha * torch.sum(torch.abs(Z[-1]), dim=0).mean() +
                            torch.sum(torch.abs(input_bs_var - torch.mm(A_tensor, Z[-1])), dim=0).mean()
                        )
                    elif loss_fn.lower() == 'gt':
                        Z_label_bs = torch.from_numpy(Z_label[:, address]).cuda()
                        E_label_bs = torch.from_numpy(E_label[:, address]).cuda()
                        loss = (
                            torch.sum((Z[-1] - Z_label_bs)**2.0, dim=0).mean() +
                            torch.sum((E[-1] - E_label_bs)**2.0, dim=0).mean()
                        )
                    else:
                        raise NotImplementedError('Specified loss function {} not implemented yet'.format(loss_fn))

                    loss.backward()
                    # gradient post-processing
                    if l > 1:
                        if s == 0:
                            i_interval = (l-1) // interval
                            for ii in range(i_interval):
                                model.fc[ii].weight.grad *= 0.0
                            if (l-1) % interval > 0:
                                model.fc[i_interval].weight.grad *= 0.0
                        # else:
                        #     model.fc.weight.grad *= layer_decay ** (l - 1)
                        for k in range(l-2,-1,-1):
                            # k = l-2, ..., 0
                            # print(model.beta1[k].grad, model.ss1[k].grad, model.active_para[k].grad)
                            model.beta1[k].grad *= ld ** (l - k - 1) # (l-2) - k + 1
                            model.ss1[k].grad *= ld ** (l - k - 1)
                            model.active_para[k].grad *= ld ** (l - k - 1)
                            if k < l - 2:
                                model.beta2[k].grad *= ld ** (l - k - 2)
                                model.beta3[k].grad *= ld ** (l - k - 2)
                                model.ss2[k].grad *= ld ** (l - k - 2)
                                model.active_para1[k].grad *= ld ** (l - k - 2)
                    optimizer.step()

                l1l1_value = 0.0
                for j in range(n_test//batch_size):
                    input_bs = X_ts[:, j*batch_size:(j+1)*batch_size]
                    input_bs = torch.from_numpy(input_bs)
                    input_bs_var = input_bs.cuda()
                    with torch.no_grad():
                        Z, E, L = model(input_bs_var, l)

                        l1l1_value += (
                            alpha * torch.sum(torch.abs(Z[-1])) +
                            torch.sum(torch.abs(input_bs_var - torch.mm(A_tensor, Z[-1])))
                        ).detach().cpu().item()

                l1l1_value /= n_test
                if l1l1_value < best_l1l1_value:
                    best_l1l1_value = l1l1_value
                    best_epoch = epoch
                if epoch % 50 == 0:
                    print(
                        '==>> layer {layer}\tstage {stage}\tepoch {epoch:4d}\t'
                        'obj = {obj:.6f} (best obj = {best_obj:.6f} at epoch {best_epoch:4d})'.format(
                            layer=l, stage=s+1, epoch=epoch+1,
                            obj=l1l1_value, best_obj=best_l1l1_value, best_epoch=best_epoch+1)
                    )
                if epoch > best_epoch + best_wait:
                    print(
                        '==>> layer {layer}\tstage {stage}\tepoch {epoch:4d}\t'
                        'obj = {obj:.6f} (best obj = {best_obj:.6f} at epoch {best_epoch:4d})'.format(
                            layer=l, stage=s+1, epoch=epoch+1,
                            obj=l1l1_value, best_obj=best_l1l1_value, best_epoch=best_epoch+1)
                    )
                    break

            torch.save(
                model.state_dict(),
                '{}_layers{}_{}_alpha{}_ls{}_bs{}_gamma{}_ld{}.pth'.format(
                    model.name(), layers, loss_fn.lower(), alpha, learning_rate, batch_size, gamma, layer_decay))

            torch.cuda.empty_cache()

