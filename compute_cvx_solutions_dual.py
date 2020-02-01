import os
import numpy as np
import cvxpy as cp
import scipy.io as sio
import scipy.misc

from utils import setup_logger

import argparse

parser = argparse.ArgumentParser(description='Test LADMM with synthetic data')
parser.add_argument('-c', '--cols', type=int, default=0, help='number of columns in A to be replaced')
parser.add_argument('-p', '--p', type=float, default=0.2, help='p in the Bernoulli distribution')
parser.add_argument('-m', '--mu', type=float, default=0.0, help='mu of Gaussian dist')
parser.add_argument('-s', '--sigma', type=float, default=2.0, help='sigma of Gaussian dist')
parser.add_argument('--data-type', type=str, default='gaussian', help='data type')
parser.add_argument('-a', '--alpha', type=float, default=0.01, help='hyper-param in the objective')
parser.add_argument('-e', '--epsilon', type=float, default=1e-4, help='hyper-param in the objective')
parser.add_argument('--split', type=str, default='test', help='calculate train or test split')
parser.add_argument('--batch-size', type=int, default=20, help='batch size')

def loss_l1(X):
    return cp.sum(cp.abs(X))

def loss_l2(X):
    return cp.sum(X**2)

def augmented_objective_fn(Z, E, alpha, eps):
    Z_part = alpha * (loss_l1(Z) + eps / 2.0 * loss_l2(Z))
    E_part = loss_l1(E) + eps / 2.0 * loss_l2(E)
    return (Z_part + E_part) / Z.shape[1]

if __name__ == '__main__':
    args = parser.parse_args()

    alpha = args.alpha
    eps = args.epsilon
    split = args.split
    batch_size = args.batch_size

    # test data file
    test_file = 'syn_data'
    test_file += '_cols{}'.format(args.cols) if args.cols > 0 else ''
    test_file += '_p{}_mu{}_s{}'.format(args.p, args.mu, args.sigma)
    test_file += '_{}'.format(args.data_type) if args.data_type != 'gaussian' else ''
    test_file += '.mat'
    print('using testing data file {}'.format(test_file))

    # logger file
    if not os.path.isdir('cvx-solutions'):
        os.makedirs('cvx-solutions')
    if not os.path.isdir('cvx-solutions/logs'):
        os.makedirs('cvx-solutions/logs')
    save_file = os.path.join(
        'cvx-solutions',
        '{}-dual-alpha{}-eps{}-{}.npz'.format(test_file[:-4], alpha, eps, split))
    log_file = os.path.join(
        'cvx-solutions/logs',
        '{}-dual-alpha{}-eps{}-{}.log'.format(test_file[:-4], alpha, eps, split))
    print = setup_logger(log_file)

    syn_data = sio.loadmat(test_file)
    A = syn_data['A'].astype(np.float32)
    m, n = A.shape

    X = syn_data[split + '_x'].astype(np.float32).T # (m, #samples)
    Z = syn_data[split + '_z'].astype(np.float32).T # (n, #samples)
    E = syn_data[split + '_e'].astype(np.float32).T # (m, #samples)
    n_samples = X.shape[1]

    Z_var = cp.Variable((n,batch_size))
    E_var = cp.Variable((m,batch_size))
    X_param = cp.Parameter((m,batch_size))
    objective = cp.Minimize(augmented_objective_fn(Z_var, E_var, alpha, eps))
    constraints = [cp.matmul(A, Z_var) + E_var == X_param]
    problem = cp.Problem(objective, constraints)

    Z_sol = np.zeros(Z.shape, dtype=np.float32)
    E_sol = np.zeros(E.shape, dtype=np.float32)
    L_sol = np.zeros((m, E.shape[1]), dtype=np.float32)

    for i in range(n_samples // batch_size):

        X_param.value = X[:, i*batch_size:(i+1)*batch_size]
        out = problem.solve()
        print('[{:2d}/{:2d}]\t{}'.format(i+1, n_samples//batch_size, out))
        Z_sol[:, i*batch_size:(i+1)*batch_size] = Z_var.value
        E_sol[:, i*batch_size:(i+1)*batch_size] = E_var.value
        L_sol[:, i*batch_size:(i+1)*batch_size] = constraints[0].dual_value

    np.savez(save_file, Z=Z_sol, E=E_sol, L=L_sol)
    print('Solutions saved to file {}'.format(save_file))

