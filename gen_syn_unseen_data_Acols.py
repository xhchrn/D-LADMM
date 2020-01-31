import argparse
import numpy as np
import scipy.io as sio

parser = argparse.ArgumentParser(description='Test LADMM with synthetic data')
parser.add_argument('-c', type=int)
parser.add_argument('-p', type=float)
parser.add_argument('-m', type=float, default=0.0)
parser.add_argument('-s', type=float)
parser.add_argument('--A-file', type=str, default='./syn_data.mat')
args = parser.parse_args()

if __name__ == "__main__":
    np.random.seed(19950118)
    # generate A
    m = 250
    n = 500
    # A = np.random.normal(size=(m,n))
    # colnorm = np.sqrt(np.sum(A**2.0, axis=0, keepdims=True))
    # A = A / colnorm
    A = sio.loadmat(args.A_file)['A'].astype(np.float32)
    col_perm = np.random.permutation(n)
    changed_cols = col_perm[:args.c]
    new_Acols = np.random.normal(size=(m, args.c))
    new_Acols /= np.sqrt(np.sum(new_Acols**2.0, axis=0, keepdims=True))
    A[:, changed_cols] = new_Acols

    # generate Z
    train_size = 10000
    test_size = 1000
    p = args.p
    mu = args.m
    sigma = args.s
    # bernoulli
    bern_train = np.random.binomial(size=[train_size, n], n=1, p=p)
    bern_test = np.random.binomial(size=[test_size, n], n=1, p=p)
    # gaussian
    gaus_train = np.random.normal(size=[train_size, n], loc=mu, scale=sigma)
    gaus_test = np.random.normal(size=[test_size, n], loc=mu, scale=sigma)
    # z = bernoulli * gaussian
    train_z = bern_train * gaus_train
    test_z = bern_test * gaus_test

    # generate E
    # bernoulli
    bern_train = np.random.binomial(size=[train_size, m], n=1, p=p)
    bern_test = np.random.binomial(size=[test_size, m], n=1, p=p)
    # gaussian
    gaus_train = np.random.normal(size=[train_size, m], loc=mu, scale=sigma)
    gaus_test = np.random.normal(size=[test_size, m], loc=mu, scale=sigma)
    # z = bernoulli * gaussian
    train_e = bern_train * gaus_train
    test_e = bern_test * gaus_test

    # generate X with X = AZ + E
    train_x = np.transpose(np.dot(A, train_z.T) + train_e.T)
    test_x = np.transpose(np.dot(A, test_z.T) + test_e.T)

    # create dict to save
    d = dict(A       = A,
             train_x = train_x,
             test_x  = test_x,
             train_z = train_z,
             test_z  = test_z,
             train_e = train_e,
             test_e  = test_e)
    # save mat file
    sio.savemat('syn_data_cols{}_p{}_mu{}_s{}.mat'.format(args.c, p, mu, sigma), d)

