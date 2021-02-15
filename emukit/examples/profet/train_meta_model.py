import argparse
import json
import os
import pickle
from copy import deepcopy
import numpy as np
import tarfile
import functools

from urllib.request import urlretrieve

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError('pytorch is not installed. Please install it by running pip install torch torchvision')

try:
    from pybnn.bohamiann import Bohamiann
    from pybnn.util.layers import AppendLayer
except ImportError:
    raise ImportError('pybnn is not installed. Please install it by running pip install pybnn')

try:
    import GPy
    from GPy.models import BayesianGPLVM
except ImportError:
    raise ImportError('GPy is not installed. Please install it by running pip install GPy')


from emukit.examples.profet.meta_benchmarks.architecture import get_default_architecture
from emukit.examples.profet.meta_benchmarks.meta_forrester import get_architecture_forrester


def download_data(path, source='http://www.ml4aad.org/wp-content/uploads/2019/05/profet_data.tar.gz'):

    l = urlretrieve(source)[0]

    tar = tarfile.open(l)
    tar.extractall(path=path)
    tar.close()


def load_data(path, filename):
    res = json.load(open(os.path.join(path, filename), "r"))
    return np.array(res["X"]), np.array(res["Y"]), np.array(res["C"])


def normalize_Y(Y, indexD):
    max_idx = np.max(indexD)
    Y_mean = np.zeros(max_idx + 1)
    Y_std = np.zeros(max_idx + 1)
    for i in range(max_idx + 1):
        Y_mean[i] = Y[indexD == i].mean()
        Y_std[i] = Y[indexD == i].std() + 1e-8
        Y[indexD == i] = (Y[indexD == i] - Y_mean[i]) / Y_std[i]
    return Y, Y_mean[:, None], Y_std[:, None]


hidden_space = {
    'forrester': 2,
    'fcnet': 5,
    'svm': 5,
    'xgboost': 5,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', default=1000, type=int, nargs='?',
                        help='number of samples to draw from the meta-model')
    parser.add_argument('--output_path', default="./", type=str, nargs='?',
                        help='specifies the path where the samples will be saved')
    parser.add_argument('--benchmark', default="forrester", type=str, nargs='?',
                        help='specifies the benchmark')
    parser.add_argument('--input_path', default="./", type=str, nargs='?',
                        help='path to the input data')
    parser.add_argument('--n_hidden', default=5, type=int, nargs='?',
                        help='number of hidden dimensions')
    parser.add_argument('--burnin', default=50000, type=int, nargs='?',
                        help='number of burnin steps for SGHMC')
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()

    n_samples = args.n_samples
    n_samples_task = args.n_hidden
    n_inducing_lvm = 50
    Q_h = hidden_space[args.benchmark]  # the dimensionality of the latent space
    mcmc_thining = 100
    num_burnin_steps = args.burnin
    num_steps = 100 * n_samples + 1
    lr = 1e-2
    batch_size = 5

    # load and normalize data
    if args.benchmark == "forrester":
        fname = "data_sobol_forrester.json"
        get_architecture = get_architecture_forrester
    elif args.benchmark == "svm":
        get_architecture = functools.partial(get_default_architecture, classification=True)
        fname = "data_sobol_svm.json"
    elif args.benchmark == "fcnet":
        get_architecture = functools.partial(get_default_architecture, classification=True)
        fname = "data_sobol_fcnet.json"
    elif args.benchmark == "xgboost":
        fname = "data_sobol_xgboost.json"
        get_architecture = get_default_architecture

    if args.download:
        download_data(args.input_path)

    X, Y, C = load_data(args.input_path, fname)
    if len(X.shape) == 1:
        X = X[:, None]

    n_tasks = Y.shape[0]
    n_configs = X.shape[0]
    index_task = np.repeat(np.arange(n_tasks), n_configs)
    Y_norm, _, _ = normalize_Y(deepcopy(Y.flatten()), index_task)

    # train the probabilistic encoder
    kern = GPy.kern.Matern52(Q_h, ARD=True)

    m_lvm = BayesianGPLVM(Y_norm.reshape(n_tasks, n_configs), Q_h, kernel=kern,
                          num_inducing=n_inducing_lvm)
    m_lvm.optimize(max_iters=10000, messages=1)

    ls = np.array([m_lvm.kern.lengthscale[i] for i in range(m_lvm.kern.lengthscale.shape[0])])

    # generate data to train the multi-task model
    task_features_mean = np.array(m_lvm.X.mean / ls)
    task_features_std = np.array(np.sqrt(m_lvm.X.variance) / ls)
    X_train = []
    Y_train = []
    C_train = []

    for i, xi in enumerate(X):
        for idx in range(n_tasks):
            for _ in range(n_samples_task):
                ht = task_features_mean[idx] + task_features_std[idx] * np.random.randn(ls.shape[0])

                x = np.concatenate((xi, ht), axis=0)
                X_train.append(x)
                Y_train.append(Y[idx, i])
                C_train.append(C[idx, i])

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    C_train = np.array(C_train)
    if args.benchmark != "forrester":
        C_train = np.log(C_train)

    if args.benchmark == "xgboost":
        Y_train = np.log(Y_train)

    normalize_targets = True
    if args.benchmark == "fcnet" or args.benchmark == "svm":
        normalize_targets = False

    model_objective = Bohamiann(get_network=get_architecture, print_every_n_steps=10000,
                                normalize_output=normalize_targets)
    model_objective.train(X_train, Y_train, num_steps=num_steps + num_burnin_steps,
                          num_burn_in_steps=num_burnin_steps, keep_every=mcmc_thining,
                          lr=lr, verbose=True, batch_size=batch_size)

    if args.benchmark != "forrester":
        model_cost = Bohamiann(get_network=get_default_architecture, print_every_n_steps=10000)
        model_cost.train(X_train, C_train, num_steps=num_steps + num_burnin_steps,
                         num_burn_in_steps=num_burnin_steps, keep_every=mcmc_thining,
                         lr=lr, verbose=True, batch_size=batch_size)

    # generate samples
    sampled_h = np.zeros([n_samples, ls.shape[0]])

    os.makedirs(args.output_path, exist_ok=True)

    for i in range(n_samples):
        print("Generate sample %d" % i)
        idx = np.random.randint(n_tasks)
        ht = task_features_mean[idx] + task_features_std[idx] * np.random.randn(ls.shape[0])
        sampled_h[i] = ht
        net = model_objective.get_network(X_train.shape[1])
        net.float()

        with torch.no_grad():
            weights = model_objective.sampled_weights[i]
            for parameter, sample in zip(net.parameters(), weights):
                parameter.copy_(torch.from_numpy(sample))

        data = dict()

        data["state_dict"] = net.state_dict()
        data["x_mean"] = model_objective.x_mean
        data["x_std"] = model_objective.x_std
        if args.benchmark == "forrester" or args.benchmark == "xgboost":
            data["y_mean"] = model_objective.y_mean
            data["y_std"] = model_objective.y_std

        data["task_feature"] = ht
        pickle.dump(data, open(os.path.join(args.output_path, "sample_objective_%d.pkl" % i), "wb"))

        if args.benchmark != "forrester":
            net = model_cost.get_network(X_train.shape[1])
            net.float()

            with torch.no_grad():
                weights = model_cost.sampled_weights[i]
                for parameter, sample in zip(net.parameters(), weights):
                    parameter.copy_(torch.from_numpy(sample))

            data = dict()

            data["state_dict"] = net.state_dict()
            data["x_mean"] = model_cost.x_mean
            data["x_std"] = model_cost.x_std
            data["y_mean"] = model_cost.y_mean
            data["y_std"] = model_cost.y_std

            data["task_feature"] = ht
            pickle.dump(data, open(os.path.join(args.output_path, "sample_cost_%d.pkl" % i), "wb"))
