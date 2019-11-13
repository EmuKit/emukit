import torch
import numpy as np


def objective_function(config, model_objective, model_cost,
                       task_feature_objective, task_feature_cost,
                       x_mean_objective, x_std_objective,
                       x_mean_cost, x_std_cost,
                       y_mean_objective=None, y_std_objective=None,
                       y_mean_cost=None, y_std_cost=None,
                       log_objective=False,
                       with_noise=True):

    Ht = np.repeat(task_feature_objective[None, :], config.shape[0], axis=0)
    x = np.concatenate((config, Ht), axis=1)
    x_norm = torch.from_numpy((x - x_mean_objective) / x_std_objective).float()
    output = model_objective.forward(x_norm).data.numpy()

    mean = output[:, 0]
    log_variance = output[:, 1]
    if y_mean_objective is not None or y_std_objective is not None:
        mean = mean * y_std_objective + y_mean_objective
        log_variance *= y_std_objective ** 2

    feval = mean
    if with_noise:
        feval += np.random.randn() * np.sqrt(np.exp(log_variance))

    if log_objective:
        feval = np.exp(feval)

    Ht = np.repeat(task_feature_cost[None, :], config.shape[0], axis=0)
    x = np.concatenate((config, Ht), axis=1)
    x_norm = torch.from_numpy((x - x_mean_cost) / x_std_cost).float()
    output = model_cost.forward(x_norm).data.numpy()

    log_mean = output[:, 0]
    log_log_variance = output[:, 1]
    if y_mean_cost is not None or y_std_cost is not None:
        log_mean = log_mean * y_std_cost + y_mean_cost
        log_log_variance *= y_std_cost ** 2

    log_cost = log_mean
    if with_noise:
        log_cost += np.random.randn() * np.sqrt(np.exp(log_log_variance))

    return feval[:, None], np.exp(log_cost)[:, None]
