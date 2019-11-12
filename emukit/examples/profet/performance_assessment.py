import numpy as np
import scipy.stats as stats


def compute_runtime_feval(trajectory: np.ndarray, target: float) -> int:
    """
    Computes the runtime in terms of number of function evaluations until a performance smaller or
    equal than a specified target value is achieved.

    :param trajectory: trajectory of the performance of an optimizer after each function evaluation
    :param target: target value
    :return: runtime
    """
    rt = np.where(trajectory <= target)[0]
    if len(rt) == 0:
        rt = np.inf
        return rt
    else:
        return rt[0] + 1


def compute_ecdf(error: np.ndarray, targets: np.ndarray) -> tuple:
    """
    Computes the empirical cumulative distribution (ECDF) of the runtime of an optimizer
    across different targets and tasks.

    :param error: matrix with I x R x N entries, where I are the number of instances or tasks,
    R is the number of runs per task and N are the number of function evaluations per task and run
    :param targets: matrix with I x T entries, where I are the number of instances or tasks and T are the number
    of targets values
    :return: ECDF as tuple, where the first entries defines the runtime and the second the CDF
    """
    n_instances = error.shape[0]
    n_runs = error.shape[1]
    n_targets = targets.shape[1]

    runtime = []
    for i in range(n_instances):
        for t in range(n_targets):
            for r in range(n_runs):
                rt = compute_runtime_feval(error[i, r], targets[i, t])
                runtime.append(rt)
    sorted_error = np.sort(runtime)
    yvals = np.arange(len(sorted_error)) / float(len(sorted_error))

    return sorted_error.tolist(), yvals.tolist()


def compute_ranks(errors, n_bootstrap=1000) -> np.ndarray:
    """
    Computes the averaged ranking score in every iteration and for every task..

    :param errors: matrix with M x I x R x N entries, where M are the number of optimizers,
    I are the number of instances or tasks, R is the number of runs per task and
    N are the number of function evaluations per task and run
    :param n_bootstrap: number bootstrap samples to compute the ranks
    :return: the ranks after each iteration as a MxN ndarray, where, as for errors, M are the number of optimizers
    and N are the number of function evaluations
    """
    n_methods = errors.shape[0]
    n_instances = errors.shape[1]
    n_runs = errors.shape[2]
    n_iters = errors.shape[3]

    ranks = np.zeros([n_methods, n_iters])
    for instance_id in range(n_instances):
        for _ in range(n_bootstrap):
            runs = [np.random.randint(n_runs) for i in range(n_methods)]

            rank_samples = [stats.rankdata([errors[i, instance_id, ri, t] for i, ri in enumerate(runs)])
                            for t in range(n_iters)]
            ranks += np.array(rank_samples).T
    ranks /= n_instances * n_bootstrap

    return ranks
