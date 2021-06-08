import numpy as np
from numpy import linalg as la
import logging
from typing import Callable, List, Tuple, Dict

def random_sample(bounds: np.ndarray, k: int) -> np.ndarray:
    """
    Generate a set of k n-dimensional points sampled uniformly at random
    :param bounds: n x 2 dimenional array containing upper/lower bounds for each dimension
    :param k: number of samples
    :return: k x n array containing the sampled points
    """
    
    # k: Number of points
    n = len(bounds)  # Dimensionality of each point
    X = np.zeros((k, n))
    for i in range(n):
        X[:, i] = np.random.uniform(bounds[i][0], bounds[i][1], k)

    return X

def static_sample(bounds: np.ndarray) -> np.ndarray:
    '''
    Generate two locations outside the bounds
    
    :param bounds: n x 2 dimenional array containing upper/lower bounds for each dimension
    :return: 2 x n array containing the sampled points
    '''
    n = len(bounds)  # Dimensionality of each point
    X = np.zeros((2, n))
    for i in range(n):
        X[0, i] = bounds[i][1] + 5*(bounds[i][1]-bounds[i][0])
        X[1, i] = bounds[i][0] - 5*(bounds[i][1]-bounds[i][0])
    return X

def grid_sample(n: int) -> np.ndarray:
    '''
    Get a grid with equally distanced points
    
    :param n: dimensionality of the data
    :return: 2^n samples that are placed in a n dimendional grid
    '''
    if n == 1:
        ret = np.array([[0],[1]])
    if n == 2:
        ret = np.array([[0,0],[0,1], [1,0], [1,1]])
    if n == 3:
        ret = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
    if n == 4:
        ret = np.array([[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1], [0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1], [1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1], [1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]])
    if n == 5:
        ret = np.array([[0,0,0,0,0], [0,0,0,0,1], [0,0,0,1,0], [0,0,0,1,1], [0,0,1,0,0], [0,0,1,0,1], [0,0,1,1,0], [0,0,1,1,1], [0,1,0,0,0], [0,1,0,0,1], [0,1,0,1,0], [0,1,0,1,1], [0,1,1,0,0], [0,1,1,0,1], [0,1,1,1,0], [0,1,1,1,1], \
                        [1,0,0,0,0], [1,0,0,0,1], [1,0,0,1,0], [1,0,0,1,1], [1,0,1,0,0], [1,0,1,0,1], [1,0,1,1,0], [1,0,1,1,1], [1,1,0,0,0], [1,1,0,0,1], [1,1,0,1,0], [1,1,0,1,1], [1,1,1,0,0], [1,1,1,0,1], [1,1,1,1,0], [1,1,1,1,1]])
    return (ret)*(0.5)+0.25

def give_comparisons(func: Callable, points: np.ndarray, si: int=0) -> List[List[Tuple[int, int]]]:
    """
    Evaluates the function in given batch and returns the batch winner in a form accepted by the inference algorithms
    
    :param func: the black box objective
    :param points: batch locations
    :param si: starting index of the batch (the number of points evaluated so far)
    :return: Compar
    """
    ind = np.random.permutation(points.shape[0])
    y = func(points).reshape((-1,1))
    s1, s2 = ind[::2], ind[1::2]
    return [ [(s1[i]+si, s2[i]+si)] if y[s1[i]] < y[s2[i]] else [(s1[i]+si, s2[i]+si)] for i in range(len(s1))]


def boundit(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """
    Projects input to be within the bounds
    
    :param x: The point to be projected
    :param bounds: bounds of the hyper rectangle the point should be within
    :return: the point projected to be within the inputs
    """
    for i, bound in enumerate(bounds):
        if x[i] < bound[0]:
            x[i] = bound[0]
        elif x[i] > bound[1]:
            x[i] = bound[1]
    return x

def running_mean(x: np.ndarray, N: int) -> float:
    """
    Computes the running mean of N members for the array. The returned array is of length len(x)-N
    
    :param x: the input array for which we want to compute the running mean
    :param N: The length of the running mean window
    :return: The running mean of the array. Te returned array is of length len(x)-N
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def max_difference(xs: np.ndarray):
    """
    Computes the maximum change in the input when iterating through it and moving from the previus value to the next one
    
    :param xs: array from which the max difference is wanted to be computed
    :return: maximum difference between the proceeding indices
    """
    min_elem = xs[0]
    max_elem = xs[0]
    max_diff = -1

    for elem in xs[1:]:
        min_elem = min(elem, min_elem)
        if elem > max_elem:
            max_elem = elem
    max_diff = max(max_diff, max_elem - min_elem)
    return max_diff


def adam(objective: Callable, x0: np.ndarray, args: Dict, max_it: int=500, alpha: float=0.01,
         beta1: float=0.9, beta2: float=0.999, eps: float=1e-8, tol: float=1e-3, lag: int=100,
         bounds: np.ndarray=None, Nw: int=20, get_logger: Callable=None) -> Tuple[float, float, float]:
    """
    Runs adam optimization algorithm for the objective
    
    :param objective: The function to be optimized
    :param x0: starting location
    :param args: additional arguments for objective
    :param max_it: maximum number of iterations
    :param alpha: learning rate
    :param beta1: tuning parameter between [0,1], generally close to 1
    :param beta2: tuning parameter between [0,1], generally close to 1
    :param eps: a pparameter that helps avoid division by zero when computing the new parameter
    :param tol: tolerance the optimization is ended after which the running mean goes
    :param lag: how many iterations are waited and the change in the running mean must be less than the tolerance
    :param bounds: optimization bounds
    :param Nw: how many iterations are used t ovompute the running mean
    :param get_logger: Function for receiving the legger where the prints are forwarded.
    :return: A tuple that contain the optimized location, mean of the 5 last iterations, and the whole optimization history of points
    """
    x=x0
    c=0
    mt,vt = 0.0,0.0 #moving average of gradient and squared gradient
    fs=[]
    xs=[x0]
    if get_logger is not None:
        f, grad = objective(x,args)
        get_logger().debug("Starting from: {} with acq {}".format(x, f))
    it=0
    while(it < max_it):
        try:
            f, grad = objective(x,args)
        except:
            if get_logger is not None:
                get_logger().error("Error in objective when optimizing with Adam. Location:\n".format(x))
            break
        mt = beta1*mt + (1-beta1)*grad #update the moving averages of the gradient
        vt = beta2*vt + (1-beta2)*(grad**2) #updates the moving averages of the squared gradient
        m_cap = mt/(1-(beta1**(it+1))) #calculates the bias-corrected estimates
        v_cap = vt/(1-(beta2**(it+1))) #calculates the bias-corrected estimates
        delta = (alpha*m_cap)/(np.sqrt(v_cap)+eps)
        x = x - delta
        if bounds is not None:
            x = boundit(x, bounds)
        fs += [f]
        xs += [x]
        if (it>lag+Nw-1) and (max_difference(running_mean(fs,Nw)[-lag:]) < tol): break
        it+=1
    if get_logger is not None:
        get_logger().debug("Ended to: {} with acq {} and gradient {} in {} iterations".format(x, f, grad, it))
    fsr = np.mean(fs[-5:]) if len(fs)>4 else np.mean(fs)
    return x, fsr, xs  # return the mean of the 5 previous values to reduce stochasticity


def configure_logger(log_file=None) -> None:
    """
    Configures the file the prints are forwarded to
    
    :param log_file: The file the prints are forwarded to
    """
    if log_file is None:
        logger = logging.getLogger('logger.log') #default logger
        file_handler = logging.StreamHandler()
    else:
        logger = logging.getLogger(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)-8s \n %(message)s')
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False


def comparison_form(y: np.ndarray) -> List[Tuple[int,int]]:
    """
    Transforms the already existing observations to the comparison form
    
    :param y: function evaluations in a batch
    :return: comparison form for the outputs
    """
    m=len(y)
    assert(m>1)
    ind = np.argsort(y.flatten())
    return [(ind[0], ind[i]) for i in range(1,m)]