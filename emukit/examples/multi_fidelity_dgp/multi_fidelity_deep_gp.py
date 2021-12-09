"""
This file contains the multi-fidelity deep Gaussian process model from:
Deep Gaussian Processes for Multi-fidelity Modeling (Kurt Cutajar, Mark Pullin, Andreas Damianou, Neil Lawrence, Javier González)

The class intended for public consumption is MultiFidelityDeepGP, which is an emukit model class.

This file requires the following packages:
- tensorflow 1.x
- gpflow 1.x
- doubly_stochastic_dgp https://github.com/ICL-SML/Doubly-Stochastic-DGP/tree/master/doubly_stochastic_dgp
"""
import logging
from typing import List, Tuple

import numpy as np
from gpflow import ParamList, autoflow, params_as_tensors, settings
from gpflow.actions import Action, Loop
from gpflow.kernels import RBF, Linear, White
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero
from gpflow.models.model import Model
from gpflow.params import DataHolder, Minibatch
from gpflow.training import AdamOptimizer

from doubly_stochastic_dgp.utils import BroadcastingLikelihood

from ...core.interfaces import IModel
from ...multi_fidelity.convert_lists_to_array import convert_x_list_to_array, convert_y_list_to_array

# Import packages that are not required by emukit and throw warning if they are not installed
try:
    import tensorflow as tf
except ImportError:
    raise ImportError('tensorflow is not installed. Please installed version 1.8 by running pip install tensorflow==1.8')

try:
    from doubly_stochastic_dgp.layers import SVGP_Layer
except ImportError:
    raise ImportError('doubly_stochastic_dgp is not installed. '
                      'Please run pip install git+https://github.com/ICL-SML/Doubly-Stochastic-DGP.git')

try:
    import gpflow
except ImportError:
    raise ImportError('gpflow is not installed. Please run pip install gpflow==1.1.1')


float_type = settings.float_type
_log = logging.getLogger(__name__)


def init_layers_mf(Y, Z, kernels, num_outputs=None, Layer=SVGP_Layer):
    """
    Creates layer objects from initial data

    :param Y: Numpy array of training targets
    :param Z: List of numpy arrays of inducing point locations for each layer
    :param kernels: List of kernels for each layer
    :param num_outputs: Number of outputs (same for each layer)
    :param Layer: The layer object to use
    :return: List of layer objects with which to build a multi-fidelity deep Gaussian process model
    """
    num_outputs = num_outputs or Y[-1].shape[1]

    layers = []
    num_layers = len(Z)

    for i in range(0, num_layers):
        layers.append(Layer(kernels[i], Z[i], num_outputs, Zero()))
    return layers


class DGP_Base(Model):
    """
    The base class for Deep Gaussian process models.

    Implements a Monte-Carlo variational bound and convenience functions.
    """

    def __init__(self, X, Y, likelihood, layers, minibatch_size=None, num_samples=1, **kwargs):
        """

        :param X: List of training inputs where each element of the list is a numpy array corresponding to the inputs of one fidelity.
        :param Y: List of training targets where each element of the list is a numpy array corresponding to the inputs of one fidelity.
        :param likelihood: gpflow likelihood object for use at the final layer
        :param layers: List of doubly_stochastic_dgp.layers.Layer objects
        :param minibatch_size: Minibatch size if using minibatch trainingz
        :param num_samples: Number of samples when propagating predictions through layers
        :param kwargs: kwarg inputs to gpflow.models.Model
        """

        Model.__init__(self, **kwargs)

        self.Y_list = Y
        self.X_list = X
        self.minibatch_size = minibatch_size

        self.num_samples = num_samples

        # This allows a training regime where the first layer is trained first by itself, then the subsequent layer
        # and so on.
        self._train_upto_fidelity = -1

        if minibatch_size:
            for i, (x, y) in enumerate(zip(X, Y)):
                setattr(self, 'num_data' + str(i), x.shape[0])
                setattr(self, 'X' + str(i), Minibatch(x, minibatch_size, seed=0))
                setattr(self, 'Y' + str(i), Minibatch(y, minibatch_size, seed=0))
        else:
            for i, (x, y) in enumerate(zip(X, Y)):
                setattr(self, 'num_data' + str(i), x.shape[0])
                setattr(self, 'X' + str(i), DataHolder(x))
                setattr(self, 'Y' + str(i), DataHolder(y))

        self.num_layers = len(layers)
        self.layers = ParamList(layers)

        self.likelihood = BroadcastingLikelihood(likelihood)

    @params_as_tensors
    def propagate(self, X, full_cov=False, S=1, zs=None):
        """
        Propagate some prediction to the final layer and return predictions at each intermediate layer

        :param X: Input(s) at which to predict at
        :param full_cov: Whether the predict with the full covariance matrix
        :param S: Number of samples to use for sampling at intermediate layers
        :param zs: ??
        :return:
        """
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1])

        Fs, Fmeans, Fvars = [], [], []

        F = sX
        zs = zs or [None, ] * len(self.layers)

        for i, (layer, z) in enumerate(zip(self.layers, zs)):
            if i == 0:
                F, Fmean, Fvar = layer.sample_from_conditional(F, z=z, full_cov=full_cov)
            else:
                '''

                KC - At all layers 1..L, the input to the next layer is original input augmented with 
                the realisation of the function at the previous layer at that input.

                '''
                F_aug = tf.concat([sX, F], 2)
                F, Fmean, Fvar = layer.sample_from_conditional(F_aug, z=z, full_cov=full_cov)

            Fs.append(F)
            Fmeans.append(Fmean)
            Fvars.append(Fvar)

        return Fs, Fmeans, Fvars

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1, fidelity=None):
        """
        Predicts from the fidelity level specified. If fidelity is not specified, return prediction at highest fidelity.

        :param X: Location at which to predict
        :param full_cov: Whether to predict full covariance matrix
        :param S: Number of samples to use for MC sampling between layers
        :param fidelity: zero based fidelity index at which to predict
        :return: (mean, variance) where each is [S, N, 1] where S is number of samples and N is number of predicted points.
        """

        if fidelity is None:
            fidelity = -1

        _, Fmeans, Fvars = self.propagate(X, full_cov=full_cov, S=S)
        return Fmeans[fidelity], Fvars[fidelity]

    def _likelihood_at_fidelity(self, Fmu, Fvar, Y, variance):
        """
        Calculate likelihood term for observations corresponding to one fidelity

        :param Fmu: Posterior mean
        :param Fvar: Posterior variance
        :param Y: training observations
        :param variance: likelihood variance
        :return:
        """
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / variance

    def E_log_p_Y(self, X_f, Y_f, fidelity=None):
        """
        Calculate the expectation of the data log likelihood under the variational distribution with MC samples

        :param X_f: Training inputs for a given
        :param Y_f:
        :param fidelity:
        :return:
        """

        Fmean, Fvar = self._build_predict(X_f, full_cov=False, S=self.num_samples, fidelity=fidelity)

        if fidelity == (self.num_layers - 1):
            """
            KC - The likelihood of the observations at the last layer is computed using the model's 'likelihood' object
            """
            var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y_f)  # S, N, D
        else:
            """
            KC - The Gaussian likelihood of the observations at the intermediate layers is computed using the noise 
            parameter pertaining to the White noise kernel.

            This assumes that a White kernel should be added to all layers except for the last!
            If no noise is desired, the variance parameter in the White kernel should be set to zero and fixed.
            """
            variance = self.layers[fidelity].kern.kernels[-1].variance

            f = lambda vars_SND, vars_ND, vars_N: self._likelihood_at_fidelity(vars_SND[0],
                                                                               vars_SND[1],
                                                                               vars_ND[0],
                                                                               vars_N)

            var_exp = f([Fmean, Fvar], [tf.expand_dims(Y_f, 0)], variance)

        return tf.reduce_mean(var_exp, 0)  # N, D

    @params_as_tensors
    def _build_likelihood(self):
        """
        ELBO calculation
        :return: MC estimate of lower bound
        """
        L = 0.
        KL = 0.
        for fidelity in range(self.num_layers):

            if (self._train_upto_fidelity != -1) and (fidelity > self._train_upto_fidelity):
                continue

            X_l = getattr(self, 'X' + str(fidelity))
            Y_l = getattr(self, 'Y' + str(fidelity))

            n_data = getattr(self, 'num_data' + str(fidelity))
            scale = tf.cast(n_data, float_type)/tf.cast(tf.shape(X_l)[0], float_type)

            L += (tf.reduce_sum(self.E_log_p_Y(X_l, Y_l, fidelity)) * scale)
            KL += tf.reduce_sum(self.layers[fidelity].KL())

        self.L = L
        self.KL = KL

        return self.L - self.KL

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples, fidelity=None):
        return self._build_predict(Xnew, full_cov=False, S=num_samples, fidelity=fidelity)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f_full_cov(self, Xnew, num_samples, fidelity=None):
        return self._build_predict(Xnew, full_cov=True, S=num_samples, fidelity=fidelity)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_all_layers_full_cov(self, Xnew, num_samples):
        return self.propagate(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        return self.likelihood.predict_mean_and_var(Fmean, Fvar)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=num_samples)
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)

    @classmethod
    def make_mf_dgp(cls, X, Y, Z, add_linear=True, minibatch_size=None):
        """
        Constructor for convenience. Constructs a mf-dgp model from training data and inducing point locations

        :param X: List of target
        :param Y:
        :param Z:
        :param add_linear:
        :return:
        """

        n_fidelities = len(X)

        Din = X[0].shape[1]
        Dout = Y[0].shape[1]

        kernels = [RBF(Din, active_dims=list(range(Din)), variance=1., lengthscales=1, ARD=True)]
        for l in range(1, n_fidelities):
            D = Din + Dout
            D_range = list(range(D))
            k_corr = RBF(Din, active_dims=D_range[:Din], lengthscales=1, variance=1.0, ARD=True)
            k_prev = RBF(Dout, active_dims=D_range[Din:], variance=1., lengthscales=1.0)
            k_in = RBF(Din, active_dims=D_range[:Din], variance=1., lengthscales=1, ARD=True)
            if add_linear:
                k_l = k_corr * (k_prev + Linear(Dout, active_dims=D_range[Din:], variance=1.)) + k_in
            else:
                k_l = k_corr * k_prev + k_in
            kernels.append(k_l)

        """
        A White noise kernel is currently expected by Mf-DGP at all layers except the last.
        In cases where no noise is desired, this should be set to 0 and fixed, as follows:

            white = White(1, variance=0.)
            white.variance.trainable = False
            kernels[i] += white
        """
        for i, kernel in enumerate(kernels[:-1]):
            kernels[i] += White(1, variance=1e-6)

        num_data = 0
        for i in range(len(X)):
            _log.info('\nData at Fidelity {}'.format(i + 1))
            _log.info('X - {}'.format(X[i].shape))
            _log.info('Y - {}'.format(Y[i].shape))
            _log.info('Z - {}'.format(Z[i].shape))
            num_data += X[i].shape[0]

        layers = init_layers_mf(Y, Z, kernels, num_outputs=Dout)

        model = DGP_Base(X, Y, Gaussian(), layers, num_samples=10, minibatch_size=minibatch_size)

        return model

    def multi_step_training(self, n_iter=5000, n_iter_2=15000):
        """
        Train with variational covariance fixed to be small first, then free up and train covariance alongside other
        parameters. Inducing point locations are fixed throughout.
        """
        for layer in self.layers[:-1]:
            layer.q_sqrt = layer.q_sqrt.value * 1e-8
            layer.q_sqrt.trainable = False
        self.layers[-1].q_sqrt = self.layers[-1].q_sqrt.value * self.Y_list[-1].var() * 0.01
        self.layers[-1].q_sqrt.trainable = False
        self.likelihood.likelihood.variance = self.Y_list[-1].var() * .01
        self.likelihood.likelihood.variance.trainable = False

        # Run with covariance fixed
        self.run_adam(3e-3, n_iter)

        # Run with covariance free
        self.likelihood.likelihood.variance.trainable = True

        for layer in self.layers:
            layer.q_sqrt.trainable = True

        self.run_adam(1e-3, n_iter_2)

    def fix_inducing_point_locations(self):
        """
        Fix all inducing point locations
        """
        for layer in self.layers:
            layer.feature.Z.trainable = False

    def run_adam(self, lr, iterations):
        adam = AdamOptimizer(lr).make_optimize_action(self)
        actions = [adam, PrintAction(self, 'MF-DGP with Adam')]
        loop = Loop(actions, stop=iterations)()
        self.anchor(self.enquire_session())


class PrintAction(Action):
    """
    For progress printing during optimization
    """

    def __init__(self, model, text):
        self.model = model
        self.text = text

    def run(self, ctx):
        if ctx.iteration % 2000 == 0:
            objective = ctx.session.run(self.model.objective)
            _log.info('ELBO {:.4f};  KL {:.4f}'.format(ctx.session.run(self.model.L), ctx.session.run(self.model.KL)))
            _log.info('{}: iteration {} objective {:.4f}'.format(self.text, ctx.iteration, objective))


class MultiFidelityDeepGP(IModel):
    """
    Inducing points are fixed for first part of optimization then freed.
    Both sets of inducing points are initialized at low fidelity training data locations.
    """

    def __init__(self, X, Y, Z=None, n_iter=5000, fix_inducing=True, multi_step_training=True, minibatch_size=None):
        self._Y = Y
        self._X = X
        self.minibatch_size = minibatch_size

        if Z is None:
            self.Z = self._make_inducing_points(X, Y)
        else:
            self.Z = Z

        self.model = self._get_model(X, Y, self.Z)
        self.name = 'mfdgp'
        self.n_fidelities = len(X)
        self.n_iter = n_iter
        self.fix_inducing = fix_inducing
        self.model.fix_inducing_point_locations()
        self.multi_step_training = multi_step_training

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        raise NotImplementedError()

    def _get_model(self, X, Y, Z):
        return DGP_Base.make_mf_dgp(X, Y, Z, minibatch_size=self.minibatch_size)

    def predict(self, X: np.array) -> Tuple[np.array, np.array]:
        # assume high fidelity only!!!!
        assert np.all(X[:, -1] == (self.n_fidelities - 1))

        x_test = X[:, :-1]
        y_m, y_v = self.model.predict_y(x_test, 250)
        y_mean_high = np.mean(y_m, axis=0).flatten()
        y_var_high = np.mean(y_v, axis=0).flatten() + np.var(y_m, axis=0).flatten()
        return y_mean_high[:, None], y_var_high[:, None]

    def optimize(self) -> None:
        """
        Optimize variational parameters alongside kernel and likelihood parameters using the following regime:
            1. Optimize the parameters while fixing the intermediate layer mean variational parameters
            2. Free the mean of the variational distribution and optimize all parameters together
        """

        if self.multi_step_training:
            _log.info('\n--- Optimization: {} ---\n'.format(self.name))
            self.model.layers[0].q_mu = self._Y[0]
            for i, layer in enumerate(self.model.layers[1:-1]):
                layer.q_mu = self._Y[i][::2]
                layer.q_mu.trainable = False

            self.model.fix_inducing_point_locations()
            self.model.multi_step_training(self.n_iter)

            for layer in self.model.layers[:-1]:
                layer.q_mu.trainable = True
            self.model.run_adam(1e-3, 15000)
        else:
            self.model.run_adam(1e-3, 20000)

    @staticmethod
    def _make_inducing_points(X: List, Y: List) -> List:
        """
        Makes inducing points at every other training point location which is deafult behaviour if no inducing point
        locations are passed

        :param X: training locations
        :param Y: training targets
        :return: List of numpy arrays containing inducing point locations
        """
        Z = [X[0].copy()]
        for x, y in zip(X[:-1], Y[:-1]):
            Z.append(np.concatenate((x.copy()[::2], y.copy()[::2]), axis=1))
        return Z

    @property
    def X(self) -> np.array:
        return convert_x_list_to_array(self._X)

    @property
    def Y(self) -> np.array:
        return convert_y_list_to_array(self._Y)
