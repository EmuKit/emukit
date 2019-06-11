from typing import Union, Callable

import scipy
import numpy as np

from ...core import InformationSourceParameter
from ...core.acquisition import Acquisition
from ...core.interfaces import IModel
from ...core.parameter_space import ParameterSpace
from ...samplers import AffineInvariantEnsembleSampler, McmcSampler

from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.bayesian_optimization.interfaces import IEntropySearchModel
from emukit.bayesian_optimization import epmgp
from emukit.bayesian_optimization.acquisitions.entropy_search import EntropySearch


class ContinuousFidelityEntropySearch(EntropySearch):
    """
    Entropy search acquisition for multi-information source problems where the objective function is the output of one
    of the information sources. The other information sources provide auxiliary information about the objective function
    """
    def __init__(self, model: Union[IModel, IEntropySearchModel], space: ParameterSpace,
                 target_fidelity_index: int=None, num_samples: int = 100,
                 num_representer_points: int = 50, burn_in_steps: int = 50):
        """
        :param model: Gaussian process model of the objective function that implements IEntropySearchModel
        :param space: Parameter space of the input domain
        :param target_information_source_index: The index of the information source we want to minimise
        :param num_samples: Integer determining how many samples to draw for each candidate input
        :param num_representer_points: Integer determining how many representer points to sample
        :param burn_in_steps: Integer that defines the number of burn-in steps when sampling the representer points
        """

        # Find information source parameter in parameter space
        if target_fidelity_index is None:
            self.target_fidelity_index == len(space.parameters) - 1
        else:
            self.target_fidelity_index = target_fidelity_index
        self.fidelity_parameter = space.parameters[target_fidelity_index]
        self.high_fidelity = self.fidelity_parameter.max

        # Sampler of representer points should sample x location at the target information source only so make a
        # parameter space without the information source parameter
        parameters_without_info_source = space.parameters.copy()
        parameters_without_info_source.remove(self.fidelity_parameter)
        space_without_info_source = ParameterSpace(parameters_without_info_source)

        # Create sampler of representer points
        sampler = AffineInvariantEnsembleSampler(space_without_info_source)

        proposal_func = self._get_proposal_function(model, space)

        super().__init__(model, space, sampler, num_samples, num_representer_points, proposal_func, burn_in_steps)

    def _sample_representer_points(self):
        repr_points, repr_points_log = super()._sample_representer_points()

        # Add fidelity index to representer points
        idx = np.ones((repr_points.shape[0])) * self.high_fidelity
        repr_points = np.insert(repr_points, self.target_fidelity_index, idx, axis=1)
        return repr_points, repr_points_log

    def _get_proposal_function(self, model, space):

        # Define proposal function for multi-fidelity
        ei = ExpectedImprovement(model)

        def proposal_func(x):
            x_ = x[None, :]
            # Add information source parameter into array
            idx = np.ones((x_.shape[0], 1)) * self.high_fidelity

            x_ = np.insert(x_, self.target_fidelity_index, idx, axis=1)

            if space.check_points_in_domain(x_):
                val = np.log(np.clip(ei.evaluate(x_)[0], 0., np.PINF))
                if np.any(np.isnan(val)):
                    return np.array([np.NINF])
                else:
                    return val
            else:
                return np.array([np.NINF])

        return proposal_func