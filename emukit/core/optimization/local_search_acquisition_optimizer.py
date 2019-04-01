import logging
from typing import Sequence
from typing import List
from typing import Tuple
from typing import Optional

import numpy as np
from GPyOpt.optimization.acquisition_optimizer import ContextManager

from .. import ContinuousParameter
from .. import DiscreteParameter
from .. import CategoricalParameter
from .. import OneHotEncoding
from .. import OrdinalEncoding
from .. import Parameter
from .. import ParameterSpace
from ..acquisition import Acquisition
from ..optimization.acquisition_optimizer import AcquisitionOptimizerBase

_log = logging.getLogger(__name__)


class LocalSearchAcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function by greedily following one-exchange neighbours
        of random initial points.

    One-exchange neighbourhood is defined per parameter type:
      :Categorical parameter with one-hot encoding: All other categories
      :Categorical parameter with ordinal encoding: Only preceeding and following categories
    """
    def __init__(self, space: ParameterSpace, num_steps: int, num_samples: int,
                 std_dev: float = 0.02, num_continuous: int = 1) -> None:
        """
        :param space: The parameter space spanning the search problem.
        :param num_steps: Maximum number of steps to follow from each start point.
        :param num_samples: Number of initial sampled points where the local search starts.
        :param std_dev: Neighbourhood sampling standard deviation of continuous parameters.
        :param num_continuous: Number of sampled neighbourhoods per continuous parameter.
        """
        self.space = space
        self.gpyopt_space = space.convert_to_gpyopt_design_space()
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.std_dev = std_dev
        self.num_continuous = num_continuous

    def _neighbours_per_parameter(self, all_features: np.ndarray, parameters: Sequence[Parameter])\
        -> List[np.ndarray]:
        """ Generates parameter encodings for one-exchange neighbours of
            parameters encoded in parameter feature vector

        :param all_features: The encoded parameter point (1d-array)
        :return: List of numpy arrays. Each array contains all one-exchange encodings of a parameter
        """
        neighbours = []
        current_feature = 0
        for parameter in parameters:
            features = parameter.round(
                all_features[current_feature:(current_feature + parameter.dimension)]
                .reshape(1, -1)).ravel()
            if isinstance(parameter, CategoricalParameter):
                if isinstance(parameter.encoding, OrdinalEncoding):
                    left_right = np.unique([parameter.encoding.round_row(features - 1),
                                            parameter.encoding.round_row(features + 1)])
                    neighbours.append(left_right[left_right != features].reshape(-1, 1))
                elif isinstance(parameter.encoding, OneHotEncoding):
                    # All categories apart from current one are valid neighbours with one hot encoding
                    neighbours.append(parameter.encodings[
                        (parameter.encodings != features).any(axis=1)])
                else:
                    raise TypeError("{} not a supported parameter encoding."
                                    .format(type(parameter.encoding)))
            elif isinstance(parameter, DiscreteParameter):
                # Find current position in domain while being robust to numerical precision problems
                current_index = np.argmin(np.abs(
                    np.subtract(parameter.domain, np.asscalar(features))))
                this_neighbours = []
                if current_index > 0:
                    this_neighbours.append([parameter.domain[current_index - 1]])
                elif current_index < len(parameter.domain) - 1:
                    this_neighbours.append([parameter.domain[current_index + 1]])
                neighbours.append(np.asarray(this_neighbours))
            elif isinstance(parameter, ContinuousParameter):
                neighbours.append(
                    np.random.normal(np.asscalar(features), self.std_dev, (self.num_continuous, 1)))
            else:
                raise TypeError("{} not a supported parameter type."
                                 .format(type(parameter)))
            current_feature += parameter.dimension
        return neighbours

    def _neighbours(self, all_features: np.ndarray, parameters: Sequence[Parameter])\
        -> np.ndarray:
        """ Generates one-exchange neighbours of encoded parameter point.

        :param all_features: The encoded parameter point (1d-array)
        :return: All one-exchange neighbours as 2d-array (neighbours, features)
        """
        neighbours_per_param = self._neighbours_per_parameter(
            all_features, parameters)
        num_neighbours = sum(param.shape[0] for param in neighbours_per_param)
        neighbours = np.full((num_neighbours, all_features.shape[0]), all_features)
        current_neighbour, current_feature = 0, 0
        for this_neighbours in neighbours_per_param:
            next_neighbour = current_neighbour + this_neighbours.shape[0]
            next_feature = current_feature + this_neighbours.shape[1]
            neighbours[current_neighbour:next_neighbour,
                       current_feature:next_feature] = this_neighbours
            current_neighbour, current_feature = next_neighbour, next_feature
        return neighbours

    def _one_local_search(self, acquisition: Acquisition, x: np.ndarray,
                          noncontext_space: ParameterSpace,
                          context_manager: Optional[ContextManager] = None):
        """ Local maximum search on acquisition starting at a single point.

        :param acquisition: The acquisition the maximum is searched of.
        :param x: The initial point.
        :return: Tuple of (maximum point as 1d-array, value of acquisition at this point)
        """
        incumbent_value = acquisition.evaluate(x.reshape(1, -1)).item()
        _log.debug("Start local search with acquisition={:.4f} at {}"
                   .format(incumbent_value, str(x)))
        for step in range(self.num_steps):
            neighbours = self._neighbours(x, noncontext_space.parameters)
            if context_manager is not None:
                neighbours_with_context = context_manager._expand_vector(neighbours)
            else:
                neighbours_with_context = neighbours
            acquisition_values = acquisition.evaluate(neighbours_with_context)
            max_index = np.argmax(acquisition_values)
            max_neighbour = neighbours[max_index]
            max_value = np.asscalar(acquisition_values[max_index])
            if max_value < incumbent_value:
                _log.debug("End after {} steps at maximum of acquisition={:.4f} at {}"
                           .format(step, incumbent_value, str(x)))
                return x, incumbent_value
            else:
                incumbent_value = max_value
                x = max_neighbour
        _log.debug("End at step limit with acquisition={:.4f} at {}"
                   .format(incumbent_value, str(x)))
        return x, incumbent_value

    def optimize(self, acquisition: Acquisition, context: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Optimizes the acquisition function, taking into account gradients if it supports them
        :param acquisition: The acquisition function to be optimized
        :param context: Optimization context.
                        Determines whether any variable values should be fixed during the optimization
        :return: Tuple of (location of maximum, acquisition value at maximizer)
        """
        if context is not None:
            context_manager = ContextManager(self.gpyopt_space, context)
            noncontext_space = ParameterSpace(
                [param for param in self.space.parameters if param.name in context])
        else:
            context_manager = None
            noncontext_space = self.space

        X_init = noncontext_space.sample_uniform(self.num_samples)
        X_max = np.empty_like(X_init)
        acq_max = np.empty((self.num_samples,))
        _log.info("Starting local optimization of acquisition function {}"
                  .format(type(acquisition)))
        for sample in range(self.num_samples):  # this loop could be parallelized
            X_max[sample], acq_max[sample] = self._one_local_search(
                acquisition, X_init[sample], noncontext_space, context_manager)
        max_sample = np.argmax(acq_max)
        if context_manager is not None:
            X_max_with_context = context_manager._expand_vector(X_max)
        else:
            X_max_with_context = X_max
        rounded_max_sample = self.space.round(X_max_with_context[[max_sample]])
        rounded_max_value = acquisition.evaluate(rounded_max_sample)
        return rounded_max_sample, rounded_max_value
