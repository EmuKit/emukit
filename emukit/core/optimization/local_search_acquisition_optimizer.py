# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Sequence, List, Tuple, Optional

import numpy as np

from .acquisition_optimizer import AcquisitionOptimizerBase
from .context_manager import ContextManager
from .. import CategoricalParameter, ContinuousParameter, DiscreteParameter
from .. import OneHotEncoding, OrdinalEncoding
from .. import Parameter, ParameterSpace
from ..acquisition import Acquisition

_log = logging.getLogger(__name__)


class LocalSearchAcquisitionOptimizer(AcquisitionOptimizerBase):
    """ Optimizes the acquisition function by multiple local searches starting at random points.
    Each local optimization iteratively evaluates the one-exchange neighbourhoods.
    Can be used for discrete and continuous acquisition functions.

    This kind of optimization is also known as Variable Neighbourhood Search
    (e.g. see https://en.wikipedia.org/wiki/Variable_neighborhood_search).
    Neighbourhood definitions and default parameters are based on the search used
    in SMAC [1].

    .. warning:: The local search heuristic here currently differs to SMAC [1].
                 The neighbourhood of a point is evaluated completely,
                 the search continues at the best neighbour (best improvement heuristic).
                 SMAC iteratively samples neighbours and continues at the first which
                 is better than the current (first improvement heuristic).
                 Therefore this implementation is time consuming for large neighbourhoods
                 (e.g. parameters with hundreds of categories).

    One-exchange neighbourhood is defined for the following parameter types:
      :Categorical parameter with one-hot encoding: All other categories
      :Categorical parameter with ordinal encoding: Only preceeding and following categories
      :Continuous parameter: Gaussian samples (default: 4) around current value. Standard deviation (default: 0.2) is
                             scaled by parameter value range.
      :Discrete parameter: Preceeding and following discrete values.

    .. [1] Hutter, Frank, Holger H. Hoos, and Kevin Leyton-Brown.
           "Sequential model-based optimization for general algorithm configuration."
           International Conference on Learning and Intelligent Optimization.
           Springer, Berlin, Heidelberg, 2011.
    """
    def __init__(self, space: ParameterSpace, num_steps: int = 10, num_init_points: int = 5,
                 std_dev: float = 0.02, num_continuous: int = 4) -> None:
        """
        :param space: The parameter space spanning the search problem.
        :param num_steps: Maximum number of steps to follow from each start point.
        :param num_init_points: Number of initial sampled points where the local search starts.
        :param std_dev: Neighbourhood sampling standard deviation of continuous parameters.
        :param num_continuous: Number of sampled neighbourhoods per continuous parameter.
        """
        super().__init__(space)
        self.num_steps = num_steps
        self.num_init_points = num_init_points
        self.std_dev = std_dev
        self.num_continuous = num_continuous

    def _neighbours_per_parameter(self, all_features: np.ndarray, parameters: Sequence[Parameter]) -> List[np.ndarray]:
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
                if current_index < len(parameter.domain) - 1:
                    this_neighbours.append([parameter.domain[current_index + 1]])
                neighbours.append(np.asarray(this_neighbours).reshape(-1, 1))
            elif isinstance(parameter, ContinuousParameter):
                samples, param_range = [], parameter.max - parameter.min
                while len(samples) < self.num_continuous:
                    sample = np.random.normal(np.asscalar(features), self.std_dev * param_range, (1, 1))
                    if parameter.min <= sample <= parameter.max:
                        samples.append(sample)
                neighbours.append(np.vstack(samples))
            else:
                raise TypeError("{} not a supported parameter type."
                                 .format(type(parameter)))
            current_feature += parameter.dimension
        return neighbours

    def _neighbours(self, all_features: np.ndarray, parameters: Sequence[Parameter]) -> np.ndarray:
        """ Generates one-exchange neighbours of encoded parameter point.

        :param all_features: The encoded parameter point (1d-array)
        :return: All one-exchange neighbours as 2d-array (neighbours, features)
        """
        neighbours_per_param = self._neighbours_per_parameter(all_features, parameters)
        num_neighbours = sum(param.shape[0] for param in neighbours_per_param)
        neighbours = np.full((num_neighbours, all_features.shape[0]), all_features)
        current_neighbour, current_feature = 0, 0
        for this_neighbours in neighbours_per_param:
            next_neighbour = current_neighbour + this_neighbours.shape[0]
            next_feature = current_feature + this_neighbours.shape[1]
            neighbours[current_neighbour:next_neighbour, current_feature:next_feature] = this_neighbours
            current_neighbour, current_feature = next_neighbour, next_feature
        return neighbours

    def _one_local_search(self, acquisition: Acquisition, x: np.ndarray,
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
            neighbours = self._neighbours(x, context_manager.contextfree_space.parameters)
            neighbours_with_context = context_manager.expand_vector(neighbours)
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
        _log.debug("End at step limit with acquisition={:.4f} at {}".format(incumbent_value, str(x)))
        return x, incumbent_value

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implementation of abstract method.

        See AcquisitionOptimizerBase._optimizer for parameter descriptions.
        See class docstring for implementation details.
        """
        X_init = context_manager.contextfree_space.sample_uniform(self.num_init_points)
        X_max = np.empty_like(X_init)
        acq_max = np.empty((self.num_init_points, 1))
        _log.info("Starting local optimization of acquisition function {}".format(type(acquisition)))
        for sample in range(self.num_init_points):  # this loop could be parallelized
            X_max[sample], acq_max[sample] = self._one_local_search(acquisition, X_init[sample], context_manager)
        max_index = np.argmax(acq_max)
        X_max_with_context = context_manager.expand_vector(X_max)
        return X_max_with_context[[max_index]], acq_max[[max_index]]
