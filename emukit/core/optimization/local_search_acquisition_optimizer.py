import logging
from typing import List
from typing import Tuple

import numpy as np

from .. import ContinuousParameter
from .. import DiscreteParameter
from .. import CategoricalParameter
from .. import OneHotEncoding
from .. import OrdinalEncoding
from .. import ParameterSpace
from ...experimental_design.model_free.random_design import RandomDesign
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
    def __init__(self, space: ParameterSpace, num_steps: int, num_samples: int, **kwargs) -> None:
        """
        :param space: The parameter space spanning the search problem.
        :param num_steps: Maximum number of steps to follow from each start point.
        :param num_samples: Number of initial sampled points where the local search starts.
        """
        self.space = space
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.sampler = RandomDesign(space)

    def _neighbours_per_parameter(self, all_features: np.ndarray) -> List[np.ndarray]:
        """ Generates parameter encodings for one-exchange neighbours of
            parameters encoded in parameter feature vector

        :param all_features: The encoded parameter point (1d-array)
        :return: List of numpy arrays. Each array contains all one-exchange encodings of a parameter
        """
        parameters = self.space.parameters
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
                    neighbours.append(parameter.encodings[
                        (parameter.encodings != features).any(axis=1)])
                else:
                    raise TypeError("{} not a supported parameter encoding."
                                    .format(type(parameter.encoding)))
            elif isinstance(parameter, DiscreteParameter):
                raise NotImplementedError("Handling discrete parameter is not yet implemented")
            elif isinstance(parameter, ContinuousParameter):
                raise NotImplementedError("Handling continuous parameter is not yet implemented")
            else:
                raise TypeError("{} not a supported parameter type."
                                 .format(type(parameter)))
            current_feature += parameter.dimension
        return neighbours

    def _neighbours(self, all_features: np.ndarray) -> np.ndarray:
        """ Generates one-exchange neighbours of encoded parameter point.

        :param all_features: The encoded parameter point (1d-array)
        :return: All one-exchange neighbours as 2d-array (neighbours, features)
        """
        neighbours_per_param = self._neighbours_per_parameter(all_features)
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

    def _one_local_search(self, acquisition: Acquisition, x: np.ndarray):
        """ Local maximum search on acquisition starting at a single point.

        :param acquisition: The acquisition the maximum is searched of.
        :param x: The initial point.
        :return: Tuple of (maximum point as 1d-array, value of acquisition at this point)
        """
        incumbent_value = acquisition.evaluate(x.reshape(1, -1)).item()
        _log.debug("Start local search with acquisition={:.4f} at {}"
                   .format(incumbent_value, x))
        for step in range(self.num_steps):
            _log.debug("Local search step {}/{}".format(step + 1, self.num_steps))
            neighbours = self._neighbours(x)
            acquisition_values = acquisition.evaluate(neighbours)
            max_index = np.argmax(acquisition_values)
            max_neighbour = neighbours[max_index]
            max_value = acquisition_values[max_index]
            if max_value < incumbent_value:
                _log.debug("End after {} steps at minimum of acquisition={:.4f} at {}"
                           .format(step, incumbent_value, x))
                return x, incumbent_value
            else:
                incumbent_value = max_value
                x = max_neighbour
        _log.debug("End at step limit with acquisition={:.4f} at {}"
                   .format(incumbent_value, x))
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
            raise NotImplementedError("Handling context is currently not implemented.")

        X_init = self.sampler.get_samples(self.num_samples)
        X_max = np.empty_like(X_init)
        acq_max = np.empty((self.num_samples,))
        _log.info("Starting local optimization of acquisition function {}"
                  .format(type(acquisition)))
        for sample in range(self.num_samples):  # this loop could be parallelized
            X_max[sample], acq_max[sample] = self._one_local_search(acquisition, X_init[sample])
        max_sample = np.argmax(acq_max)
        rounded_max_sample = self.space.round(X_max[[max_sample]])
        rounded_max_value = acquisition.evaluate(rounded_max_sample)
        return rounded_max_sample, rounded_max_value
