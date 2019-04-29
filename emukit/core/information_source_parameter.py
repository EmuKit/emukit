import abc
from typing import Iterable

import numpy as np

from . import CategoricalParameter, DiscreteParameter
from . import OneHotEncoding, OrdinalEncoding
from .parameter import Parameter


class InformationSourceParameter(abc.ABC, Parameter):
    @abc.abstractmethod
    def source_indices(self, values: np.ndarray) -> np.ndarray:
        """ Calculate the indices corresponding to the parameter values.

        :param values: Parameter values as (num points, num features)
        :return: 1d-array of the source indices
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def encodings(self) -> np.ndarray:
        """ Source encodings as 2d-array (points, features). """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def n_sources(self) -> int:
        """ Number of sources. """
        raise NotImplementedError



class NominalInformationSourceParameter(CategoricalParameter, InformationSourceParameter):
    def __init__(self, n_sources: int) -> None:
        """ Information source parameter on nominal scale.

        Useful if only equality relation is valid (e.g. no ordering).
        E.g. information sources correspond to datasets.

        Parameter values are categorical and one-hot encoded.

        :param n_sources: Number of information sources in the problem
        """
        CategoricalParameter.__init__(self, 'source', OneHotEncoding(list(range(n_sources))))

    def source_indices(self, values: np.ndarray) -> np.ndarray:
        """ See InformationSourceParameter.source_indices """
        return np.asarray([self.encoding.get_category(row) for row in self.round(values)])

    @property
    def n_sources(self) -> int:
        """ Number of sources. """
        return self.encodings.shape[0]


class OrdinalInformationSourceParameter(CategoricalParameter, InformationSourceParameter):
    def __init__(self, n_sources: int) -> None:
        """ Information source parameter on ordinal scale.

        Useful if only equality and order relation is valid (e.g. no ratio).
        E.g. information sou'source', OneHotEncoding(list(range(n_sources)))rces correspond to multi-fidelity.

        Parameter values are categorical and ordinal encoded.

        :param n_sources: Number of information sources in the problem
        """
        CategoricalParameter.__init__(self, 'source', OrdinalEncoding(list(range(n_sources))))

    def source_indices(self, values: np.ndarray) -> np.ndarray:
        """ See InformationSourceParameter.source_indices """
        return np.asarray([self.encoding.get_category(row) for row in self.round(values)])

    @property
    def n_sources(self) -> int:
        """ Number of sources. """
        return self.encodings.shape[0]


class DiscreteInformationSourceParameter(DiscreteParameter, InformationSourceParameter):
    def __init__(self, sources: Iterable[int]) -> None:
        """ Discrete information source parameter.

        E.g. information sources correspond to multi-fidelity
        where the fidelity steps are not uniform.

        :param sources: Encoding values of information sources in the problem
        """
        DiscreteParameter.__init__(self, 'source', list(sources))

    def source_indices(self, values: np.ndarray) -> np.ndarray:
        """ See InformationSourceParameter.source_indices """
        return np.where(self.round(values) == np.asarray(self.domain))[1]

    @property
    def encodings(self) -> np.ndarray:
        """ Source encodings as 2d-array (points, features). """
        return np.asarray(self.domain).reshape(-1, 1)

    @property
    def n_sources(self) -> int:
        """ Number of sources. """
        return len(self.domain)
