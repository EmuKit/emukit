# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
from typing import List, Union


class Encoding(object):
    """
    Generic class that represents encodings of categorical variables
    """

    def __init__(self, categories: List[str], encodings: List):
        """
        Initializes an instance of Encoding class

        :param categories: List of categories to encode
        :param encodings: List of corresponding encodings
        """
        self.categories = categories
        self.encodings = np.array(encodings, dtype=float)

        # # As encoding is a static thing and we never add or remove categories
        # # it is fine to maintain multiple copies
        # # We need tuples cause lists are not hashable cannot be used as keys
        # self._encodings_to_categories = {tuple(e): c for e, c in zip(self.encodings, self.categories)}
        # self._categories_to_encodings = {c: e for e, c in zip(self.encodings, self.categories)}

    @property
    def dimension(self) -> int:
        """
        Dimension of the encoding
        """
        return self.encodings.shape[1]

    def round(self, x: np.ndarray) -> np.ndarray:
        """
        Rounds each row in 2d array x to represent one of encodings.

        :param x: A 2d array to be rounded
        :returns: An array where each row represents an encoding
                  that is closest to the corresponding row in x
        """
        if x.ndim != 2:
            raise ValueError("Expected 2d array, got " + str(x.ndim))

        if x.shape[1] != self.dimension:
            raise ValueError("Encoding dimension mismatch, expected {} got {}".format(self.dimension, x.shape[1]))

        x_rounded = []
        for row in x:
            row_rounded = self.round_row(row)
            x_rounded.append(row_rounded)

        return np.row_stack(x_rounded)

    def round_row(self, x_row):
        """
        Rounds the given row. See "round" method docsting for details.

        When subclassing Encoding, it is best to override this method instead of "round".

        :param x_row: A row to round.
        :returns: A rounded row.
        """

        idx = (np.linalg.norm(self.encodings - x_row, axis=1)).argmin()
        return self.encodings[idx].copy()

    def get_category(self, encoding: Union[List, np.ndarray]) -> str:
        """
        Gets the category corresponding to the encoding.

        :param encoding: An encoded value.
        :returns: A category.
        """

        indices = np.where(np.all(self.encodings == np.array(encoding), axis=1))
        if len(indices) == 0 or indices[0].size == 0:
            raise ValueError("Given encoding {} does not correspond to any category".format(encoding))

        category_idx = indices[0][0]
        return self.categories[category_idx]

    def get_encoding(self, category: str) -> List:
        """
        Gets the encoding corresponding to the category.

        :param category: A category.
        :returns: An encoding as a list.
        """
        try:
            index = self.categories.index(category)
        except ValueError:
            raise ValueError("Unknown category {}" + str(category))

        return self.encodings[index].tolist()


class OneHotEncoding(Encoding):
    def __init__(self, categories: List):
        """
        Initializes an instance of OneHotEncoding class
        and generates one hot encodings for given categories.
        Categories are assigned 1's in the order they appear in the provided list.

        An example reference about one hot encoding:
        https://www.kaggle.com/dansbecker/using-categorical-data-with-one-hot-encoding

        :param categories: List of categories to encode
        """
        encodings = []
        for i, _ in enumerate(categories):
            e = [0] * len(categories)
            e[i] = 1
            encodings.append(e)

        super(OneHotEncoding, self).__init__(categories, encodings)

    def round_row(self, x_row):
        """
        Rounds the given row. The highest value is rounded to 1
        all other values are rounded to 0

        :param x_row: A row to round.
        :returns: A rounded row.
        """
        idx = x_row.argmax()
        row_rounded = np.zeros(x_row.shape)
        row_rounded[idx] = 1
        return row_rounded


class OrdinalEncoding(Encoding):
    def __init__(self, categories: List):
        """
        Initializes an instance of OrdinalEncoding class
        and generates ordinal encodings for given categories.
        The encoding is a list of integer numbers [1, .. , n]
        where n is the number of categories.
        Categories are assigned codes in the order they appear in the provided list.

        Note that encoding categories with ordinal encoding is effectively the same as
        treating them as a discrete variable.

        :param categories: List of categories to encode
        """
        encodings = [[i + 1] for i, _ in enumerate(categories)]
        super(OrdinalEncoding, self).__init__(categories, encodings)

    def round_row(self, x_row):
        # since we used just one column for this encoding
        # x_row should contain a single number

        if x_row.shape[0] != 1:
            raise ValueError("Expected a single valued array, got array of {}" + str(x_row.shape))

        x_value = x_row[0]
        if x_value < 1:
            x_value = 1
        if x_value > len(self.categories):
            x_value = len(self.categories)

        rounded_value = int(round(x_value))

        return np.array([rounded_value])
