"""The uniform measure."""


from ..typing import BoundsType
from .lebesgue_measure import LebesgueMeasure


class UniformMeasure(LebesgueMeasure):
    r"""The Uniform measure.

    The uniform measure has density

    .. math::
        p(x)=\begin{cases} p & x\in\text{bounds}\\0 &\text{otherwise}\end{cases}.

    .. note::
        This class is syntactic sugar for a normalized Lebesgue measure ``LebesgueMeasure(bounds, normalized=True)``.

    :param bounds: List of D tuples [(lb_1, ub_1), (lb_2, ub_2), ..., (lb_D, ub_D)], where D is
                   the input dimensionality and the tuple (lb_d, ub_d) contains the lower and upper bound
                   of the uniform measure in dimension d.

    """

    def __init__(self, bounds: BoundsType):
        super().__init__(bounds=bounds,  normalized=True)
