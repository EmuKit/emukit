"""The WSABI-L loop"""


from ...core.loop.loop_state import create_loop_state
from ...core.loop import OuterLoop, SequentialPointCalculator, FixedIntervalUpdater, ModelUpdater
from ...core.optimization import AcquisitionOptimizerBase, GradientAcquisitionOptimizer
from ...core.parameter_space import ParameterSpace
from ..methods import WSABIL
from ..acquisitions import UncertaintySampling


class WSABILLoop(OuterLoop):
    def __init__(self, model: WSABIL, model_updater: ModelUpdater = None,
                 acquisition_optimizer: AcquisitionOptimizerBase = None):
        """The loop for WSABI-L.

        :param model: The WSABI-L model.
        :param model_updater: Defines how and when the quadrature model is updated if new data arrives.
                              Defaults to updating hyper-parameters every iteration.
        :param acquisition_optimizer: Optimizer selecting next evaluation points by maximizing acquisition.
                                      Gradient based optimizer is used if None. Defaults to None.
        """

        # WSABI-L is used with uncertainty sampling.
        acquisition = UncertaintySampling(model, measure_power=1)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        space = ParameterSpace(model.reasonable_box_bounds.convert_to_list_of_continuous_parameters())
        if acquisition_optimizer is None:
            acquisition_optimizer = GradientAcquisitionOptimizer(space)
        candidate_point_calculator = SequentialPointCalculator(acquisition, acquisition_optimizer)
        loop_state = create_loop_state(model.X, model.Y)

        super().__init__(candidate_point_calculator, model_updater, loop_state)

        self.model = model
