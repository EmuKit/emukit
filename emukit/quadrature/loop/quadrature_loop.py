from emukit.core.loop.outer_loop import OuterLoop

# TODO: replace this with BQ acquisition
from emukit.bayesian_optimization.acquisitions import NegativeLowerConfidenceBound
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.loop import OuterLoop, Sequential, FixedIntervalUpdater, ModelUpdater
from emukit.core.optimization import AcquisitionOptimizer

from emukit.core.parameter_space import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IModel


class VanillaBayesianQuadratureLoop(OuterLoop):
    def __init__(self, model: IModel, space: ParameterSpace, acquisition: Acquisition = None,
                 model_updater: ModelUpdater = None):

        """
        Emukit class that implement a loop for building modular Bayesian optimization

        :param model: The model that approximates the underlying function
        :param space: Input space where the optimization is carried out.
        :param acquisition: The acquisition function that will be used to collect new points (default, EI). If batch
                            size is greater than one, this acquisition must output positive values only.
        :param batch_size: How many points to evaluate in one iteration of the optimization loop. Defaults to 1.
        :param model_updater: Defines how and when the model is updated if new data arrives.
                              Defaults to updating hyper-parameters every iteration.
        """

        # TODO: this need to be e.g., variance reduction
        if acquisition is None:
            acquisition = NegativeLowerConfidenceBound(model)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)

        acquisition_optimizer = AcquisitionOptimizer(space)
        #log_acquisition = LogAcquisition(acquisition)
        candidate_point_calculator = Sequential(acquisition, acquisition_optimizer)

        loop_state = create_loop_state(model.X, model.Y)

        super(VanillaBayesianQuadratureLoop).__init__(candidate_point_calculator, model_updater, loop_state)
