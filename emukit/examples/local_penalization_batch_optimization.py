from emukit.bayesian_optimization.acquisitions.log_acquisition import LogAcquisition
from emukit.bayesian_optimization.local_penalization_calculator import LocalPenalizationPointCalculator
from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.interfaces import IDifferentiable
from emukit.core.loop import OuterLoop, ModelUpdater, FixedIntervalUpdater
from emukit.core.loop.loop_state import create_loop_state
from emukit.core.optimization import AcquisitionOptimizer


class LocalPenalizationBatchOptimization(OuterLoop):
    """
    Bayesian optimization method that collects batches of points using the local penalization method from:
    `Batch Bayesian Optimization via Local Penalization. Javier González, Zhenwen Dai, Philipp Hennig, Neil D. Lawrence
    <https://arxiv.org/abs/1505.08052>`_

    Collecting points in batches is useful when the objective function can be evaluated in parallel.
    """
    def __init__(self, parameter_space: ParameterSpace, model: IDifferentiable, base_acquisition: Acquisition,
                 batch_size: int, model_updater: ModelUpdater=None):
        """
        :param parameter_space: ParameterSpace object defining the input domain
        :param model: Model of the objective function. Must implement IDifferentiable
        :param base_acquisition: Base acquisition function to use before applying local penalization. This acquisition
                                 function must output positive only values.
        :param batch_size: Number of points to collect in one batch
        :param model_updater: Model updater object. If None model hyper-parameters are updated every iteration.
        """
        loop_state = create_loop_state(model.X, model.Y)
        log_acquisition = LogAcquisition(base_acquisition)
        acquisition_optimizer = AcquisitionOptimizer(parameter_space)
        candidate_point_calculator = LocalPenalizationPointCalculator(log_acquisition, acquisition_optimizer, model,
                                                                      parameter_space, batch_size)

        if model_updater is None:
            model_updater = FixedIntervalUpdater(model, 1)
        super().__init__(candidate_point_calculator, model_updater, loop_state)
