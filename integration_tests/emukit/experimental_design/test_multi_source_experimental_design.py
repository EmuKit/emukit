import GPy
import numpy as np

from emukit.core.loop import FixedIntervalUpdater, OuterLoop
from emukit.core.loop.candidate_point_calculators import GreedyBatchPointCalculator
from emukit.core.loop.loop_state import LoopState
from emukit.core.optimization import GradientAcquisitionOptimizer
from emukit.core.optimization.multi_source_acquisition_optimizer import MultiSourceAcquisitionOptimizer
from emukit.core.initial_designs import RandomDesign
from emukit.experimental_design.acquisitions import ModelVariance
from emukit.model_wrappers import GPyModelWrapper
from emukit.test_functions import multi_fidelity_forrester_function


def test_multi_source_batch_experimental_design():
    objective, space = multi_fidelity_forrester_function()

    # Create initial data
    random_design = RandomDesign(space)
    x_init = random_design.get_samples(10)
    intiial_results = objective.evaluate(x_init)
    y_init = np.array([res.Y for res in intiial_results])

    # Create multi source acquisition optimizer
    acquisition_optimizer = GradientAcquisitionOptimizer(space)
    multi_source_acquisition_optimizer = MultiSourceAcquisitionOptimizer(acquisition_optimizer, space)

    # Create GP model
    gpy_model = GPy.models.GPRegression(x_init, y_init)
    model = GPyModelWrapper(gpy_model)

    # Create acquisition
    acquisition = ModelVariance(model)

    # Create batch candidate point calculator
    batch_candidate_point_calculator = GreedyBatchPointCalculator(model, acquisition,
                                                                  multi_source_acquisition_optimizer, batch_size=5)

    initial_loop_state = LoopState(intiial_results)
    loop = OuterLoop(batch_candidate_point_calculator, FixedIntervalUpdater(model, 1), initial_loop_state)

    loop.run_loop(objective, 10)
    assert loop.loop_state.X.shape[0] == 60
