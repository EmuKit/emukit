# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## Development Commands

```bash
# Install for development
pip install -e .[tests]          # core + test tooling
pip install -e .[dev]            # everything (tests, docs, examples, optional ML backends)

# Run tests
pytest tests/                                           # unit tests
pytest integration_tests/                               # integration tests
pytest --cov emukit --cov-report term-missing tests/    # with coverage
pytest -m 'not (gpy or pybnn or sklearn or notebooks)'  # skip optional-dependency tests
pytest -m gpy                                           # only GPy tests

# Lint and format (enforced in CI)
black .
isort .
flake8 .
```

**Line length:** 120 characters. **Exceptions:** E731, E127 in flake8.

## Architecture

Emukit is a modular, framework-agnostic library for emulation-based decision-making (Bayesian optimization, experimental design, Bayesian quadrature, sensitivity analysis). The central design is the **OuterLoop**:

```
while stopping_condition not met:
    candidate_point_calculator → next points to evaluate
    user_function(points)       → evaluations
    model_updater               → update model with new data
```

All loop components are swappable, enabling model-agnostic algorithms.

### Key Packages

- **`emukit/core/`** — All shared abstractions:
  - `interfaces/` — Model interfaces (`IModel`, `IDifferentiable`, `IJointlyDifferentiable`, `IPriorHyperparameters`, `IModelWithNoise`)
  - `loop/` — `OuterLoop`, `LoopState`, `CandidatePointCalculator`, `ModelUpdater`, `StoppingCondition`, `UserFunction`, `EventHandler`
  - `acquisition/` — `Acquisition` base class; supports `+`, `*`, `/` operator overloading for composing acquisitions
  - `optimization/` — `AcquisitionOptimizer` (maximizes acquisition over parameter space)
  - `parameter_space.py` — `ParameterSpace` composed of `ContinuousParameter`, `DiscreteParameter`, `CategoricalParameter`, `BanditParameter`
  - `initial_designs/` — Sampling strategies for initialization
  - `encodings.py` — `OneHotEncoding`, `OrdinalEncoding`

- **`emukit/bayesian_optimization/`** — `BayesianOptimizationLoop` (wraps OuterLoop with sensible defaults), acquisitions (EI, EI-MCMC, entropy search, max-value entropy search, local penalization, NegativeLowerConfidenceBound, PoF, PoI)

- **`emukit/experimental_design/`** — `ExperimentalDesignLoop`, design-specific acquisitions

- **`emukit/quadrature/`** — Bayesian quadrature: specialized kernels, loop, and `WarpedBayesianQuadratureModel`

- **`emukit/multi_fidelity/`** — Multi-fidelity GP models built on GPy

- **`emukit/sensitivity/`** — Monte Carlo sensitivity analysis (Sobol indices)

- **`emukit/model_wrappers/`** — Bridges external ML libraries to emukit interfaces: `GPyModelWrapper`, `GPyMultiOutputWrapper`, `SklearnModelWrapper`, `SimpleGaussianProcessModel`

- **`emukit/samplers/`** — MCMC and other samplers

- **`emukit/test_functions/`** — Benchmark functions (Branin, Forrester, etc.)

### Interface Conventions

Interface names are prefixed with `I` (e.g., `IModel`, `IDifferentiable`). Models only need to implement the interfaces required by the algorithms they are used with — there is no single monolithic model class. Type hints are required on all public functions.

### Optional Dependencies

Optional backends (GPy, pybnn/torch, sklearn) are guarded by `pytest.importorskip()` in tests and declared as optional extras in `pyproject.toml`. Tests for these backends are marked with `@pytest.mark.gpy`, `@pytest.mark.pybnn`, `@pytest.mark.sklearn`, or `@pytest.mark.notebooks`.
