# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.2] - 2024-10-11
### Added
- `n_restarts` and `n_raw_samples` keywords to configure continuous optimization
  behavior for `BotorchRecommender`
- User guide for utilities

### Changed
- Utility `add_fake_results` renamed to `add_fake_measurements`
- Utilities `add_fake_measurements` and `add_parameter_noise` now also return the
  dataframe they modified in-place

### Fixed
- Leftover attrs-decorated classes are garbage collected before the subclass tree is
  traversed, avoiding sporadic serialization problems

## [0.11.1] - 2024-10-01
### Added
- Continuous linear constraints have been consolidated in the new
  `ContinuousLinearConstraint` class

### Changed
- `get_surrogate` now also returns the model for transformed single targets or
  desirability objectives

### Fixed
- Unsafe name-based matching of columns in `get_comp_rep_parameter_indices`

### Deprecations
- `ContinuousLinearEqualityConstraint` and `ContinuousLinearInequalityConstraint`
  replaced by `ContinuousLinearConstraint` with the corresponding `operator` keyword

## [0.11.0] - 2024-09-09
### Breaking Changes
- The public methods of `Surrogate` models now operate on dataframes in experimental
  representation instead of tensors in computational representation
- `Surrogate.posterior` models now returns a `Posterior` object
- `param_bounds_comp` of `SearchSpace`, `SubspaceDiscrete` and `SubspaceContinuous` has
  been replaced with `comp_rep_bounds`, which returns a dataframe

### Added
- `py.typed` file to enable the use of type checkers on the user side
- `IndependentGaussianSurrogate` base class for surrogate models providing independent 
  Gaussian posteriors for all candidates (cannot be used for batch prediction)
- `comp_rep_columns` property for `Parameter`, `SearchSpace`, `SubspaceDiscrete`
  and `SubspaceContinuous` classes
- New mechanisms for surrogate input/output scaling configurable per class
- `SurrogateProtocol` as an interface for user-defined surrogate architectures
- Support for binary targets via `BinaryTarget` class
- Support for bandit optimization via `BetaBernoulliMultiArmedBanditSurrogate` class
- Bandit optimization example
- `qThompsonSampling` acquisition function
- `BetaPrior` class
- `recommend` now accepts the `pending_experiments` argument, informing the algorithm
  about points that were already selected for evaluation
- Pure recommenders now have the `allow_recommending_pending_experiments` flag,
  controlling whether pending experiments are excluded from candidates in purely
  discrete search spaces
- `get_surrogate` and `posterior` methods to `Campaign`
- `tenacity` test dependency
- Multi-version documentation

### Changed
- The transition from experimental to computational representation no longer happens
  in the recommender but in the surrogate
- Fallback models created by `catch_constant_targets` are stored outside the surrogate
- `to_tensor` now also handles `numpy` arrays
- `MIN` mode of `NumericalTarget` is now implemented via the acquisition function
  instead of negating the computational representation
- Search spaces now store their parameters in alphabetical order by name
- Improvement-based acquisition functions now consider the maximum posterior mean
  instead of the maximum noisy measurement as reference value
- Iteration tests now attempt up to 5 repeated executions if they fail due to numerical
  reasons

### Fixed
- `CategoricalParameter` and `TaskParameter` no longer incorrectly coerce a single
  string input to categories/tasks
- `farthest_point_sampling` no longer depends on the provided point order
- Batch predictions for `RandomForestSurrogate`
- Surrogates providing only marginal posterior information can no longer be used for
  batch recommendation
- `SearchSpace.from_dataframe` now creates a proper empty discrete subspace without
  index when called with continuous parameters only
- Metadata updates are now only triggered when a discrete subspace is present
- Unintended reordering of discrete search space parts for recommendations obtained
  with `BotorchRecommender`

### Removed
- `register_custom_architecture` decorator
- `Scalar` and `DefaultScaler` classes

### Deprecations
- The role of `register_custom_architecture` has been taken over by
  `baybe.surrogates.base.SurrogateProtocol`
- `BayesianRecommender.surrogate_model` has been replaced with `get_surrogate`

## [0.10.0] - 2024-08-02
### Breaking Changes
- Providing an explicit `batch_size` is now mandatory when asking for recommendations
- `RecommenderProtocol.recommend` now accepts an optional `Objective` 
- `RecommenderProtocol.recommend` now expects training data to be provided as a single
  dataframe in experimental representation instead of two separate dataframes in
  computational representation
- `Parameter.is_numeric` has been replaced with `Parameter.is_numerical`
- `DiscreteParameter.transform_rep_exp2comp` has been replaced with
  `DiscreteParameter.transform` 
- `filter_attributes` has been replaced with `match_attributes`

### Added
- `Surrogate` base class now exposes a `to_botorch` method
- `SubspaceDiscrete.to_searchspace` and `SubspaceContinuous.to_searchspace`
  convenience constructor
- Validators for `Campaign` attributes
- `_optional` subpackage for managing optional dependencies
- New acquisition functions for active learning: `qNIPV` (negative integrated posterior
  variance) and `PSTD` (posterior standard deviation)
- Acquisition function: `qKG` (knowledge gradient)
- Abstract `ContinuousNonlinearConstraint` class
- Abstract `CardinalityConstraint` class and
  `DiscreteCardinalityConstraint`/`ContinuousCardinalityConstraint` subclasses
- Uniform sampling mechanism for continuous spaces with cardinality constraints
- `register_hooks` utility enabling user-defined augmentation of arbitrary callables
- `transform` methods of `SearchSpace`, `SubspaceDiscrete` and `SubspaceContinuous`
  now take additional `allow_missing` and `allow_extra` keyword arguments
- More details to the transfer learning user guide
- Activated doctests
- `SubspaceDiscrete.from_parameter`, `SubspaceContinuous.from_parameter`,
  `SubspaceContinuous.from_product` and `SearchSpace.from_parameter`
   convenience constructors
- `DiscreteParameter.to_subspace`, `ContinuousParameter.to_subspace` and
  `Parameter.to_searchspace` convenience constructors
- Utilities for permutation and dependency data augmentation
- Validation and translation tests for kernels
- `BasicKernel` and `CompositeKernel` base classes
- Activated `pre-commit.ci` with auto-update
- User guide for active learning
- Polars expressions for `DiscreteSumConstraint`, `DiscreteProductConstraint`, 
  `DiscreteExcludeConstraint`, `DiscreteLinkedParametersConstraint` and 
  `DiscreteNoLabelDuplicatesConstraint`
- Discrete search space Cartesian product can be created lazily via Polars
- Examples demonstrating the `register_hooks` utility: basic registration mechanism,
  monitoring the probability of improvement, and automatic campaign stopping
- Documentation building now uses a lockfile to fix the exact environment

### Changed
- Passing an `Objective` to `Campaign` is now optional
- `GaussianProcessSurrogate` models are no longer wrapped when cast to BoTorch
- Restrict upper versions of main dependencies, motivated by major `numpy` release
- Sampling methods in `qNIPV` and `BotorchRecommender` are now specified via 
  `DiscreteSamplingMethod` enum
- `Interval` class now supports degenerate intervals containing only one element
- `add_fake_results` now directly processes `Target` objects instead of a `Campaign`
- `path` argument in plotting utility is now optional and defaults to `Path(".")`
- `UnusedObjectWarning` by non-predictive recommenders is now ignored during simulations
- The default kernel factory now avoids strong jumps by linearly interpolating between
  two fixed low and high dimensional prior regimes
- The previous default kernel factory has been renamed to `EDBOKernelFactory` and now
  fully reflects the original logic
- The default acquisition function has been changed from `qEI` to `qLogEI` for improved
  numerical stability

### Removed
- Support for Python 3.9 removed due to new [BoTorch requirements](https://github.com/pytorch/botorch/pull/2293) 
  and guidelines from [Scientific Python](https://scientific-python.org/specs/spec-0000/)
- Linter `typos` for spellchecking

### Fixed
- `sequential` flag of `SequentialGreedyRecommender` is now set to `True`
- Serialization bug related to class layout of `SKLearnClusteringRecommender`
- `MetaRecommender`s no longer trigger warnings about non-empty objectives or
  measurements when calling a `NonPredictiveRecommender`
- Bug introduced in 0.9.0 (PR #221, commit 3078f3), where arguments to `to_gpytorch` 
  are not passed on to the GPyTorch kernels
- Positive-valued kernel attributes are now correctly handled by validators
  and hypothesis strategies
- As a temporary workaround to compensate for missing `IndexKernel` priors, 
 `fit_gpytorch_mll_torch` is used instead of `fit_gpytorch_mll` when a `TaskParameter`
  is present, which acts as regularization via early stopping during model fitting

### Deprecations
- `SequentialGreedyRecommender` class replaced with `BotorchRecommender`
- `SubspaceContinuous.samples_random` has been replaced with
  `SubspaceContinuous.sample_uniform`
- `SubspaceContinuous.samples_full_factorial` has been replaced with
  `SubspaceContinuous.sample_from_full_factorial`
- Passing a dataframe via the `data` argument to the `transform` methods of
  `SearchSpace`, `SubspaceDiscrete` and `SubspaceContinuous` is no longer possible.
  The dataframe must now be passed as positional argument.
- The new `allow_extra` flag is automatically set to `True` in `transform` methods
  of search space classes when left unspecified

### Expired Deprecations (from 0.7.*)
- `Interval.is_finite` property
- Specifying target configs without type information 
- Specifying parameters/constraints at the top level of a campaign configs
- Passing `numerical_measurements_must_be_within_tolerance` to `Campaign`
- `batch_quantity` argument 
- Passing `allow_repeated_recommendations` or `allow_recommending_already_measured` 
  to `MetaRecommender` (or former `Strategy`)
- `*Strategy` classes and `baybe.strategies` subpackage
- Specifying `MetaRecommender` (or former `Strategy`) configs without type information 

## [0.9.1] - 2024-06-04
### Changed
- Discrete searchspace memory estimate is now natively represented in bytes 

### Fixed
- Non-GP surrogates not working with `deepcopy` and the simulation package due to
  slotted base class
- Datatype inconsistencies for various parameters' `values` and `comp_df` and 
  `SubSelectionCondition`'s `selection` related to floating point precision

## [0.9.0] - 2024-05-21
### Added
- Class hierarchy for objectives
- `AdditiveKernel`, `LinearKernel`, `MaternKernel`, `PeriodicKernel`, 
  `PiecewisePolynomialKernel`, `PolynomialKernel`, `ProductKernel`, `RBFKernel`, 
  `RFFKernel`, `RQKernel`, `ScaleKernel` classes
- `KernelFactory` protocol enabling context-dependent construction of kernels
- Preset mechanism for `GaussianProcessSurrogate`
- `hypothesis` strategies and roundtrip test for kernels, constraints, objectives,
  priors and acquisition functions
- New acquisition functions: `qSR`, `qNEI`, `LogEI`, `qLogEI`, `qLogNEI`
- `GammaPrior`, `HalfCauchyPrior`, `NormalPrior`, `HalfNormalPrior`, `LogNormalPrior`
  and `SmoothedBoxPrior` classes
- Possibility to deserialize classes from optional class name abbreviations
- Basic deserialization tests using different class type specifiers
- Serialization user guide
- Environment variables user guide
- Utility for estimating memory requirements of discrete product search space
- `mypy` for search space and objectives

### Changed
- Reorganized acquisition.py into `acquisition` subpackage
- Reorganized simulation.py into `simulation` subpackage
- Reorganized gaussian_process.py into `gaussian_process` subpackage
- Acquisition functions are now their own objects
- `acquisition_function_cls` constructor parameter renamed to `acquisition_function`
- User guide now explains the new objective classes
- Telemetry deactivation warning is only shown to developers
- `torch`, `gpytorch` and `botorch` are lazy-loaded for improved startup time
- If an exception is encountered during simulation, incomplete results are returned 
  with a warning instead of passing through the uncaught exception
- Environment variables `BAYBE_NUMPY_USE_SINGLE_PRECISION` and
  `BAYBE_TORCH_USE_SINGLE_PRECISION` to enforce single point precision usage

### Removed
- `model_params` attribute from `Surrogate` base class, `GaussianProcessSurrogate` and
  `CustomONNXSurrogate`
- Dependency on `requests` package
  
### Fixed
- `n_task_params` now evaluates to 1 if `task_idx == 0`
- Simulation no longer fails in `ignore` mode when lookup dataframe contains duplicate
  parameter configurations
- Simulation no longer fails for targets in `MATCH` mode
- `closest_element` now works for array-like input of all kinds
- Structuring concrete subclasses no longer requires providing an explicit `type` field
- `_target(s)` attributes of `Objectives` are now de-/serialized without leading
  underscore to support user-friendly serialization strings
- Telemetry does not execute any code if it was disabled
- Running simulations no longer alters the states of the global random number generators

### Deprecations
- The former `baybe.objective.Objective` class has been replaced with
  `SingleTargetObjective` and `DesirabilityObjective`
- `acquisition_function_cls` constructor parameter for `BayesianRecommender`
- `VarUCB` and `qVarUCB` acquisition functions

### Expired Deprecations (from 0.6.*)
- `BayBE` class
- `baybe.surrogate` module
- `baybe.targets.Objective` class
- `baybe.strategies.Strategy` class

## [0.8.2] - 2024-03-27
### Added
- Simulation user guide
- Example for transfer learning backtesting utility
- `pyupgrade` pre-commit hook
- Better human readable `__str__` representation of objective and targets
- Alternative dataframe deserialization from `pd.DataFrame` constructors

### Changed
- More detailed and sophisticated search space user guide
- Support for Python 3.12
- Upgraded syntax to Python 3.9
- Bumped `onnx` version to fix vulnerability
- Increased threshold for low-dimensional GP priors
- Replaced `fit_gpytorch_mll_torch` with `fit_gpytorch_mll`
- Use `tox-uv` in pipelines

### Fixed
- `telemetry` dependency is no longer a group (enables Poetry installation)

## [0.8.1] - 2024-03-11
### Added
- Better human readable `__str__` representation of campaign
- README now contains an example on substance encoding results
- Transfer learning user guide
- `from_simplex` constructor now also takes and applies optional constraints

### Changed
- Full lookup backtesting example now tests different substance encodings
- Replaced unmaintained `mordred` dependency by `mordredcommunity`
- `SearchSpace`s now use `ndarray` instead of `Tensor`

### Fixed
- `from_simplex` now efficiently validated in `Campaign.validate_config`

## [0.8.0] - 2024-02-29
### Changed
- BoTorch dependency bumped to `>=0.9.3`

### Removed
- Workaround for BoTorch hybrid recommender data type
- Support for Python 3.8

## [0.7.4] - 2024-02-28
### Added
- Subpackages for the available recommender types
- Multi-style plotting capabilities for generated example plots
- JSON file for plotting themes
- Smoke testing in relevant tox environments
- `ContinuousParameter` base class
- New environment variable `BAYBE_CACHE_DIR` that can customize the disk cache directory
  or turn off disk caching entirely
- Options to control the number of nonzero parameters in `SubspaceDiscrete.from_simplex`
- Temporarily ignore ONNX vulnerabilities
- Better human readable `__str__` representation of search spaces
- `pretty_print_df` function for printing shortened versions of dataframes
- Basic Transfer Learning example
- Repo now has reminders (https://github.com/marketplace/actions/issue-reminder) enabled
- `mypy` for recommenders

### Changed
- `Recommender`s now share their core logic via their base class
- Remove progress bars in examples
- Strategies are now called `MetaRecommender`'s and part of the `recommenders.meta`
  module
- `Recommender`'s are now called `PureRecommender`'s and part of the `recommenders.pure`
  module
- `strategy` keyword of `Campaign` renamed to `recommender`
- `NaiveHybridRecommender` renamed to `NaiveHybridSpaceRecommender`

### Fixed
- Unhandled exception in telemetry when username could not be inferred on Windows
- Metadata is now correctly updated for hybrid spaces
- Unintended deactivation of telemetry due to import problem
- Line wrapping in examples

### Deprecations
- `TwoPhaseStrategy`, `SequentialStrategy` and `StreamingSequentialStrategy` have been
  replaced with their new `MetaRecommender` versions

## [0.7.3] - 2024-02-09
### Added
- Copy button for code blocks in documentation
- `mypy` for campaign, constraints and telemetry
- Top-level example summaries
- `RecommenderProtocol` as common interface for `Strategy` and `Recommender`
- `SubspaceDiscrete.from_simplex` convenience constructor

### Changed
- Order of README sections
- Imports from top level `baybe.utils` no longer possible
- Renamed `utils.numeric` to `utils.numerical`
- Optional `chem` dependencies are lazily imported, improving startup time

### Fixed
- Several minor issues in documentation
- Visibility and constructor exposure of `Campaign` attributes that should be private
- `TaskParameter`s no longer disappear from computational representation when the
  search space contains only one task parameter value
- Failing `baybe` import from environments containing only core dependencies caused by
  eagerly loading `chem` dependencies
- `tox` `coretest` now uses correct environment and skips unavailable tests
- Basic serialization example no longer requires optional `chem` dependencies

### Removed
- Detailed headings in table of contents of examples

### Deprecations
- Passing `numerical_measurements_must_be_within_tolerance` to the `Campaign` 
  constructor is no longer supported. Instead, `Campaign.add_measurements` now
  takes an additional parameter to control the behavior.
- `batch_quantity` replaced with `batch_size`
- `allow_repeated_recommendations` and `allow_recommending_already_measured` are now 
  attributes of `Recommender` and no longer attributes of `Strategy`

## [0.7.2] - 2024-01-24
### Added
- Target enums 
- `mypy` for targets and intervals
- Tests for code blocks in README and user guides
- `hypothesis` strategies and roundtrip tests for targets, intervals, and dataframes
- De-/serialization of target subclasses via base class
- Docs building check now part of CI
- Automatic formatting checks for code examples in documentation
- Deserialization of classes with classmethod constructors can now be customized
  by providing an optional `constructor` field
- `SearchSpace.from_dataframe` convenience constructor

### Changed
- Renamed `bounds_transform_func` target attribute to `transformation`
- `Interval.is_bounded` now implements the mathematical definition of boundedness
- Moved and renamed target transform utility functions
- Examples have two levels of headings in the table of content
- Fix orders of examples in table of content
- `DiscreteCustomConstraint` validator now expects dataframe instead of series
- `ignore_example` flag builds but does not execute examples when building documentation
- New user guide versions for campaigns, targets and objectives
- Binarization of dataframes now happens via pickling

### Fixed
- Wrong use of `tolerance` argument in constraints user guide
- Errors with generics and type aliases in documentation
- Deduplication bug in substance_data `hypothesis` strategy
- Use pydoclint as flake8 plugin and not as a stand-alone linter
- Margins in documentation for desktop and mobile version
- `Interval`s can now also be deserialized from a bounds iterable
- `SubspaceDiscrete` and `SubspaceContinuous` now have de-/serialization methods

### Removed
- Conda install instructions and version badge
- Early fail for different Python versions in regular pipeline

### Deprecations
- `Interval.is_finite` replaced with `Interval.is_bounded`
- Specifying target configs without explicit type information is deprecated
- Specifying parameters/constraints at the top level of a campaign configuration JSON is
  deprecated. Instead, an explicit `searchspace` field must be provided with an optional
  `constructor` entry

## [0.7.1] - 2023-12-07
### Added
- Release pipeline now also publishes source distributions
- `hypothesis` strategies and tests for parameters package

### Changed
- Reworked validation tests for parameters package
- `SubstanceParameter` now collects inconsistent user input in an `ExceptionGroup`

### Fixed
- Link handling in documentation

## [0.7.0] - 2023-12-04
### Added
- GitHub CI pipelines
- GitHub documentation pipeline
- Optional `--force` option for building the documentation despite errors
- Enabled passing optional arguments to `tox -e docs` calls
- Logo and banner images
- Project metadata for pyproject.toml
- PyPI release pipeline
- Favicon for homepage
- More literature references
- First drafts of first user guides

### Changed
- Reworked README for GitHub landing page
- Now has concise contribution guidelines
- Use Furo theme for documentation

### Removed
- `--debug` flag for documentation building

## [0.6.1] - 2023-11-27
### Added
- Script for building HTML documentation and corresponding `tox` environment
- Linter `typos` for spellchecking
- Parameter encoding enums
- `mypy` for parameters package
- `tox` environments for `mypy`

### Changed
- Replacing `pylint`, `flake8`, `µfmt` and `usort` with `ruff`
- Markdown based documentation replaced with HTML based documentation

### Fixed
- `encoding` is no longer a class variable
- Now installed with correct `pandas` dependency flag
- `comp_df` column names for `CustomDiscreteParameter` are now safe

## [0.6.0] - 2023-11-17
### Added
- `Raises` section for validators and corresponding contributing guideline
- Bring your own model: surrogate classes for custom model architectures and pre-trained ONNX models
- Test module for deprecation warnings
- Option to control the switching point of `TwoPhaseStrategy` (former `Strategy`)
- `SequentialStrategy` and `StreamingSequentialStrategy` classes
- Telemetry env variable `BAYBE_TELEMETRY_VPN_CHECK` turning the initial connectivity check on/off 
- Telemetry env variable `BAYBE_TELEMETRY_VPN_CHECK_TIMEOUT` for setting the connectivity check timeout

### Changed
- Reorganized modules into subpackages
- Serialization no longer relies on cattrs' global converter
- Refined (un-)structuring logic
- Telemetry env variable `BAYBE_TELEMETRY_HOST` renamed to `BAYBE_TELEMETRY_ENDPOINT`
- Telemetry env variable `BAYBE_DEBUG_FAKE_USERHASH` renamed to `BAYBE_TELEMETRY_USERNAME`
- Telemetry env variable `BAYBE_DEBUG_FAKE_HOSTHASH` renamed to `BAYBE_TELEMETRY_HOSTNAME`
- Bumped cattrs version

### Fixed
- Now supports Python 3.11
- Removed `pyarrow` version pin
- `TaskParameter` added to serialization test
- Deserialization (e.g. from config) no longer silently drops unknown arguments

### Deprecations
- `BayBE` class replaced with `Campaign`
- `baybe.surrogate` replaced with `baybe.surrogates`
- `baybe.targets.Objective` replaced with `baybe.objective.Objective`
- `baybe.strategies.Strategy` replaced with `baybe.strategies.TwoPhaseStrategy`

## [0.5.1] - 2023-10-19
### Added
- Linear in-/equality constraints over continuous parameters
- Constrained optimization for `SequentialGreedyRecommender`
- `RandomRecommender` now supports linear in-/equality constraints via polytope sampling

### Changed
- Include linting for all functions
- Rewrite functions to distinguish between private and public ones
- Unreachable telemetry endpoints now automatically disables telemetry and no longer cause
any data submission loops
- `add_fake_results` utility now considers potential target bounds
- Constraint names have been refactored to indicate whether they operate on discrete 
or continuous parameters

### Fixed
- Random recommendation failing for small discrete (sub-)spaces
- Deserialization issue with `TaskParameter`

## [0.5.0] - 2023-09-15
### Added
- `TaskParameter` for multitask modelling
- Basic transfer learning capability using multitask kernels
- Advanced simulation mechanisms for transfer learning and search space partitioning
- Extensive docstring documentation in all files
- Autodoc using sphinx
- Script for automatic code documentation
- New `tox` environments for a full and a core-only pytest run

### Changed
- Discrete subspaces require unique indices
- Simulation function signatures are redesigned (but largely backwards compatible)
- Docstring contents and style (numpy -> google)
- Regrouped additional dependencies

## [0.4.2] - 2023-08-29
### Added
- Test environments for multiple Python versions via `tox`

### Changed
- Removed `environment.yml`
- Telemetry host endpoint is now flexible via the environment variable `BAYBE_TELEMETRY_HOST`

### Fixed
- Inference for `__version__`

## [0.4.1] - 2023-08-23
### Added
- Vulnerability check via `pip-audit`
- `tests` dependency group

### Changed
- Removed no longer required `fsspec` dependency

### Fixed
- Scipy vulnerability by bumping version to 1.10.1
- Missing `pyarrow` dependency

## [0.4.0] - 2023-08-16
### Added
- `from_dataframe` convenience constructors for discrete and continuous subspaces 
- `from_bounds` convenience constructor for continuous subspaces
- `empty` convenience constructors discrete and continuous subspaces
- `baybe`, `strategies` and `utils` namespace for convenient imports
- Simple test for config validation
- `VarUCB` and `qVarUCB` acquisition functions emulating maximum variance for active learning
- Surrogate model serialization
- Surrogate model parameter passing

### Changed
- Renamed `create` constructors to `from_product`
- Renamed `empty` checks for subspaces to `is_empty`
- Fixed inconsistent class names in surrogate.py
- Fixed inconsistent class names in parameters.py
- Cached recommendations are now private
- Parameters, targets and objectives are now immutable
- Adjusted comments in example files
- Accelerated the slowest tests
- Removed try blocks from config examples
- Upgraded numpy requirement to >= 1.24.1
- Requires `protobuf<=3.20.3`
- `SearchSpace` parameters in surrogate models are now handled in `fit`
- Dataframes are encoded in binary for serialization
- `comp_rep` is loaded directly from the serialization string

### Fixed
- Include scaling in FPS recommender
- Support for pandas>=2.0.0

## [0.3.2] - 2023-07-24
### Added
- Constraints serialization

### Changed
- A maximum of one `DependenciesConstraint` is allowed
- Bumped numpy and matplotlib versions

## [0.3.1] - 2023-07-17
### Added
- Code coverage check with pytest-cov
- Hybrid mode for `SequentialGreedyRecommender`

### Changed
- Removed support for infinite parameter bounds
- Removed not yet implemented MULTI objective mode

### Fixed
- Changelog assert in Azure pipeline
- Bug: telemetry could not be fully deactivated

## [0.3.0] - 2023-06-27
### Added
- `Interval` class for representing parameter/target bounds
- Activated mypy for the first few modules and fixed their type issues
- Automatic (de-)serialization and `SerialMixin` class
- Basic serialization example, demo and tests
- Mechanisms for loading and validating config files
- Telemetry via OpenTelemetry
- More detailed package installation info
- Fallback mechanism for `NonPredictiveRecommender`
- Introduce naive hybrid recommender

### Changed
- Switched from pydantic to attrs in all modules except constraints.py
- Removed subclass initialization hooks and `type` attribute
- Refactored class attributes and their conversion/validation/initialization
- Removed no longer needed `HashableDict` class
- Refactored strategy and recommendation module structures
- Replaced dict-based configuration logic with object-based logic
- Overall versioning scheme and version inference for telemetry
- No longer using private telemetry imports
- Fixed package versions for dev tools
- Revised "Getting Started" section in README.md
- Revised examples

### Fixed
- Telemetry no longer crashing when package was not installed

## [0.2.4] - 2023-03-24
### Added
- Tests for different search space types and their compatible recommenders

### Changed
- Initial strategies converted to recommenders
- Config keyword `initial_strategy` replaced by `initial_recommender_cls`
- Config keywords for the clustering recommenders changed from `x` to `CLUSTERING_x`
- skicit-learn-extra is now optional dependency in the [extra] group
- Type identifiers of greedy recommenders changed to 'SEQUENTIAL_GREEDY_x'

### Fixed
- Parameter bounds now only contain dimensions that actually appear in the search space 

## [0.2.3] - 2023-03-14
### Added
- Parsing for continuous parameters
- Caching of recommendations to avoid unnecessary computations
- Strategy support for hybrid spaces
- Custom discrete constraint with user-provided validator

### Changed
- Parameter class hierarchy
- `SearchSpace` has now a discrete and continuous subspace
- Model fit now done upon requesting recommendations

### Fixed
- Updated BoTorch and GPyTorch versions are also used in pyproject.toml

## [0.2.2] - 2023-01-13
### Added
- `SearchSpace` class
- Code testing with pytest
- Option to specify initial data for backtesting simulations
- SequentialGreedyRecommender class

### Changed
- Switched from miniconda to micromamba in Azure pipeline

### Fixed
- BoTorch version upgrade to fix critical bug (https://github.com/pytorch/botorch/pull/1454)

## [0.2.1] - 2022-12-01
### Fixed
- Parameters cannot be initialized with duplicate values

## [0.2.0] - 2022-11-10
### Added
- Initial strategy: Farthest Point Sampling
- Initial strategy: Partitioning Around Medoids
- Initial strategy: K-means
- Initial strategy: Gaussian Mixture Model
- Constraints and conditions for discrete parameters
- Data scaling functionality
- Decorator for automatic model scaling
- Decorator for handling constant targets
- Decorator for handling batched model input
- Surrogate model: Mean prediction
- Surrogate model: Random forrest
- Surrogate model: NGBoost
- Surrogate model: Bayesian linear
- Save/load functionality for BayBE objects

### Fixed
- UCB now usable as acquisition function, hard-set beta parameter to 1.0
- Temporary GP priors now exactly reproduce EDBO setting

## [0.1.0] - 2022-10-01
### Added
- Code skeleton with a central object to access functionality
- Basic parser for categorical parameters with one-hot encoding
- Basic parser for discrete numerical parameters
- Azure pipeline for code formatting and linting
- Single-task Gaussian process strategy
- Streamlit dashboard for comparing single-task strategies
- Input functionality to read measurements including automatic matching to search space
- Integer encoding for categorical parameters
- Parser for numerical discrete parameters
- Single numerical target with Min and Max mode
- Recommendation functionality
- Parameter scaling depending on parameter types and user-chosen scalers
- Noise and fake-measurement utilities
- Internal metadata storing various info about datapoints in the search space
- BayBE options controlling recommendation and data addition behavior
- Config parsing and validation using pydantic
- Global random seed control
- Strategy connection with BayBE object
- Custom parameters as labels with user-provided encodings
- Substance parameters which are encoded via cheminformatics descriptors
- Data cleaning utilities useful for descriptors
- Simulation capabilities for testing the package on existing data
- Parsing and preprocessing for multiple targets / desirability ansatz
- Basic README file
- Automatic publishing of tagged versions
- Caching of experimental parameters and chemical descriptors
- Choices for acquisition functions and their usage with arbitrary surrogate models
- Temporary logic for selecting GP priors
