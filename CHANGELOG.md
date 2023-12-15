# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Target enums 
- `mypy` for targets package

### Changed
- Renamed `bounds_transform_func` target attribute to `transformation`
- Moved and renamed target transform utility functions

### Removed
- Conda install instructions and version badge

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

### Fixed
- `encoding` is no longer a class variable
- Now installed with correct `pandas` dependency flag
- `comp_df` column names for `CustomDiscreteParameter` are now safe

### Deprecations
- Markdown based documentation replaced with HTML based documentation

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
- Test environments for multiple python versions via `tox`

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
- SearchSpace has now a discrete and continuous subspace
- Model fit now done upon requesting recommendations

### Fixed
- Updated BoTorch and GPyTorch versions are also used in pyproject.toml

## [0.2.2] - 2023-01-13
### Added
- SearchSpace class
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
- Input functionality to read measurements including automatic matching to searchspace
- Integer encoding for categorical parameters
- Parser for numerical discrete parameters
- Single numerical target with Min and Max mode
- Recommendation functionality
- Parameter scaling depending on parameter types and user-chosen scalers
- Noise and fake-measurement utilities
- Internal metadata storing various info about datapoints in the searchspace
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
