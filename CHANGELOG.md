# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `from_dataframe` convenience constructors for discrete and continuous subspaces 
- `from_bounds` convenience constructor for continuous subspaces

### Changed
- Renamed `create` constructors to `from_product`
- Cached recommendations are now a private attribute
- Parameters, targets and objectives are now immutable

## [0.3.2] - 2023-07-24
### Added
- Constraints serialization

### Changed
- A maxiumum of one `DependenciesConstraint` is allowed
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
- Fallback mechanism for NonPredictiveRecommenders
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
