# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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
