# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Code skeleton with a central object to access functionality.
- Basic parser for categorical parameters with one-hot encoding.
- Basic parser for discrete numerical parameters.
- Azure pipeline for code formatting and linting.
- Single-task Gaussian process strategy.
- Streamlit dashboard for comparing single-task strategies.
- Input functionality to read measurements including automatic matching to searchspace.
- Integer encoding for categorical parameters.
- Parser for numerical discrete parameters.
- Single numerical target with Min and Max mode.
- Recommendation functionality.
- Parameter scaling depending on parameter types and user-chosen scalers.
- Noise and fake-measurement utilities.
- Internal metadata storing various info about datapoints in the searchspace.
- BayBE options controlling recommendation and data addition behavior.
- Config parsing and validation using pydantic.
- Global random seed control.
- Strategy connection with BayBE object.
- Custom parameters as labels with user-provided encodings.
- Substance parameters which are encoded via cheminformatics descriptors.
- Data cleaning utilities useful for descriptors.
- Simulation capabilities for testing the package on existing data.
- Parsing and preprocessing for multiple targets / desirability ansatz.
- Basic README file.
