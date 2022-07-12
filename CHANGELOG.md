# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- Azure pipeline for code formatting and linting.
- Single-task Gaussian process strategy.
- Streamlit dashboard for comparing single-task strategies.
- Input functionality to read measurements including automatic matching to searchspace.
- Categorical parameters with either one-hot or integer encoding.
- Numerical discrete parameters.
- Single numerical target with Min and Max mode.
- Recommendation functionality.
- Parameter scaling depending on parameter types and user-chosen scalers.
- Noise and fake-measurement utilities.
- Internal metadata storing various info about datapoints in the searchspace.
- BayBE options controlling recommendation and data addition behavior.