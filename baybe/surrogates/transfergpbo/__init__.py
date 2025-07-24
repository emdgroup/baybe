"""Surrogates from "Transfer Learning with GPs for BO" by Tighineanu et al. (2022)."""

from baybe.surrogates.transfergpbo.torchmodels import MHGPModel, MHGPModelStable

__all__ = ["MHGPModel", "MHGPModelStable"]
