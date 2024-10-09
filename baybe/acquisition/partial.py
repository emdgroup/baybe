"""Wrapper functionality for hybrid spaces."""

import gc

import torch
from attr import define
from botorch.acquisition import AcquisitionFunction as BotorchAcquisitionFunction
from torch import Tensor


@define
class PartialAcquisitionFunction:
    """Acquisition function for evaluating points in a hybrid search space.

    It can either pin the discrete or the continuous part. The pinned part is assumed
    to be a tensor of dimension ``d x 1`` where d is the computational dimension of
    the search space that is to be pinned. The acquisition function is assumed to be
    defined for the full hybrid space.
    """

    botorch_acqf: BotorchAcquisitionFunction
    """The acquisition function for the hybrid space."""

    pinned_part: Tensor
    """The values that will be attached whenever evaluating the acquisition function."""

    pin_discrete: bool
    """A flag for denoting whether ``pinned_part`` corresponds to the discrete
    subspace."""

    def _lift_partial_part(self, partial_part: Tensor) -> Tensor:
        """Lift ``partial_part`` to the original hybrid space.

        Depending on whether the discrete or the variable part of the search space is
        pinned, this function identifies whether the partial_part is the continuous
        or discrete part and then constructs the full tensor accordingly.

        Args:
            partial_part: The part of the tensor that is to be evaluated in the partial
                space

        Returns:
            The full point in the hybrid space.
        """
        # Might be necessary to insert a dummy dimension
        if partial_part.ndim == 2:
            partial_part = partial_part.unsqueeze(-2)
        # Repeat the pinned part such that it matches the dimension of the partial_part
        pinned_part = self.pinned_part.repeat(
            (partial_part.shape[0], partial_part.shape[1], 1)
        )
        # Check which part is discrete and which is continuous
        if self.pin_discrete:
            disc_part = pinned_part
            cont_part = partial_part
        else:
            disc_part = partial_part
            cont_part = pinned_part
        # Concat the parts and return the concatenated point
        full_point = torch.cat((disc_part, cont_part), -1)
        return full_point

    def __call__(self, variable_part: Tensor) -> Tensor:
        """Lift the point to the hybrid space and evaluate the acquisition function.

        Args:
            variable_part: The part that should be lifted.

        Returns:
            The evaluation of the lifted point in the full hybrid space.
        """
        full_point = self._lift_partial_part(variable_part)
        return self.botorch_acqf(full_point)

    def __getattr__(self, item):
        return getattr(self.botorch_acqf, item)

    def set_X_pending(self, X_pending: Tensor | None):
        """Inform the acquisition function about pending design points.

        Enhances the original ``set_X_pending`` function from the full acquisition
        function as we need to store the full point, i.e., the point in the hybrid space
        for the ``PartialAcquisitionFunction`` to work properly.

        Args:
            X_pending: ``n x d`` Tensor with n d-dim design points that have been
                submitted for evaluation but have not yet been evaluated.
        """
        if X_pending is not None:  # Lift point to hybrid space and add additional dim
            X_pending = self._lift_partial_part(X_pending)
            X_pending = torch.squeeze(X_pending, -2)
        # Now use the original set_X_pending function
        self.botorch_acqf.set_X_pending(X_pending)


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
