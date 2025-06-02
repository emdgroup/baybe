"""BoTorch preset for Gaussian process surrogates."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from attrs import define
from typing_extensions import override

from baybe.parameters import TaskParameter
from baybe.surrogates.gaussian_process.model_factory import ModelFactory

if TYPE_CHECKING:
    from botorch.models.model import Model
    from botorch.models.transforms.input import InputTransform
    from botorch.models.transforms.outcome import OutcomeTransform
    from gpytorch.likelihoods import Likelihood
    from gpytorch.means import Mean
    from torch import Tensor

    from baybe.searchspace.core import SearchSpace
    from baybe.surrogates.gaussian_process.kernel_factory import KernelFactory


@define
class BotorchModelFactory(ModelFactory):
    """A model factory with BoTorch defaults for Gaussian process surrogates."""

    @override
    def __call__(
        self,
        searchspace: SearchSpace,
        train_x: Tensor,
        train_y: Tensor,
        input_transform: InputTransform,
        outcome_transform: OutcomeTransform | None = None,
        kernel_factory: KernelFactory | None = None,
        mean_module: Mean | None = None,
        likelihood: Likelihood | None = None,
    ) -> Model:
        import botorch
        import torch

        is_multitask = searchspace.task_idx is not None

        if is_multitask and outcome_transform is None:
            # TODO See https://github.com/pytorch/botorch/issues/2739
            if train_y.shape[-1] != 1:
                raise NotImplementedError(
                    "Task-stratified output transform currently does not support"
                    + "multiple outputs."
                )
            outcome_transform = botorch.models.transforms.outcome.StratifiedStandardize(
                task_values=torch.tensor(
                    [p for p in searchspace.parameters if isinstance(p, TaskParameter)][
                        0
                    ].comp_df.values.ravel(),
                    dtype=torch.long,
                ),
                stratification_idx=searchspace.task_idx,
            )

        # define the covariance module for the numeric dimensions
        if kernel_factory is not None:
            base_covar_module = kernel_factory(
                searchspace, train_x, train_y
            ).to_gpytorch(
                ard_num_dims=train_x.shape[-1]
                - (1 if searchspace.task_idx is not None else 0),
                batch_shape=train_x.shape[:-2],
            )
        else:
            base_covar_module = None
        # The active_dims parameter is omitted as it is not needed for both
        # - single-task SingleTaskGP: all features are used
        # - multi-task MultiTaskGP: the model splits task and non-task features
        #   before passing them to the covariance kernel

        if not is_multitask:
            model_cls = botorch.models.SingleTaskGP
            model_kwargs = {}
        else:
            model_cls = botorch.models.MultiTaskGP
            # TODO
            #  It is assumed that there is only one task parameter with only
            #  one active value.
            #  One active task value is required for MultiTaskGP as else
            #  one posterior per task would be returned:
            #  https://github.com/pytorch/botorch/blob/a018a5ffbcbface6229d6c39f7ac6ef9baf5765e/botorch/models/gpytorch.py#L951
            # TODO
            #  The below code implicitly assumes there is single task parameter,
            #  which is already checked in the SearchSpace.
            task_param = [
                p
                for p in searchspace.discrete.parameters
                if isinstance(p, TaskParameter)
            ][0]
            if len(task_param.active_values) > 1:
                raise NotImplementedError(
                    "Does not support multiple active task values."
                )
            model_kwargs = {
                "task_feature": searchspace.task_idx,
                "output_tasks": [
                    task_param.comp_df.at[task_param.active_values[0], task_param.name]
                ],
                "rank": searchspace.n_tasks,
                "task_covar_prior": None,
                "all_tasks": task_param.comp_df[task_param.name].astype(int).to_list(),
            }

        if outcome_transform is not None:
            # The outcome_transform must be passed optionally rather than None
            # if default should be used as None is interpreted as no transform
            model_kwargs["outcome_transform"] = outcome_transform

        return model_cls(
            train_x,
            train_y,
            input_transform=input_transform,
            mean_module=mean_module,
            covar_module=base_covar_module,
            likelihood=likelihood,
            **model_kwargs,
        )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
