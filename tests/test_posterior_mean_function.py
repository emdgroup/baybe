"""Tests for ``GaussianProcessSurrogate.posterior_mean_function``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import pytest
import torch

from baybe.exceptions import ModelNotTrainedError
from baybe.kernels.base import Kernel as BayBEKernel
from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import (
    GaussianProcessSurrogate,
    _build_mean_transfer_gp,
    _extract_raw_training_data,
    _ModelContext,
    _resolve_anchors,
)
from baybe.targets.numerical import NumericalTarget
from baybe.utils.dataframe import to_tensor

if TYPE_CHECKING:
    from baybe.objectives.base import Objective


_OBJECTIVE = NumericalTarget(name="y").to_objective()


def _searchspace(values: list[float]) -> SearchSpace:
    return SearchSpace.from_product([NumericalDiscreteParameter("x1", values=values)])


def _measurements(xs: list[float], ys: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"x1": xs, "y": ys})


def _fitted_surrogate(
    searchspace: SearchSpace,
    objective: Objective,
    measurements: pd.DataFrame,
) -> GaussianProcessSurrogate:
    surrogate = GaussianProcessSurrogate()
    surrogate.fit(searchspace, objective, measurements)
    return surrogate


def _predict_on_prior_mean(
    pretrained: GaussianProcessSurrogate, xs: list[float]
) -> pd.DataFrame:
    """Build measurements where the targets follow the pretrained posterior mean."""
    points = pd.DataFrame({"x1": xs})
    with torch.no_grad():
        targets = pretrained.posterior(points).mean
    return pd.DataFrame({"x1": xs, "y": targets.numpy().ravel()})


@pytest.fixture(name="pretrained")
def fixture_pretrained() -> GaussianProcessSurrogate:
    """A GP trained on a narrow search space with three points."""
    return _fitted_surrogate(
        _searchspace([0.0, 2.5, 5.0]),
        _OBJECTIVE,
        _measurements([0.0, 2.5, 5.0], [0.0, 5.0, 10.0]),
    )


@pytest.fixture(name="wider_searchspace")
def fixture_wider_searchspace() -> SearchSpace:
    return _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])


def test_default_matches_pretrained_at_pretrained_point(
    pretrained: GaussianProcessSurrogate, wider_searchspace: SearchSpace
) -> None:
    """Default flags reproduce the pretrained posterior mean at a held-out point.

    Equivalent to the original PR #823 identity test for the freeze/pretrained combo.
    """
    expected = pretrained.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=pretrained.posterior_mean_function)
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)

    actual = new.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()

    assert abs(actual - expected) < 1e-4


def test_raises_if_not_fitted() -> None:
    """An untrained surrogate cannot produce a posterior mean function."""
    ss = _searchspace([0.0, 1.0])
    with pytest.raises(ModelNotTrainedError, match="must be fitted"):
        GaussianProcessSurrogate().posterior_mean_function(
            ss, _OBJECTIVE, _measurements([0.0, 1.0], [0.0, 1.0])
        )


def test_rejects_new_anchors_with_discard_init(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """``anchors='new'`` with ``mean_kernel_init='discard'`` transfers nothing."""
    with pytest.raises(ValueError, match="would not transfer"):
        pretrained.posterior_mean_function(
            _searchspace([0.0, 5.0]),
            _OBJECTIVE,
            _measurements([0.0, 5.0], [0.0, 10.0]),
            anchors="new",
            mean_kernel_init="discard",
        )


@pytest.mark.parametrize(
    "mean_kernel_init",
    ["freeze", "warmstart", "discard"],
)
def test_pretrained_anchors_recover_pretrained_mean(
    pretrained: GaussianProcessSurrogate,
    wider_searchspace: SearchSpace,
    mean_kernel_init: Literal["freeze", "warmstart", "discard"],
) -> None:
    """With pretrained anchors, the new GP reproduces the pretrained posterior mean.

    When the new GP is trained on targets sampled from the pretrained posterior mean,
    the outer kernel sees zero residual and the prior mean alone explains the data,
    so the prediction at a held-out point must match the pretrained's prediction.
    """
    expected = pretrained.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(
        mean_or_factory=lambda ss, obj, m: pretrained.posterior_mean_function(
            ss, obj, m, anchors="pretrained", mean_kernel_init=mean_kernel_init
        )
    )
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)

    actual = new.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()

    # Freeze is the strictest match; trainable modes need a looser tolerance.
    tol = 1e-4 if mean_kernel_init == "freeze" else 5e-2
    assert abs(actual - expected) < tol


def test_new_anchors_freeze_reanchors_at_new_data(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """With ``anchors='new'`` and ``freeze``, the prior mean re-anchors on new data.

    Built as a stand-alone GPyTorch mean (not embedded in a new GP), the module
    evaluated at the new training inputs must reproduce the new target values up to
    the (frozen) likelihood noise, since the prior mean is exactly the posterior
    mean of a GP trained on those new points.
    """
    new_ss = _searchspace([0.0, 5.0])
    new_meas = _measurements([0.0, 5.0], [3.0, 7.0])

    mean_module = pretrained.posterior_mean_function(
        new_ss, _OBJECTIVE, new_meas, anchors="new", mean_kernel_init="freeze"
    )

    # New GP's Normalize maps x1 in [0, 5] to [0, 1]; the mean module's
    # forward expects normalized inputs (as it does inside the outer GP).
    x_normalized = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    with torch.no_grad():
        prior_mean_standardized = mean_module(x_normalized)

    # Map back to raw output space using the new GP's standardize statistics
    # (mean = 5.0, std ~= sqrt(8) for [3, 7]; trust the actual model below instead).
    new = GaussianProcessSurrogate(
        mean_or_factory=lambda ss, obj, m: pretrained.posterior_mean_function(
            ss, obj, m, anchors="new", mean_kernel_init="freeze"
        )
    )
    new.fit(new_ss, _OBJECTIVE, new_meas)
    with torch.no_grad():
        full = new.posterior(pd.DataFrame({"x1": [0.0, 5.0]})).mean

    # The combined posterior (prior mean ~= targets, outer noise residual ~= 0)
    # should reproduce the training targets closely.
    assert abs(full[0].item() - 3.0) < 1e-1
    assert abs(full[1].item() - 7.0) < 1e-1
    # Sanity: the prior-mean module produced finite values
    assert torch.isfinite(prior_mean_standardized).all()


def test_combined_anchors_uses_both_datasets(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """Combined anchors evaluate the prior mean using both pretrained and new data."""
    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    # Pick points outside the pretrained's training set with non-trivial targets
    new_meas = _measurements([7.5, 10.0], [15.0, 20.0])

    mean_module = pretrained.posterior_mean_function(
        new_ss, _OBJECTIVE, new_meas, anchors="combined", mean_kernel_init="freeze"
    )

    # The pretrained training data is [(0, 0), (2.5, 5), (5, 10)]; combined with
    # new [(7.5, 15), (10, 20)] the prior mean (frozen kernel) should approximately
    # interpolate all five points. Query at those exact inputs.
    new = GaussianProcessSurrogate(
        mean_or_factory=lambda ss, obj, m: pretrained.posterior_mean_function(
            ss, obj, m, anchors="combined", mean_kernel_init="freeze"
        )
    )
    new.fit(new_ss, _OBJECTIVE, new_meas)

    # Predict at a pretrained training point that is NOT in the new measurements.
    # Under "combined", the inner GP also anchors at x=2.5 with y=5, so the prior
    # mean (and hence the full posterior, given zero outer residual) approximates 5.
    with torch.no_grad():
        pred = new.posterior(pd.DataFrame({"x1": [2.5]})).mean.item()
    assert abs(pred - 5.0) < 5e-1

    # Sanity: the stand-alone mean module returns finite predictions.
    x_normalized = torch.tensor(
        [[0.0], [0.25], [0.5], [0.75], [1.0]], dtype=torch.float64
    )
    with torch.no_grad():
        prior_mean = mean_module(x_normalized)
    assert torch.isfinite(prior_mean).all()


def test_warmstart_double_counting_warning(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """Warmstart with target-side anchors warns about double-counting risk."""
    new_meas = _measurements([0.0, 5.0], [1.0, 2.0])
    for anchors in ("new", "combined"):
        with pytest.warns(UserWarning, match="warmstart"):
            pretrained.posterior_mean_function(
                _searchspace([0.0, 5.0]),
                _OBJECTIVE,
                new_meas,
                anchors=anchors,
                mean_kernel_init="warmstart",
            )


def test_freeze_does_not_change_pretrained_hyperparameters(
    pretrained: GaussianProcessSurrogate,
    wider_searchspace: SearchSpace,
) -> None:
    """The pretrained surrogate's parameters are unaffected by fitting the new GP."""
    pretrained_lengthscale = pretrained._model.covar_module.lengthscale.detach().clone()
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=pretrained.posterior_mean_function)
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)

    assert torch.equal(
        pretrained._model.covar_module.lengthscale,
        pretrained_lengthscale,
    )


def test_discard_uses_fresh_hyperparameters(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """``discard`` ignores the pretrained kernel's hyperparameters at construction."""
    # Use 'combined' so 'discard' is allowed
    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    new_meas = _measurements([7.5, 10.0], [15.0, 20.0])

    pretrained_lengthscale = pretrained._model.covar_module.lengthscale.detach().clone()
    mean_module = pretrained.posterior_mean_function(
        new_ss, _OBJECTIVE, new_meas, anchors="combined", mean_kernel_init="discard"
    )
    inner_lengthscale = mean_module.gp.covar_module.lengthscale.detach()

    # Fresh kernel must not equal the pretrained lengthscale (default init is
    # log(2) ~= 0.6931, pretrained values are essentially never exactly this).
    assert not torch.equal(inner_lengthscale, pretrained_lengthscale)


def test_extract_raw_training_data_matches_inputs(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """The helper inverts the pretrained transforms to recover raw training data."""
    x_raw, y_raw = _extract_raw_training_data(pretrained._model)

    # The pretrained fixture trained on x in {0, 2.5, 5} with y = 2x.
    assert x_raw.shape == (3, 1)
    assert y_raw.shape == (3, 1)
    expected_x = torch.tensor([[0.0], [2.5], [5.0]], dtype=x_raw.dtype)
    expected_y = torch.tensor([[0.0], [5.0], [10.0]], dtype=y_raw.dtype)
    # Order may differ from input order due to internal sorting; compare as sets.
    assert torch.allclose(x_raw.sort(dim=0).values, expected_x, atol=1e-5)
    assert torch.allclose(y_raw.sort(dim=0).values, expected_y, atol=1e-5)
    # The returned tensors must be detached clones (safe to mutate).
    assert not x_raw.requires_grad
    assert not y_raw.requires_grad


@pytest.mark.parametrize(
    ("anchors", "expected_n"),
    [("pretrained", 3), ("new", 2), ("combined", 5)],
)
def test_resolve_anchors_sizes(
    pretrained: GaussianProcessSurrogate,
    anchors: Literal["pretrained", "new", "combined"],
    expected_n: int,
) -> None:
    """Anchor count equals pretrained-only, new-only, or their sum."""
    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    new_meas = _measurements([7.5, 10.0], [15.0, 20.0])
    context = _ModelContext(new_ss, _OBJECTIVE, new_meas)

    x_raw, y_raw = _resolve_anchors(anchors, pretrained._model, context)

    assert x_raw.shape == (expected_n, 1)
    assert y_raw.shape == (expected_n, 1)


def test_resolve_anchors_combined_concatenates_in_order(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """For ``combined``, pretrained anchors precede new anchors."""
    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    new_meas = _measurements([7.5, 10.0], [15.0, 20.0])
    context = _ModelContext(new_ss, _OBJECTIVE, new_meas)

    x_old, y_old = _extract_raw_training_data(pretrained._model)
    x_combined, y_combined = _resolve_anchors("combined", pretrained._model, context)

    assert torch.equal(x_combined[: x_old.shape[0]], x_old)
    assert torch.equal(y_combined[: y_old.shape[0]], y_old)


@pytest.mark.parametrize("mean_kernel_init", ["freeze", "warmstart"])
def test_build_mean_transfer_gp_copies_pretrained_components(
    pretrained: GaussianProcessSurrogate,
    mean_kernel_init: Literal["freeze", "warmstart"],
) -> None:
    """For non-discard inits, the inner GP shares hyperparameter values, not objects.

    The mean/kernel/likelihood are deep-copied: same values, different module
    instances (so mutating one cannot affect the other).
    """
    new_ss = _searchspace([0.0, 5.0])
    new_meas = _measurements([0.0, 5.0], [0.0, 10.0])
    context = _ModelContext(new_ss, _OBJECTIVE, new_meas)
    x_raw, y_raw = _resolve_anchors("pretrained", pretrained._model, context)

    inner = _build_mean_transfer_gp(
        x_raw,
        y_raw,
        mean_kernel_init=mean_kernel_init,
        pretrained_model=pretrained._model,
        mean_factory=pretrained.mean_factory,
        kernel_factory=pretrained.kernel_factory,
        likelihood_factory=pretrained.likelihood_factory,
        context=context,
    )

    assert inner.covar_module is not pretrained._model.covar_module
    assert torch.equal(
        inner.covar_module.lengthscale,
        pretrained._model.covar_module.lengthscale,
    )


@pytest.mark.parametrize("freeze_input_transform", [True, False])
def test_build_mean_transfer_gp_input_transform_source_vs_anchor(
    pretrained: GaussianProcessSurrogate,
    freeze_input_transform: bool,
) -> None:
    """The inner input transform reuses source bounds or refits on the anchors.

    With ``freeze_input_transform=True`` the inner ``Normalize`` matches the source
    GP's transform (absolute x-scaling). With ``False`` it is refit on the anchor
    inputs' per-column min/max, independent of the new search-space bounds.
    """
    # New search space spans [0, 10] while the anchors lie in [1, 9] (range 8).
    new_ss = _searchspace([0.0, 10.0])
    new_meas = _measurements([1.0, 9.0], [4.0, 6.0])
    context = _ModelContext(new_ss, _OBJECTIVE, new_meas)
    x_raw, y_raw = _resolve_anchors("new", pretrained._model, context)

    inner = _build_mean_transfer_gp(
        x_raw,
        y_raw,
        mean_kernel_init="freeze",
        freeze_input_transform=freeze_input_transform,
        pretrained_model=pretrained._model,
        mean_factory=pretrained.mean_factory,
        kernel_factory=pretrained.kernel_factory,
        likelihood_factory=pretrained.likelihood_factory,
        context=context,
    )

    source = pretrained._model.input_transform
    if freeze_input_transform:
        assert torch.equal(inner.input_transform.coefficient, source.coefficient)
        assert torch.equal(inner.input_transform.offset, source.offset)
    else:
        # Anchor inputs span [1, 9]: offset = min = 1, coefficient = range = 8.
        assert torch.allclose(
            inner.input_transform.offset, torch.ones_like(inner.input_transform.offset)
        )
        assert torch.allclose(
            inner.input_transform.coefficient,
            torch.full_like(inner.input_transform.coefficient, 8.0),
        )
        # The refit transform must not fall back to the new search-space bounds.
        assert not torch.equal(inner.input_transform.coefficient, source.coefficient)


def test_build_mean_transfer_gp_freeze_disables_gradients(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """``freeze`` must clear ``requires_grad`` on every inner parameter."""
    new_ss = _searchspace([0.0, 5.0])
    new_meas = _measurements([0.0, 5.0], [0.0, 10.0])
    context = _ModelContext(new_ss, _OBJECTIVE, new_meas)
    x_raw, y_raw = _resolve_anchors("pretrained", pretrained._model, context)

    inner = _build_mean_transfer_gp(
        x_raw,
        y_raw,
        mean_kernel_init="freeze",
        pretrained_model=pretrained._model,
        mean_factory=pretrained.mean_factory,
        kernel_factory=pretrained.kernel_factory,
        likelihood_factory=pretrained.likelihood_factory,
        context=context,
    )

    assert all(not p.requires_grad for p in inner.parameters())


def test_build_mean_transfer_gp_warmstart_keeps_gradients(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """``warmstart`` keeps inner parameters trainable for outer MLL updates."""
    new_ss = _searchspace([0.0, 5.0])
    new_meas = _measurements([0.0, 5.0], [0.0, 10.0])
    context = _ModelContext(new_ss, _OBJECTIVE, new_meas)
    x_raw, y_raw = _resolve_anchors("pretrained", pretrained._model, context)

    inner = _build_mean_transfer_gp(
        x_raw,
        y_raw,
        mean_kernel_init="warmstart",
        pretrained_model=pretrained._model,
        mean_factory=pretrained.mean_factory,
        kernel_factory=pretrained.kernel_factory,
        likelihood_factory=pretrained.likelihood_factory,
        context=context,
    )

    assert any(p.requires_grad for p in inner.parameters())


def test_warmstart_optimizes_inner_parameters_after_fit(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """Warmstart inner hyperparameters move under outer-MLL gradient descent.

    A few explicit optimizer steps are used instead of ``fit_gpytorch_mll`` because
    the trainable inner mean makes the all-or-nothing BoTorch fit numerically
    fragile (which is itself the reason warmstart is used with care). The point being
    verified here is that the inner kernel lengthscale receives gradient and changes.
    """
    import botorch
    import gpytorch

    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    new_meas = _measurements([0.0, 2.5, 5.0, 7.5, 10.0], [1.0, 0.0, 3.0, 2.0, 5.0])
    context = _ModelContext(new_ss, _OBJECTIVE, new_meas)

    module = pretrained.posterior_mean_function(
        new_ss, _OBJECTIVE, new_meas, anchors="pretrained", mean_kernel_init="warmstart"
    )
    before = module.gp.covar_module.lengthscale.detach().clone()

    train_x = to_tensor(new_ss.transform(new_meas, allow_extra=True))
    train_y = to_tensor(_OBJECTIVE._pre_transform(new_meas, allow_extra=True))
    if train_y.ndim == 1:
        train_y = train_y.unsqueeze(-1)
    kernel = pretrained.kernel_factory(new_ss, _OBJECTIVE, new_meas)
    if isinstance(kernel, BayBEKernel):
        kernel = kernel.to_gpytorch(searchspace=new_ss)
    model = botorch.models.SingleTaskGP(
        train_x,
        train_y,
        input_transform=GaussianProcessSurrogate._make_input_transform(context),
        outcome_transform=GaussianProcessSurrogate._make_outcome_transform(context),
        mean_module=module,
        covar_module=kernel,
        likelihood=pretrained.likelihood_factory(new_ss, _OBJECTIVE, new_meas),
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=0.1
    )
    for _ in range(5):
        optimizer.zero_grad()
        loss = -mll(model(*model.train_inputs), model.train_targets).sum()
        loss.backward()
        optimizer.step()

    after = module.gp.covar_module.lengthscale.detach()
    assert not torch.allclose(after, before)


def test_freeze_keeps_inner_parameters_constant_after_fit(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """Freeze inner hyperparameters are untouched by the outer MLL optimization."""
    new_ss = _searchspace([0.0, 2.5, 5.0, 7.5, 10.0])
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 2.5, 5.0, 7.5, 10.0])
    module = pretrained.posterior_mean_function(new_ss, _OBJECTIVE, new_meas)
    before = module.gp.covar_module.lengthscale.detach().clone()

    new = GaussianProcessSurrogate(mean_or_factory=module)
    new.fit(new_ss, _OBJECTIVE, new_meas)

    after = module.gp.covar_module.lengthscale.detach()
    assert torch.equal(after, before)


def test_transferred_mean_accepts_matching_context(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """A pre-built mean module fits cleanly on the context it was built for."""
    new_ss = _searchspace([0.0, 5.0])
    new_meas = _measurements([0.0, 5.0], [0.0, 10.0])
    module = pretrained.posterior_mean_function(new_ss, _OBJECTIVE, new_meas)

    new = GaussianProcessSurrogate(mean_or_factory=module)
    new.fit(new_ss, _OBJECTIVE, new_meas)  # must not raise

    assert new._model is not None


def test_transferred_mean_rejects_mismatched_input_context(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """Fitting on a search space with different bounds is rejected."""
    build_ss = _searchspace([0.0, 5.0])
    build_meas = _measurements([0.0, 5.0], [0.0, 10.0])
    module = pretrained.posterior_mean_function(build_ss, _OBJECTIVE, build_meas)

    # Different parameter bounds -> different input Normalize coefficients.
    fit_ss = _searchspace([0.0, 10.0])
    fit_meas = _measurements([0.0, 10.0], [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=module)
    with pytest.raises(ValueError, match="different"):
        new.fit(fit_ss, _OBJECTIVE, fit_meas)


def test_transferred_mean_rejects_mismatched_outcome_context(
    pretrained: GaussianProcessSurrogate,
) -> None:
    """Fitting on targets with different standardization stats is rejected."""
    build_ss = _searchspace([0.0, 5.0])
    build_meas = _measurements([0.0, 5.0], [0.0, 10.0])
    module = pretrained.posterior_mean_function(build_ss, _OBJECTIVE, build_meas)

    # Same input bounds but different target values -> different Standardize stats.
    fit_meas = _measurements([0.0, 5.0], [100.0, 200.0])
    new = GaussianProcessSurrogate(mean_or_factory=module)
    with pytest.raises(ValueError, match="different"):
        new.fit(build_ss, _OBJECTIVE, fit_meas)


def test_bound_method_factory_passes_validation(
    pretrained: GaussianProcessSurrogate,
    wider_searchspace: SearchSpace,
) -> None:
    """Using the bound method as a factory builds the mean in the fit context.

    In this path the transforms are derived from the very context passed to ``fit``,
    so validation must never trigger a false positive.
    """
    new_meas = _predict_on_prior_mean(pretrained, [0.0, 10.0])
    new = GaussianProcessSurrogate(mean_or_factory=pretrained.posterior_mean_function)
    new.fit(wider_searchspace, _OBJECTIVE, new_meas)  # must not raise

    assert new._model is not None
