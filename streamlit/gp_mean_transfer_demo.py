"""Streamlit demo for Gaussian Process mean transfer."""

from functools import partial

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

from baybe.parameters.numerical import NumericalDiscreteParameter
from baybe.searchspace.core import SearchSpace
from baybe.surrogates.gaussian_process.core import GaussianProcessSurrogate
from baybe.targets.numerical import NumericalTarget

st.set_page_config(page_title="GP Mean Transfer Demo", layout="wide")

st.title("Gaussian Process Mean Transfer Demo")

prior_tab, new_tab, general_tab = st.sidebar.tabs(
    ["🔵 Prior GP", "🟢 New GP", "⚙️ General"]
)

with prior_tab:
    prior_n_points = st.slider("Training points", 3, 20, 5, key="prior_n")
    prior_x_range = st.slider(
        "X range", -50.0, 50.0, (0.0, 5.0), step=0.5, key="prior_x_range"
    )
    prior_x_min, prior_x_max = prior_x_range

with new_tab:
    new_n_points = st.slider("Training points", 2, 20, 2, key="new_n")
    new_x_range = st.slider(
        "X range", -50.0, 50.0, (0.0, 10.0), step=0.5, key="new_x_range"
    )
    new_x_min, new_x_max = new_x_range
    new_y_scale = st.slider(
        "Y scale factor",
        0.1,
        5.0,
        1.0,
        step=0.1,
        help="Multiply new GP training y-values by this factor",
    )
    use_prior_mean_data = st.toggle(
        "Train on prior mean",
        value=False,
        help="When enabled, new GP training data lies exactly on the prior mean",
    )

    st.markdown("**Mean transfer**")
    anchors = st.selectbox(
        "Anchors",
        options=["pretrained", "new", "combined"],
        index=0,
        help=(
            "Which inputs/targets to condition the inner (prior-mean) GP on:\n"
            "- pretrained: original training data\n"
            "- new: new GP's data (reuses only kernel structure)\n"
            "- combined: both"
        ),
    )
    mean_kernel_init = st.selectbox(
        "Inner kernel/mean init",
        options=["freeze", "warmstart", "discard"],
        index=0,
        help=(
            "How to initialize the inner kernel/likelihood/mean:\n"
            "- freeze: copy and freeze pretrained hyperparameters\n"
            "- warmstart: copy but leave trainable\n"
            "- discard: fresh components from default factories"
        ),
    )

    constrain_lengthscale = st.toggle(
        "Constrain lengthscale",
        value=False,
        help="Apply hard bounds on the new GP's kernel lengthscale",
    )
    if constrain_lengthscale:
        ls_c1, ls_c2 = st.columns(2)
        ls_lower = ls_c1.number_input(
            "LS lower", 0.01, 10.0, 0.1, step=0.01, key="ls_lower"
        )
        ls_upper = ls_c2.number_input(
            "LS upper", 0.01, 100.0, 2.0, step=0.01, key="ls_upper"
        )
    else:
        ls_lower, ls_upper = 0.1, 2.0

with general_tab:
    noise_level = st.slider("Noise level", 0.0, 2.0, 0.3)
    seed = st.number_input("Random seed", 0, 1000, 42)

    st.markdown("**Plot window**")
    plot_x_range = st.slider(
        "X range", -50.0, 50.0, (-10.0, 15.0), step=0.5, key="plot_x_range"
    )
    plot_x_min, plot_x_max = plot_x_range

    st.markdown("**Display**")
    show_std = st.toggle(
        "Show confidence intervals",
        value=True,
        help="Display 95% confidence intervals (±2 std)",
    )


# Generate synthetic data
@st.cache_data
def generate_data(x_min, x_max, n_points, noise, random_seed, y_scale=1.0):
    """Generate synthetic training data."""
    np.random.seed(random_seed)
    # Random x positions within the range
    x = np.random.uniform(x_min, x_max, n_points)
    x = np.sort(x)  # Sort for better visualization
    # Random y values (no correlation with x)
    y = np.random.randn(n_points) * noise * y_scale
    return pd.DataFrame({"x": x, "y": y})


def fit_prior_gp(prior_data, x_min, x_max):
    """Fit the prior GP."""
    # Create discrete values for parameter
    x_values = np.linspace(x_min, x_max, 50).tolist()
    params = [NumericalDiscreteParameter("x", values=x_values)]
    ss = SearchSpace.from_product(params)
    obj = NumericalTarget(name="y").to_objective()

    gp = GaussianProcessSurrogate()
    gp.fit(ss, obj, prior_data)
    return gp, ss


def _make_constrained_kernel(ls_lower, ls_upper):
    """Create a GPyTorch kernel with lengthscale constraints."""
    from gpytorch.constraints import Interval
    from gpytorch.kernels import MaternKernel, ScaleKernel

    return ScaleKernel(
        MaternKernel(
            nu=2.5,
            lengthscale_constraint=Interval(ls_lower, ls_upper),
        )
    )


def fit_new_gp(
    new_data,
    x_min,
    x_max,
    prior_gp,
    *,
    anchors="pretrained",
    mean_kernel_init="freeze",
    ls_bounds=None,
):
    """Fit the new GP with transferred mean."""
    # Create discrete values for parameter
    x_values = np.linspace(x_min, x_max, 100).tolist()
    params = [NumericalDiscreteParameter("x", values=x_values)]
    ss = SearchSpace.from_product(params)
    obj = NumericalTarget(name="y").to_objective()

    kernel_kwargs = {}
    if ls_bounds is not None:
        kernel_kwargs["kernel_or_factory"] = _make_constrained_kernel(*ls_bounds)

    mean_factory = partial(
        prior_gp.posterior_mean_function,
        anchors=anchors,
        mean_kernel_init=mean_kernel_init,
    )

    # Option 1: GP with transferred mean
    gp_with_mean = GaussianProcessSurrogate(
        mean_or_factory=mean_factory, **kernel_kwargs
    )
    gp_with_mean.fit(ss, obj, new_data)

    # Option 2: Standard GP (for comparison)
    gp_standard = GaussianProcessSurrogate(**kernel_kwargs)
    gp_standard.fit(ss, obj, new_data)

    return gp_with_mean, gp_standard, ss


# Generate data
prior_data = generate_data(prior_x_min, prior_x_max, prior_n_points, noise_level, seed)

# Fit prior GP first
prior_gp, prior_ss = fit_prior_gp(prior_data, prior_x_min, prior_x_max)

# Generate new GP data based on toggle
if use_prior_mean_data:
    # Generate random x values and get predictions from prior GP
    np.random.seed(seed + 100)
    x_new = np.random.uniform(new_x_min, new_x_max, new_n_points)
    x_new = np.sort(x_new)
    new_data_df = pd.DataFrame({"x": x_new})
    with torch.no_grad():
        prior_predictions = prior_gp.posterior(new_data_df).mean.numpy().ravel()
    new_data = pd.DataFrame({"x": x_new, "y": prior_predictions * new_y_scale})
else:
    new_data = generate_data(
        new_x_min, new_x_max, new_n_points, noise_level, seed + 100, new_y_scale
    )

# Fit new GPs
ls_bounds = (ls_lower, ls_upper) if constrain_lengthscale else None
new_gp_with_mean, new_gp_standard, new_ss = fit_new_gp(
    new_data,
    new_x_min,
    new_x_max,
    prior_gp,
    anchors=anchors,
    mean_kernel_init=mean_kernel_init,
    ls_bounds=ls_bounds,
)

# Generate test points for plotting (using plot window range)
x_test = np.linspace(plot_x_min, plot_x_max, 300)
test_df = pd.DataFrame({"x": x_test})

# Get predictions for all GPs over the same plot range
with torch.no_grad():
    prior_posterior = prior_gp.posterior(test_df)
    prior_mean = prior_posterior.mean.numpy().ravel()
    prior_std = prior_posterior.variance.sqrt().numpy().ravel()

    new_with_mean_posterior = new_gp_with_mean.posterior(test_df)
    new_with_mean_mean = new_with_mean_posterior.mean.numpy().ravel()
    new_with_mean_std = new_with_mean_posterior.variance.sqrt().numpy().ravel()

    new_standard_posterior = new_gp_standard.posterior(test_df)
    new_standard_mean = new_standard_posterior.mean.numpy().ravel()
    new_standard_std = new_standard_posterior.variance.sqrt().numpy().ravel()

# Create combined plot
fig = go.Figure()

# Prior GP - confidence interval (conditional)
if show_std:
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_test, x_test[::-1]]),
            y=np.concatenate(
                [
                    prior_mean + 2 * prior_std,
                    (prior_mean - 2 * prior_std)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(0, 100, 250, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="Prior GP: 95% CI",
            showlegend=True,
            legendgroup="prior",
        )
    )

# Prior GP - mean prediction
fig.add_trace(
    go.Scatter(
        x=x_test,
        y=prior_mean,
        mode="lines",
        name="Prior GP Mean",
        line=dict(color="blue", width=3),
        legendgroup="prior",
    )
)

# Prior GP - training data
fig.add_trace(
    go.Scatter(
        x=prior_data["x"],
        y=prior_data["y"],
        mode="markers",
        name="Prior GP Training Data",
        marker=dict(color="darkblue", size=10, symbol="circle"),
        legendgroup="prior",
    )
)

# New GP with transferred mean - confidence interval (conditional)
if show_std:
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_test, x_test[::-1]]),
            y=np.concatenate(
                [
                    new_with_mean_mean + 2 * new_with_mean_std,
                    (new_with_mean_mean - 2 * new_with_mean_std)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(0, 200, 100, 0.2)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="New GP w/ Mean: 95% CI",
            showlegend=True,
            legendgroup="new_with_mean",
        )
    )

# New GP with transferred mean
fig.add_trace(
    go.Scatter(
        x=x_test,
        y=new_with_mean_mean,
        mode="lines",
        name="New GP w/ Transferred Mean",
        line=dict(color="green", width=3),
        legendgroup="new_with_mean",
    )
)

# Standard GP - confidence interval (conditional)
if show_std:
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_test, x_test[::-1]]),
            y=np.concatenate(
                [
                    new_standard_mean + 2 * new_standard_std,
                    (new_standard_mean - 2 * new_standard_std)[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(255, 165, 0, 0.15)",
            line=dict(color="rgba(255, 255, 255, 0)"),
            name="Standard GP: 95% CI",
            showlegend=True,
            legendgroup="standard",
        )
    )

# Standard GP (for comparison)
fig.add_trace(
    go.Scatter(
        x=x_test,
        y=new_standard_mean,
        mode="lines",
        name="Standard GP (no transfer)",
        line=dict(color="orange", width=3, dash="dot"),
        legendgroup="standard",
    )
)

# New GP training data
fig.add_trace(
    go.Scatter(
        x=new_data["x"],
        y=new_data["y"],
        mode="markers",
        name="New GP Training Data",
        marker=dict(color="darkgreen", size=10, symbol="diamond"),
        legendgroup="new_with_mean",
    )
)

fig.update_layout(
    xaxis_title="x",
    yaxis_title="y",
    height=600,
    legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
    hovermode="x unified",
)
st.plotly_chart(fig, width="stretch")

# Explanation
st.markdown("---")
st.subheader("What's happening?")

data_mode_text = (
    f"trained on the prior mean (zero residuals) with y-scale={new_y_scale:.1f}"
    if use_prior_mean_data
    else f"trained on random data with noise and y-scale={new_y_scale:.1f}"
)

st.markdown(
    f"""
Three GPs are plotted on the same axis:

- **🔵 Prior GP** — fitted on `{prior_n_points}` points in `[{prior_x_min}, {prior_x_max}]`.
- **🟢 New GP w/ Transferred Mean** — fitted on `{new_n_points}` points in `[{new_x_min}, {new_x_max}]` ({data_mode_text}), using the prior GP's posterior as its prior mean function.
- **🟠 Standard GP (dotted)** — same data as the green GP but with a zero mean. Baseline for comparison.

Where the new GPs have **no training data**, the green curve follows the blue prior (informed extrapolation) while the orange curve collapses back to zero with wide uncertainty. That gap is what mean transfer buys you.

**Mean transfer controls** (sidebar → 🟢 New GP):

- **Anchors** — which inputs/targets condition the inner GP that produces the prior mean.
  - `pretrained`: use the pretrained GP's training data.
  - `new`: use the new GP's data — only the kernel structure is reused.
  - `combined`: condition on both.
- **Inner kernel/mean init** — how to initialize the inner kernel/likelihood/mean.
  - `freeze`: copy pretrained components and freeze them (prior mean is fixed).
  - `warmstart`: copy them but leave trainable so the outer MLL can adjust.
  - `discard`: rebuild from default factories (no hyperparameter transfer).

**Things to try:**

- Toggle **Train on prior mean** — when training data lies exactly on the blue curve, the green GP should hug it everywhere.
- Increase **Y scale factor** — checks that outcome standardization handles different scales.
- Set **Anchors = combined** with **freeze** — richer conditioning typically tightens the green CI.
- Set **Anchors = new** with **warmstart** — rebases the prior shape onto the new data.
"""
)

st.markdown("---")