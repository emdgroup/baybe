"""Streamlit demo for Gaussian Process mean transfer."""

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
st.markdown(
    """
This demo shows how a GP's posterior mean can be transferred to serve as the prior 
mean for a new GP on a different search space.
"""
)

# Sidebar controls
st.sidebar.header("Configuration")

st.sidebar.subheader("Prior GP (narrow space)")
prior_n_points = st.sidebar.slider("Number of training points", 3, 20, 5, key="prior")
prior_x_min = st.sidebar.number_input("X min", -50.0, 0.0, 0.0, key="prior_xmin")
prior_x_max = st.sidebar.number_input("X max", 1.0, 50.0, 5.0, key="prior_xmax")

st.sidebar.subheader("New GP (wider space)")
new_n_points = st.sidebar.slider("Number of training points", 2, 20, 2, key="new")
new_x_min = st.sidebar.number_input("X min", -50.0, 0.0, 0.0, key="new_xmin")
new_x_max = st.sidebar.number_input("X max", 5.0, 50.0, 10.0, key="new_xmax")
new_y_scale = st.sidebar.slider(
    "Y scale factor",
    0.1,
    5.0,
    1.0,
    step=0.1,
    help="Multiply new GP training y-values by this factor",
)
use_prior_mean_data = st.sidebar.toggle(
    "Train on prior mean",
    value=False,
    help="When enabled, new GP training data lies exactly on the prior mean",
)

st.sidebar.subheader("New GP lengthscale")
constrain_lengthscale = st.sidebar.toggle(
    "Constrain lengthscale",
    value=False,
    help="Apply hard bounds on the new GP's kernel lengthscale",
)
ls_lower = st.sidebar.number_input(
    "Lengthscale lower bound", 0.01, 10.0, 0.1, step=0.01, key="ls_lower",
    disabled=not constrain_lengthscale,
)
ls_upper = st.sidebar.number_input(
    "Lengthscale upper bound", 0.01, 100.0, 2.0, step=0.01, key="ls_upper",
    disabled=not constrain_lengthscale,
)

st.sidebar.subheader("Data generation")
noise_level = st.sidebar.slider("Noise level", 0.0, 2.0, 0.3)
seed = st.sidebar.number_input("Random seed", 0, 1000, 42)

st.sidebar.subheader("Plot window")
plot_x_min = st.sidebar.number_input("X min", -50.0, 0.0, -10.0, key="plot_xmin")
plot_x_max = st.sidebar.number_input("X max", 1.0, 50.0, 15.0, key="plot_xmax")

st.sidebar.subheader("Display options")
show_std = st.sidebar.toggle(
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


def fit_new_gp(new_data, x_min, x_max, prior_gp, ls_bounds=None):
    """Fit the new GP with transferred mean."""
    # Create discrete values for parameter
    x_values = np.linspace(x_min, x_max, 100).tolist()
    params = [NumericalDiscreteParameter("x", values=x_values)]
    ss = SearchSpace.from_product(params)
    obj = NumericalTarget(name="y").to_objective()

    kernel_kwargs = {}
    if ls_bounds is not None:
        kernel_kwargs["kernel_or_factory"] = _make_constrained_kernel(*ls_bounds)

    # Option 1: GP with transferred mean
    gp_with_mean = GaussianProcessSurrogate(
        mean_or_factory=prior_gp.get_posterior_mean, **kernel_kwargs
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
    new_data, new_x_min, new_x_max, prior_gp, ls_bounds=ls_bounds
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
st.subheader("GP Mean Transfer Visualization")

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
    f"**trained on prior mean** (zero residuals) with y-scale={new_y_scale:.1f}"
    if use_prior_mean_data
    else f"**trained on random data** with noise and y-scale={new_y_scale:.1f}"
)

st.markdown(
    f"""
**Legend Guide:**
- 🔵 **Blue**: Prior GP trained on [{prior_x_min}, {prior_x_max}] with {prior_n_points} random points
- 🟢 **Green**: New GP with transferred mean, trained on [{new_x_min}, {new_x_max}] with {new_n_points} random points ({data_mode_text})
- 🟠 **Orange dotted**: Standard GP (no transfer) for comparison
- ⚫ **Gray dashed**: Prior GP's mean extended to the wider space

**Key Observations:**
1. The **blue Prior GP** learns from randomly positioned data in the narrow range
2. The **gray dashed line** shows how the prior GP's predictions extend to the wider space
3. The **green New GP** uses the prior mean as its baseline - notice how it stays close to the gray line where there's no training data!
4. The **orange Standard GP** has no prior knowledge and reverts to zero mean in regions without data
5. Training data at random x positions: 🔵 circles (prior) and 🟢 diamonds (new)

**Transfer Learning Benefit:**  
Where the new GP has no training data, the transferred mean provides informed predictions 
instead of reverting to a flat zero mean like the standard GP.

**Experiment:**
- Toggle "Train on prior mean" to see behavior when training data lies exactly on the transferred prior
- Adjust "Y scale factor" to see how the GP handles different data scales (tests outcome standardization)
"""
)

st.markdown("---")
