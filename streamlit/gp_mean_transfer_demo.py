"""Streamlit demo for Gaussian Process mean transfer."""

from functools import partial

import numpy as np
import pandas as pd
import streamlit as st
import torch
from bokeh.models import ColumnDataSource, CustomJS, PointDrawTool
from bokeh.plotting import figure
from streamlit_bokeh3_events import streamlit_bokeh3_events as streamlit_bokeh_events

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


# --- Session-state data with signature-based invalidation ---
# Any change to a data-generating sidebar input wipes user drag-edits.
prior_sig = (prior_n_points, prior_x_min, prior_x_max, noise_level, seed)

if st.session_state.get("prior_sig") != prior_sig:
    st.session_state.prior_data = generate_data(
        prior_x_min, prior_x_max, prior_n_points, noise_level, seed
    )
    st.session_state.prior_sig = prior_sig
    st.session_state.last_prior_drag = 0

prior_data = st.session_state.prior_data
prior_gp, prior_ss = fit_prior_gp(prior_data, prior_x_min, prior_x_max)

new_sig = (
    new_n_points,
    new_x_min,
    new_x_max,
    new_y_scale,
    noise_level,
    seed,
    use_prior_mean_data,
    prior_sig if use_prior_mean_data else None,
)

if st.session_state.get("new_sig") != new_sig:
    if use_prior_mean_data:
        np.random.seed(seed + 100)
        x_new = np.sort(np.random.uniform(new_x_min, new_x_max, new_n_points))
        with torch.no_grad():
            y_pred = prior_gp.posterior(pd.DataFrame({"x": x_new})).mean.numpy().ravel()
        st.session_state.new_data = pd.DataFrame(
            {"x": x_new, "y": y_pred * new_y_scale}
        )
    else:
        st.session_state.new_data = generate_data(
            new_x_min, new_x_max, new_n_points, noise_level, seed + 100, new_y_scale
        )
    st.session_state.new_sig = new_sig
    st.session_state.last_new_drag = 0

new_data = st.session_state.new_data

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

# --- Bokeh figure with draggable training points ---
p = figure(
    width=1200,
    height=600,
    x_axis_label="x",
    y_axis_label="y",
    x_range=(plot_x_min, plot_x_max),
    tools="pan,box_zoom,wheel_zoom,reset,save",
    toolbar_location="above",
)

if show_std:
    p.varea(
        x=x_test,
        y1=prior_mean - 2 * prior_std,
        y2=prior_mean + 2 * prior_std,
        fill_color="#6496FA",
        fill_alpha=0.2,
        legend_label="Prior GP: 95% CI",
    )
p.line(
    x_test,
    prior_mean,
    line_color="blue",
    line_width=3,
    legend_label="Prior GP Mean",
)

if show_std:
    p.varea(
        x=x_test,
        y1=new_with_mean_mean - 2 * new_with_mean_std,
        y2=new_with_mean_mean + 2 * new_with_mean_std,
        fill_color="#00C864",
        fill_alpha=0.2,
        legend_label="New GP w/ Mean: 95% CI",
    )
p.line(
    x_test,
    new_with_mean_mean,
    line_color="green",
    line_width=3,
    legend_label="New GP w/ Transferred Mean",
)

if show_std:
    p.varea(
        x=x_test,
        y1=new_standard_mean - 2 * new_standard_std,
        y2=new_standard_mean + 2 * new_standard_std,
        fill_color="#FFA500",
        fill_alpha=0.15,
        legend_label="Standard GP: 95% CI",
    )
p.line(
    x_test,
    new_standard_mean,
    line_color="orange",
    line_width=3,
    line_dash="dotted",
    legend_label="Standard GP (no transfer)",
)

prior_source = ColumnDataSource(
    data=dict(x=prior_data["x"].tolist(), y=prior_data["y"].tolist())
)
new_source = ColumnDataSource(
    data=dict(x=new_data["x"].tolist(), y=new_data["y"].tolist())
)

prior_renderer = p.scatter(
    "x",
    "y",
    source=prior_source,
    color="darkblue",
    size=12,
    marker="circle",
    legend_label="Prior GP Training Data (draggable)",
)
new_renderer = p.scatter(
    "x",
    "y",
    source=new_source,
    color="darkgreen",
    size=14,
    marker="diamond",
    legend_label="New GP Training Data (draggable)",
)

draw_tool = PointDrawTool(
    renderers=[prior_renderer, new_renderer],
    add=False,
)
p.add_tools(draw_tool)
p.toolbar.active_tap = draw_tool

p.legend.location = "top_left"
p.legend.click_policy = "hide"

_dispatch_js = """
    window.{counter} = (window.{counter} || 0) + 1;
    const payload = {{x: Array.from(src.data.x), y: Array.from(src.data.y), n: window.{counter}}};
    const ev1 = new CustomEvent("{name}", {{detail: payload}});
    const ev2 = new CustomEvent("{name}", {{detail: payload}});
    document.dispatchEvent(ev1);
    window.dispatchEvent(ev2);
    console.log("[drag] {name}", payload);
"""

prior_source.js_on_change(
    "data",
    CustomJS(
        args=dict(src=prior_source),
        code=_dispatch_js.format(counter="_priorN", name="PRIOR_MOVED"),
    ),
)
new_source.js_on_change(
    "data",
    CustomJS(
        args=dict(src=new_source),
        code=_dispatch_js.format(counter="_newN", name="NEW_MOVED"),
    ),
)

_plot_key = hash(
    (
        prior_sig,
        new_sig,
        tuple(prior_data["x"]),
        tuple(prior_data["y"]),
        tuple(new_data["x"]),
        tuple(new_data["y"]),
    )
)

event_result = streamlit_bokeh_events(
    bokeh_plot=p,
    events="PRIOR_MOVED,NEW_MOVED",
    key=f"plot_{_plot_key}",
    refresh_on_update=True,
    debounce_time=200,
    override_height=620,
)

st.write("DEBUG event_result:", event_result)

if event_result:
    rerun = False
    pm = event_result.get("PRIOR_MOVED")
    if pm and pm.get("n", 0) > st.session_state.get("last_prior_drag", 0):
        st.session_state.last_prior_drag = pm["n"]
        st.session_state.prior_data = pd.DataFrame({"x": pm["x"], "y": pm["y"]})
        rerun = True
    nm = event_result.get("NEW_MOVED")
    if nm and nm.get("n", 0) > st.session_state.get("last_new_drag", 0):
        st.session_state.last_new_drag = nm["n"]
        st.session_state.new_data = pd.DataFrame({"x": nm["x"], "y": nm["y"]})
        rerun = True
    if rerun:
        st.rerun()

st.caption(
    "💡 Drag the 🔵 / 🟢 training points to move them. "
    "Any sidebar change that regenerates data will overwrite your edits."
)

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

- **Drag** a 🔵 or 🟢 point — both GPs refit live.
- Toggle **Train on prior mean** — when training data lies exactly on the blue curve, the green GP should hug it everywhere.
- Set **Anchors = combined** with **freeze** — richer conditioning typically tightens the green CI.
- Set **Anchors = new** with **warmstart** — rebases the prior shape onto the new data.
"""
)

st.markdown("---")
