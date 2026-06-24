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

# Fixed Forrester coefficients for the prior GP.
PRIOR_FORRESTER = (1.0, 0.0, 0.0)

# The Forrester toggle lives in the General tab (rendered last), but the Prior/New
# GP tabs must react to it. Streamlit restores keyed-widget state at the start of
# each run, so reading the stored value here reflects the latest toggle state.
forrester_on = st.session_state.get("forrester_on", False)

prior_tab, new_tab, general_tab = st.sidebar.tabs(
    ["🔵 Prior GP", "🟢 New GP", "⚙️ General"]
)

with prior_tab:
    prior_n_points = st.slider("Training points", 3, 20, 5, key="prior_n")
    if forrester_on:
        prior_x_min, prior_x_max = 0.0, 1.0
        st.info("Forrester mode: X range fixed to [0, 1].")
    else:
        prior_x_range = st.slider(
            "X range", -50.0, 50.0, (0.0, 5.0), step=0.5, key="prior_x_range"
        )
        prior_x_min, prior_x_max = prior_x_range

with new_tab:
    new_n_points = st.slider("Training points", 2, 20, 2, key="new_n")
    if forrester_on:
        new_x_range = st.slider(
            "X range (within [0, 1])",
            0.0,
            1.0,
            (0.0, 0.5),
            step=0.05,
            key="new_x_range_forrester",
        )
        new_x_min, new_x_max = new_x_range
        new_y_scale = 1.0
        use_prior_mean_data = False
        st.slider(
            "Y scale factor",
            0.1,
            5.0,
            1.0,
            step=0.1,
            disabled=True,
            help="Disabled in Forrester mode (scale is determined by A, B, C).",
            key="new_y_scale_disabled",
        )
        st.toggle(
            "Train on prior mean",
            value=False,
            disabled=True,
            help="Disabled in Forrester mode.",
            key="use_prior_mean_disabled",
        )
    else:
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

    st.markdown("**Forrester set up**")
    forrester_on = st.toggle(
        "Use Forrester functions",
        value=False,
        key="forrester_on",
        help=(
            "Sample training data from the generalized Forrester function "
            "f(x) = A·(6x−2)²·sin(12x−4) + B·(x−0.5) + C instead of Gaussian noise."
        ),
    )
    if forrester_on:
        pa, pb, pc = PRIOR_FORRESTER
        st.caption(f"Prior GP (fixed): A={pa:g}, B={pb:g}, C={pc:g}")
        fc1, fc2, fc3 = st.columns(3)
        new_A = fc1.number_input("A (new)", value=0.5, step=0.1, key="forr_A")
        new_B = fc2.number_input("B (new)", value=10.0, step=0.5, key="forr_B")
        new_C = fc3.number_input("C (new)", value=-5.0, step=0.5, key="forr_C")
    else:
        new_A, new_B, new_C = 0.5, 10.0, -5.0

    st.markdown("**Plot window**")
    if forrester_on:
        plot_x_min, plot_x_max = 0.0, 1.0
        st.info("Forrester mode: plot window set to [0, 1].")
    else:
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


def forrester(x, A, B, C):
    """Evaluate the generalized Forrester function."""
    x = np.asarray(x, dtype=float)
    return A * (6.0 * x - 2.0) ** 2 * np.sin(12.0 * x - 4.0) + B * (x - 0.5) + C


@st.cache_data
def generate_forrester_data(x_min, x_max, n_points, A, B, C, noise, random_seed):
    """Generate training data sampled from the Forrester function."""
    np.random.seed(random_seed)
    x = np.sort(np.random.uniform(x_min, x_max, n_points))
    y = forrester(x, A, B, C) + np.random.randn(n_points) * noise
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
prior_sig = (
    prior_n_points,
    prior_x_min,
    prior_x_max,
    noise_level,
    seed,
    forrester_on,
)

if st.session_state.get("prior_sig") != prior_sig:
    if forrester_on:
        st.session_state.prior_data = generate_forrester_data(
            prior_x_min,
            prior_x_max,
            prior_n_points,
            *PRIOR_FORRESTER,
            noise_level,
            seed,
        )
    else:
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
    forrester_on,
    new_A,
    new_B,
    new_C,
)

if st.session_state.get("new_sig") != new_sig:
    if forrester_on:
        st.session_state.new_data = generate_forrester_data(
            new_x_min,
            new_x_max,
            new_n_points,
            new_A,
            new_B,
            new_C,
            noise_level,
            seed + 100,
        )
    elif use_prior_mean_data:
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

    # Transferred prior-mean function actually used by the green GP (raw space).
    # `mean_module.gp` is the inner GP; its posterior mean -- including any updates
    # the outer MLL applied in warmstart/discard -- is exactly the prior mean the
    # green outer GP is built on. Plotting it makes the green curve's far-away
    # reversion self-explanatory and reveals how the anchors/init modes reshape it.
    mean_module = new_gp_with_mean._model.mean_module
    inner_gp = getattr(mean_module, "gp", None)
    if inner_gp is not None:
        x_raw = torch.tensor(
            x_test, dtype=inner_gp.train_inputs[0].dtype
        ).unsqueeze(-1)
        transferred_mean = inner_gp.posterior(x_raw).mean.numpy().ravel()
    else:
        transferred_mean = None

# --- Bokeh figure with draggable training points ---
p = figure(
    height=600,
    sizing_mode="stretch_width",
    x_axis_label="x",
    y_axis_label="y",
    x_range=(plot_x_min, plot_x_max),
    tools="pan,box_zoom,wheel_zoom,reset,save",
    toolbar_location="above",
)

# Ground-truth Forrester functions (drawn first so GP means stay on top).
if forrester_on:
    prior_truth = forrester(x_test, *PRIOR_FORRESTER)
    new_truth = forrester(x_test, new_A, new_B, new_C)
    p.line(
        x_test,
        prior_truth,
        line_color="#1f3a8a",
        line_width=2,
        line_dash="dashed",
        line_alpha=0.9,
        legend_label="Prior Forrester (truth)",
    )
    p.line(
        x_test,
        new_truth,
        line_color="#0a7d3c",
        line_width=2,
        line_dash="dotdash",
        line_alpha=0.9,
        legend_label="New Forrester (truth)",
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

if transferred_mean is not None:
    p.line(
        x_test,
        transferred_mean,
        line_color="#9C27B0",
        line_width=2,
        line_dash="dashed",
        legend_label="Transferred Prior Mean (inner GP)",
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

if forrester_on:
    data_mode_text = (
        f"sampled from the Forrester function with "
        f"A={new_A:g}, B={new_B:g}, C={new_C:g}"
    )
elif use_prior_mean_data:
    data_mode_text = (
        f"trained on the prior mean (zero residuals) with y-scale={new_y_scale:.1f}"
    )
else:
    data_mode_text = f"trained on random data with noise and y-scale={new_y_scale:.1f}"

st.markdown(
    f"""
Curves plotted on the same axis:

- **🔵 Prior GP** — fitted on `{prior_n_points}` points in `[{prior_x_min}, {prior_x_max}]`.
- **🟢 New GP w/ Transferred Mean** — fitted on `{new_n_points}` points in `[{new_x_min}, {new_x_max}]` ({data_mode_text}), using the prior GP's posterior as its prior mean function.
- **🟣 Transferred Prior Mean (dashed)** — the actual inner-GP posterior mean used as the green GP's prior mean. It depends on **Anchors** and **Inner kernel/mean init**, and the green curve reverts to it where there is no new data. (Equals the blue prior only for `anchors=pretrained` + `freeze`.)
- **🟠 Standard GP (dotted)** — same data as the green GP but with a zero mean. Baseline for comparison.

Where the new GPs have **no training data**, the green curve follows the **purple transferred mean** (informed extrapolation) while the orange curve collapses back to zero with wide uncertainty. That gap is what mean transfer buys you.

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

if forrester_on:
    pa, pb, pc = PRIOR_FORRESTER
    st.markdown(
        f"""
**Forrester mode** is active. Training targets are sampled from the generalized
Forrester function $f(x) = A\\,(6x-2)^2\\sin(12x-4) + B\\,(x-0.5) + C$:

- **Prior GP** uses the fixed low-fidelity setting `A={pa:g}, B={pb:g}, C={pc:g}`
  (dashed navy *truth* curve).
- **New GP** uses your `A={new_A:g}, B={new_B:g}, C={new_C:g}`
  (dash-dot green *truth* curve).

The prior X range is fixed to `[0, 1]`; the new GP samples a sub-range of `[0, 1]`
so the region without new data shows how the transferred mean extrapolates along
the prior. Y-scale and *train on prior mean* are disabled because the targets are
fully determined by `A, B, C`.
"""
    )

st.markdown("---")
