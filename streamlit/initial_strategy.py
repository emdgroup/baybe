"""
The purpose of this script is to compare different surrogate models and verify the
performance between different initial sampling algorithms.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from baybe.utils.sampling_algorithms import farthest_point_sampling
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture


st.set_page_config(page_title="Initial Strategy", page_icon="🚀", layout="centered")
# show docstring in dashboard
st.info(__doc__)


def random_strategy(points: pd.DataFrame, batch_quantity: int = 1):
    """Random strategy"""
    return pd.Index(np.random.choice(points.index, batch_quantity, replace=False))


def uniform_distribution(n_points: int) -> np.ndarray:
    """Uniformly distributed point generation"""
    return np.random.rand(n_points, 2)


def gaussian_mixture_model(n_points: int) -> np.ndarray:
    """GMM point generation"""
    x_points, _ = make_blobs(  # pylint: disable=unbalanced-tuple-unpacking
        n_samples=n_points
    )
    gmm = GaussianMixture(n_components=3)
    gmm.fit(x_points)
    features, _ = gmm.sample(n_points)
    return features


def plot_plotly(results, title, features):
    """Visualize point selection and its selection order"""
    fig = go.Figure(data=go.Scatter(x=features[:, 0], y=features[:, 1], mode="markers"))
    for step, _ in enumerate(results):
        fig.add_annotation(
            x=features[:, 0][results[step]],
            y=features[:, 1][results[step]],
            text=str(step),
            arrowcolor="#8b0000",
            font=dict(color="#8b0000"),
        )

    fig.update_layout(
        title={
            "text": title,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        template="seaborn",
    )
    return fig


# define all available test functions
test_datas = {
    "Uniform Distribution": uniform_distribution,
    "Gaussian Mixture Model": gaussian_mixture_model,
}

test_strategies = {
    "Random": random_strategy,
    "FPS": farthest_point_sampling,
}


# simulation parameters
random_seed = int(st.sidebar.number_input("Random seed", value=42))
dataset = st.sidebar.selectbox("Data", list(test_datas.keys()))
strategy = st.sidebar.selectbox("Strategy", list(test_strategies.keys()))
data_points = st.sidebar.slider("Number of points to be generated", 10, 1000, value=50)
sampled_points = st.sidebar.slider("Number of points to be sampled", 2, 100, value=10)


feats = test_datas[dataset](data_points)
res = test_strategies[strategy](pd.DataFrame(feats), sampled_points)
st.plotly_chart(plot_plotly(res, strategy, feats), use_container_width=True)
