"""This script allows comparing selection recommenders on different data sets."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import make_blobs

from baybe.recommenders.pure.nonpredictive.base import NonPredictiveRecommender
from baybe.searchspace import SearchSpace, SubspaceDiscrete
from baybe.utils.basic import get_subclasses


def uniform_distribution(n_points: int) -> np.ndarray:
    """Uniformly distributed point generation."""
    return np.random.rand(n_points, 2)


def gaussian_mixture_model(
    n_points: int,
    n_mixtures: int = 3,
    mixture_std: float = 1.0,
) -> np.ndarray:
    """Point generation via Gaussian mixture model."""
    points, *_ = make_blobs(
        n_samples=n_points,
        centers=n_mixtures,
        cluster_std=mixture_std,
    )
    return points


def plot_point_selection(points, selection, title):
    """Visualizes a point selection and the corresponding selection order."""
    fig = go.Figure(data=go.Scatter(x=points[:, 0], y=points[:, 1], mode="markers"))
    for step, sel in enumerate(selection):
        fig.add_annotation(
            x=points[sel, 0],
            y=points[sel, 1],
            text=str(step),
            arrowcolor="#8b0000",
        )
    fig.update_layout(
        title={
            "text": title,
            "y": 0.85,
        },
        template="seaborn",
    )
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


# collect all available data distributions
data_distributions = {
    "Uniform": uniform_distribution,
    "Gaussian Mixture Model": gaussian_mixture_model,
}

# collect all available recommenders
selection_recommenders = {
    cls.__name__: cls for cls in get_subclasses(NonPredictiveRecommender)
}


def main():
    """Create the streamlit dashboard."""
    # show docstring in dashboard
    st.info(__doc__)

    # simulation parameters
    random_seed = int(st.sidebar.number_input("Random seed", value=42))
    recommender_name = st.sidebar.selectbox(
        "Recommender", list(selection_recommenders.keys())
    )
    n_points = st.sidebar.slider("Number of points to be generated", 10, 100, value=50)
    n_selected = st.sidebar.slider(
        "Number of points to be selected",
        1,
        n_points,
        value=int(np.ceil(n_points / 10)),
    )
    distribution = st.sidebar.selectbox(
        "Data distribution", list(data_distributions.keys())
    )

    # set the distribution options
    if distribution == "Gaussian Mixture Model":
        distribution_params = {
            "n_mixtures": st.sidebar.number_input("Number of mixtures", 1, value=2),
            "mixture_std": st.sidebar.slider("Mixture size", 0.0, 1.0, value=1.0),
        }
    else:
        distribution_params = {}

    # fix the random seed
    np.random.seed(random_seed)

    # create the points
    points = pd.DataFrame(
        data_distributions[distribution](n_points, **distribution_params),
        columns=["x", "y"],
    )

    # create the corresponding search space
    searchspace = SearchSpace(discrete=SubspaceDiscrete.from_dataframe(points))

    # create the recommender and generate the recommendations
    # TODO: The acquisition function should become optional for model-free methods
    recommender = selection_recommenders[recommender_name]()
    selection = recommender.recommend(searchspace=searchspace, batch_size=n_selected)

    # show the result
    fig = plot_point_selection(points.values, selection.index.values, recommender_name)
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
