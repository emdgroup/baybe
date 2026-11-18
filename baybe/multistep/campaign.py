"""Multi-step campaign facade for DAG-structured experiments.

The facade owns the multi-step measurements and fits a single
:class:`~baybe.multistep.model.MultiStepModel` that is composed of one model per
DAG node. For p-KGFN it delegates action selection to
:class:`~baybe.multistep.acquisition.PartialKnowledgeGradientFN`; for stock
acquisition functions it hands the model to a
:class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`
directly. A recommendation sees the DAG-wide search space and the final-node
objective; upstream-only measurements still train their node models.

The campaign tracks one experiment at a time via a generated identifier.
Recommendations return the identifier and continue the current experiment.
Measurements pass the identifier back, and once every node has been measured the
campaign assembles a complete DAG row.
"""

from __future__ import annotations

import gc
from functools import cached_property
from typing import Any, cast

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional
from attrs.validators import instance_of

from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import IncompatibilityError
from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender

from ._columns import EXPERIMENT_ID, NODE
from .acquisition import PartialKnowledgeGradientFN
from .dag import MultiStepDAG
from .data import MultiStepData, NodeMeasurement
from .model import MultiStepModel


@define(slots=False)
class MultiStepCampaign:
    """Stateful facade for multi-step Bayesian optimization.

    Node surrogates that produce ensemble posteriors, such as
    :class:`~baybe.surrogates.random_forest.RandomForestSurrogate`, can drive
    recommendations over discrete search spaces. Because ensemble posteriors are
    not differentiable, continuous search spaces require gradient-capable node
    surrogates.

    The facade fits a single multi-step model composed of one model per node.
    The p-KGFN policy selects actions directly; stock acquisition functions use a
    :class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`
    directly. The acquisition function is configurable; other recommender
    settings are not exposed.
    """

    dag: MultiStepDAG = field(validator=instance_of(MultiStepDAG))
    """The DAG topology."""

    data: MultiStepData = field(factory=MultiStepData.empty)
    """The measurement store holding node measurements and complete rows."""

    acquisition_function: AcquisitionFunction | None = field(
        default=None, converter=optional(convert_acqf)
    )
    """The p-KGFN policy or stock acquisition function; ``None`` selects the default."""

    _cached_recommendation: tuple[int, pd.DataFrame] | None = field(
        default=None, init=False, eq=False
    )
    """The cached recommendation and the batch size it was computed for."""

    _current_experiment_id: str | None = field(default=None, init=False, eq=False)
    """The experiment that is currently being built, if any."""

    _current_experiment: dict[str, NodeMeasurement] = field(
        factory=dict, init=False, eq=False
    )
    """The measurements of the current experiment, keyed by node name."""

    _experiment_counter: int = field(default=0, init=False, eq=False)
    """Monotonic counter for generating experiment identifiers."""

    @data.validator
    def _validate_data(  # noqa: DOC101, DOC103
        self, _: Any, value: MultiStepData
    ) -> None:
        """Validate the measurement store against the DAG."""
        value.validate(self.dag)

    @classmethod
    def from_dag(
        cls,
        dag: MultiStepDAG,
        *,
        acquisition_function: AcquisitionFunction | str | None = None,
        data: MultiStepData | None = None,
    ) -> MultiStepCampaign:
        """Create a campaign for the given DAG.

        Args:
            dag: The DAG topology, including the optimization objective.
            acquisition_function: Optional acquisition function or its name.
            data: Optional initial measurement store.

        Returns:
            The configured campaign.
        """
        return cls(
            dag=dag,
            data=MultiStepData.empty() if data is None else data,
            acquisition_function=acquisition_function,
        )

    @cached_property
    def model(self) -> MultiStepModel:
        """The multi-step model owned by the campaign."""
        return self.dag.make_model()

    def add_measurements(
        self,
        node: str,
        y: pd.DataFrame,
        x: pd.DataFrame | None = None,
        experiment_id: str | None = None,
    ) -> tuple[NodeMeasurement, ...]:
        """Record observed node outputs.

        Measurements are stored as individual node observations. When
        ``experiment_id`` matches the current experiment and the measurement
        completes it, a complete DAG row is assembled automatically.

        Args:
            node: The observed node.
            y: The observed node outputs, with exactly one column named after
                the node output.
            x: The full node-model inputs, with one column per own parameter and
                per parent output, and the same number of rows as ``y``.
            experiment_id: An optional experiment identifier. When omitted, the
                current experiment identifier is used.

        Returns:
            The recorded measurements.

        Raises:
            IncompatibilityError: If the node, the columns, the values, or the
                experiment identifier are invalid.
        """
        node_object = self.dag.node_by_name.get(node)
        if node_object is None:
            raise IncompatibilityError(
                f"Node '{node}' is not part of the DAG. Known nodes: "
                f"{sorted(self.dag.node_by_name)}."
            )

        if len(y.columns) != 1 or y.columns[0] != node_object.output:
            raise IncompatibilityError(
                f"Output data for node '{node}' must use the column name "
                f"'{node_object.output}', got {list(y.columns)}."
            )

        if x is None:
            raise IncompatibilityError(
                f"The model input 'x' is required for node '{node}'. "
                f"Expected columns: {list(node_object.model_input_names)}."
            )
        if not x.columns.is_unique:
            raise IncompatibilityError(
                f"Input columns for node '{node}' must be unique."
            )

        expected = node_object.model_input_names
        missing = sorted(set(expected) - set(x.columns))
        unexpected = sorted(set(x.columns) - set(expected))
        if missing or unexpected:
            raise IncompatibilityError(
                f"Input data for node '{node}' must contain exactly the node's "
                f"model input. Missing columns: {missing}, unexpected columns: "
                f"{unexpected}. Expected columns: {list(expected)}."
            )

        if len(x) != len(y):
            raise IncompatibilityError(
                f"Input and output data for node '{node}' must have the same "
                f"number of rows, got {len(x)} and {len(y)}."
            )

        if experiment_id is None:
            experiment_id = self._current_experiment_id

        measurements = tuple(
            NodeMeasurement(
                node=node,
                inputs=cast(dict[str, Any], inputs),
                output=output,
                experiment_id=experiment_id,
            )
            for inputs, output in zip(
                x.to_dict("records"), y[node_object.output].tolist(), strict=True
            )
        )

        self.data = self.data.add(*measurements)
        self._cached_recommendation = None

        # Update the current experiment if the measurement belongs to it.
        if experiment_id is not None and experiment_id == self._current_experiment_id:
            self._current_experiment[node] = measurements[-1]
            if self.data.is_experiment_complete(self.dag, experiment_id):
                row = self.data.assemble_complete_row(self.dag, experiment_id)
                self.data = self.data.add_complete(row)
                self._current_experiment_id = None
                self._current_experiment = {}

        return measurements

    def add_complete_measurements(self, measurements: pd.DataFrame) -> None:
        """Load complete DAG rows.

        The rows are used by the acquisition function for its reference
        computations. Node models are additionally trained on the per-node views
        of the same rows.

        Args:
            measurements: Complete DAG rows in experimental representation with
                one column per DAG parameter and one column per node output.
                Additional columns are ignored.

        Raises:
            IncompatibilityError: If required columns or values are invalid.
        """
        required = (
            *self.dag.parameter_by_name,
            *(node.output for node in self.dag.nodes),
        )
        missing = [column for column in required if column not in measurements.columns]
        if missing:
            raise IncompatibilityError(
                f"Complete measurements are missing required columns: {missing}."
            )
        self.data = self.data.add_complete(measurements[list(required)])
        self._cached_recommendation = None

    def recommend(self, batch_size: int = 1) -> pd.DataFrame:
        """Recommend the next node action.

        With :class:`~baybe.multistep.acquisition.PartialKnowledgeGradientFN`,
        select one local measurement action by its expected terminal utility
        gain per node cost. The policy supports discrete, continuous, and hybrid
        node-owned inputs, and returns the complete local input including real
        observed parent outputs. It does not require a complete DAG row.

        With other acquisition functions, score configurations in the complete
        DAG search space and select the next unmeasured node whose parents have
        been measured in the current experiment. This legacy route requires a
        discrete DAG search space.

        A recommendation continues the current experiment when one is in
        progress; otherwise it starts a fresh one. Each recommendation returns an
        ``experiment_id`` that can be passed back to
        :meth:`~add_measurements`.

        A random initial design is used while node models cannot be fitted.

        Args:
            batch_size: The number of node actions to recommend.

        Returns:
            A dataframe holding ``experiment_id``, ``node``, and the complete
            local input of the recommended node (its own parameters and parent
            outputs).

        Raises:
            ValueError: If ``batch_size`` is smaller than 1.
            IncompatibilityError: If the legacy route receives a non-discrete DAG
                search space, or if the selected policy does not support the DAG,
                objective, model, or requested batch size.
        """
        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but "
                f"provided {batch_size=}."
            )

        pkgfn_config = (
            self.acquisition_function
            if isinstance(self.acquisition_function, PartialKnowledgeGradientFN)
            else None
        )
        if pkgfn_config is not None:
            pkgfn_config.validate_context(self.dag, self.data, batch_size)

        cached = self._cached_recommendation
        if cached is not None and cached[0] == batch_size:
            return cached[1].copy()

        self._fit_node_models()
        if pkgfn_config is not None and not self.model.missing_nodes:
            action = pkgfn_config.select_action(self.data, self.model)
            node = action.node
            result = pd.DataFrame([action.inputs])
        else:
            if self.model.missing_nodes or self.data.complete_measurements.empty:
                node = self._next_node()
                node_object = self.dag.node_by_name[node]
                if node_object.inputs:
                    result = RandomRecommender().recommend(
                        batch_size=batch_size,
                        searchspace=node_object.searchspace,
                    )
                else:
                    if batch_size != 1:
                        raise IncompatibilityError(
                            f"Node '{node}' has no own parameters, so only a single "
                            f"action can be recommended, but {batch_size=} "
                            "was requested."
                        )
                    result = pd.DataFrame(index=[0])
            else:
                node, result = self._recommend_optimum(batch_size)

            missing_parents = sorted(
                set(self.dag.parents_of(node)) - set(self._current_experiment)
            )
            if missing_parents:
                raise IncompatibilityError(
                    f"Cannot recommend node '{node}' because the output of parent "
                    f"node(s) {missing_parents} has not been measured in the current "
                    f"experiment."
                )
            result = result.assign(
                **{
                    parent.output: self._current_experiment[parent.name].output
                    for parent in self.dag.node_by_name[node].parents
                }
            )

        node_object = self.dag.node_by_name[node]
        result = result[list(node_object.model_input_names)].reset_index(drop=True)
        result.insert(0, NODE, node)
        experiment_id = self._current_experiment_id
        if experiment_id is None:
            experiment_id = f"exp-{self._experiment_counter + 1}"
        result.insert(0, EXPERIMENT_ID, experiment_id)

        # Publish recommendation state only after every computation succeeds.
        if self._current_experiment_id is None:
            self._experiment_counter += 1
            self._current_experiment_id = experiment_id
            self._current_experiment = {}
        self._cached_recommendation = (batch_size, result)
        return result.copy()

    def _fit_node_models(self) -> None:
        """Fit the node models from the current measurements.

        The fit is partial-safe, so upstream-only measurements already train
        their node models before complete DAG rows exist.
        """
        if not self.data.measurements and self.data.complete_measurements.empty:
            return
        self.model.fit_node_data(
            {
                node.name: self.data.node_training_data(self.dag, node.name)
                for node in self.dag.nodes
            },
            allow_partial=True,
        )

    def _next_node(self) -> str:
        """Return the next node to execute in the current experiment.

        Returns:
            The first node in topological order that has not been measured in
            the current experiment and whose parents all have.

        Raises:
            RuntimeError: If every node has already been measured.
        """
        for name in self.dag.topological_order:
            if name in self._current_experiment:
                continue
            if all(
                parent in self._current_experiment
                for parent in self.dag.parents_of(name)
            ):
                return name
        raise RuntimeError(
            "Every node of the current experiment has already been measured."
        )

    def _recommend_optimum(self, batch_size: int) -> tuple[str, pd.DataFrame]:
        """Determine the next action from all possible DAG configurations.

        All configurations are scored with the acquisition function. Candidates
        that conflict with measurements already recorded in the current
        experiment are filtered out, so that the recommendation continues the
        current experiment rather than starting a new one.

        Args:
            batch_size: The number of node actions to recommend.

        Returns:
            The node to execute next and the distinct parameter values of that
            node among the best-scoring configurations.

        Raises:
            IncompatibilityError: If the DAG search space is not purely discrete
                or fewer distinct actions than requested are available, or if an
                acquisition value is not finite.
        """
        searchspace = self.dag.full_searchspace
        if not searchspace.continuous.is_empty:
            raise IncompatibilityError(
                f"{self.__class__.__name__} scores every configuration of the "
                f"DAG and therefore requires a purely discrete search space."
            )

        candidates = searchspace.discrete.exp_rep

        # Filter to candidates matching the current experiment.
        for node_name, measurement in self._current_experiment.items():
            own = list(self.dag.owned_parameter_names(node_name))
            fixed = pd.Series(
                {name: measurement.inputs[name] for name in own}, dtype=object
            )
            candidates = candidates.loc[candidates[own].eq(fixed).all(axis=1)]

        if candidates.empty:
            raise IncompatibilityError(
                "No configuration matches the measurements of the current experiment."
            )

        recommender = BotorchRecommender(
            surrogate_model=self.model,
            acquisition_function=self.acquisition_function,
        )
        values = recommender.acquisition_values(
            candidates,
            searchspace,
            self.dag.objective,
            self.data.complete_measurements,
        )
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise IncompatibilityError("The acquisition values must be finite.")

        node = self._next_node()
        own_columns = list(self.dag.owned_parameter_names(node))
        order = values.sort_values(ascending=False, kind="stable").index
        action_frame = candidates.loc[order, own_columns].drop_duplicates()
        if len(action_frame) < batch_size:
            raise IncompatibilityError(
                f"Only {len(action_frame)} distinct actions of node '{node}' are "
                f"available, but {batch_size=} were requested."
            )
        return node, action_frame.head(batch_size)


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
