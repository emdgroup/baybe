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
from attrs.validators import ge, instance_of, le

from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import IncompatibilityError, InfeasibilityError
from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender

from ._columns import EXPERIMENT_ID, NODE
from .acquisition import PartialKnowledgeGradientFN
from .dag import MultiStepDAG
from .data import (
    MultiStepData,
    NodeMeasurement,
    _extract_feasibility_values,
    _validate_node_inputs,
)
from .model import MultiStepModel, _AcquisitionSurrogateAdapter


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
    """The p-KGFN policy or stock acquisition function.

    ``None`` selects the default Bayesian acquisition, also with feasibility conditions.
    Random design is used only while the required initial observations are missing.
    """

    feasibility_threshold: float = field(
        default=0.5, validator=[instance_of(float), ge(0.0), le(1.0)], kw_only=True
    )
    """Minimum network success probability for stock acquisition candidates.

    p-KGFN uses the threshold configured on its acquisition policy instead.
    """

    _cached_recommendation: tuple[int, pd.DataFrame] | None = field(
        default=None, init=False, eq=False
    )
    """The cached recommendation and the batch size it was computed for."""

    _current_experiment_id: str | None = field(default=None, init=False, eq=False)
    """The experiment that is currently being built, if any."""

    _current_experiment: dict[str, NodeMeasurement] = field(
        factory=dict, init=False, eq=False
    )
    """Successful measurements of the current experiment, keyed by node name."""

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
        feasibility_threshold: float = 0.5,
    ) -> MultiStepCampaign:
        """Create a campaign for the given DAG.

        Args:
            dag: The DAG topology, including the optimization objective.
            acquisition_function: Optional acquisition function or its name.
            data: Optional initial measurement store.
            feasibility_threshold: Network success threshold for stock acquisitions.

        Returns:
            The configured campaign.
        """
        return cls(
            dag=dag,
            data=MultiStepData.empty() if data is None else data,
            acquisition_function=acquisition_function,
            feasibility_threshold=feasibility_threshold,
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
        """Record node outputs and auxiliary values used by feasibility conditions.

        Measurements are stored as individual node observations. When
        ``experiment_id`` matches the current experiment and the measurement
        completes it, a complete DAG row is assembled automatically.
        An explicit failure closes the active experiment without creating a complete
        row; upstream observations remain available for training.

        Args:
            node: The observed node.
            y: The node-output column and any auxiliary columns referenced by
                feasibility conditions. Missing values are allowed only when another
                observed condition establishes failure.
            x: The full node-model inputs, with one column per own parameter and
                per parent output, and the same number of rows as ``y``.
            experiment_id: An optional experiment identifier. When omitted, the
                current experiment identifier is used.

        Returns:
            The recorded measurements.

        Raises:
            IncompatibilityError: If the node, the columns, the values, or the
                experiment identifier are invalid.
            IncompleteMeasurementsError: If condition values are missing.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        node_object = self.dag.node_by_name.get(node)
        if node_object is None:
            raise IncompatibilityError(
                f"Node '{node}' is not part of the DAG. Known nodes: "
                f"{sorted(self.dag.node_by_name)}."
            )

        if y.empty:
            raise IncompatibilityError("Cannot add an empty measurement batch.")
        if not y.columns.is_unique:
            raise IncompatibilityError(
                f"Output columns for node '{node}' must be unique."
            )
        auxiliary_columns = {
            condition.column
            for condition in node_object.feasibility_conditions
            if condition.column != node_object.output
        }
        allowed = {node_object.output, *auxiliary_columns}
        if set(y.columns) - allowed:
            raise IncompatibilityError(
                f"Output data for node '{node}' must use columns {sorted(allowed)}, "
                f"got {list(y.columns)}."
            )
        success = _extract_feasibility_values(node_object, y)
        outputs = (
            y[node_object.output]
            if node_object.output in y
            else pd.Series(None, index=y.index)
        )
        if (success & outputs.isna()).any():
            raise IncompatibilityError(
                f"Successful observations of node '{node}' require an output."
            )

        if x is None:
            raise IncompatibilityError(
                f"The model input 'x' is required for node '{node}'. "
                f"Expected columns: {list(node_object.model_input_names)}."
            )
        _validate_node_inputs(self.dag, node_object, x)

        if len(x) != len(y):
            raise IncompatibilityError(
                f"Input and output data for node '{node}' must have the same "
                f"number of rows, got {len(x)} and {len(y)}."
            )

        if experiment_id is None:
            experiment_id = self._current_experiment_id

        measurement_rows = y.to_dict("records")
        measurements = tuple(
            NodeMeasurement(
                node=node,
                inputs=cast(dict[str, Any], inputs),
                output=None if pd.isna(output) else output,
                experiment_id=experiment_id,
                condition_values={
                    column: values[column]
                    for column in auxiliary_columns
                    if column in values
                },
            )
            for inputs, output, values in zip(
                x.to_dict("records"),
                outputs.tolist(),
                measurement_rows,
                strict=True,
            )
        )

        # Validate and assemble proposed state before publishing any campaign state.
        data = self.data.add(*measurements)
        data.validate(self.dag)
        current = self._current_experiment.copy()
        current_id = self._current_experiment_id
        if experiment_id is not None and experiment_id == self._current_experiment_id:
            current[node] = measurements[-1]
            if not success.all():
                current_id, current = None, {}
            elif data.is_experiment_complete(self.dag, experiment_id):
                row = data.assemble_complete_row(self.dag, experiment_id)
                row[EXPERIMENT_ID] = experiment_id
                data = data.add_complete(row)
                data.validate(self.dag)
                current_id, current = None, {}

        self.data = data
        self._current_experiment_id = current_id
        self._current_experiment = current
        self._cached_recommendation = None

        return measurements

    def add_complete_measurements(self, measurements: pd.DataFrame) -> None:
        """Load complete DAG rows.

        The rows are used by the acquisition function for its reference
        computations. Node models are additionally trained on the per-node views
        of the same rows.

        Args:
            measurements: Complete DAG rows in experimental representation with
                one column per DAG parameter and node output plus all configured
                auxiliary condition columns. Additional columns are ignored. All
                configured conditions must pass.

        Raises:
            IncompatibilityError: If required columns or values are invalid.
            IncompleteMeasurementsError: If condition values are missing.
            InvalidTargetValueError: If condition values have invalid types or values.
        """
        required = self.dag._complete_measurement_columns
        if measurements.empty:
            raise IncompatibilityError("Cannot import an empty measurement batch.")
        if not measurements.columns.is_unique:
            raise IncompatibilityError("Complete measurement columns must be unique.")
        missing = [column for column in required if column not in measurements.columns]
        if missing:
            raise IncompatibilityError(
                f"Complete measurements are missing required columns: {missing}."
            )
        data = self.data.add_complete(measurements[list(required)])
        data.validate(self.dag)
        self.data = data
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

        A random initial design is used only while the required output or feasibility
        models cannot be fitted. Thereafter, the configured acquisition selects the
        action; ``None`` selects the default Bayesian acquisition.

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
        needs_initial_data = (
            self.model.missing_nodes or self.model.missing_feasibility_nodes
        )
        if pkgfn_config is not None and not needs_initial_data:
            action = pkgfn_config.select_action(self.data, self.model)
            node = action.node
            result = pd.DataFrame([action.inputs])
        else:
            if needs_initial_data or self.data.complete_measurements.empty:
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
            feasibility_frames={
                node.name: self.data.node_feasibility_training_data(self.dag, node.name)
                for node in self.dag.nodes
                if node.feasibility_conditions
            },
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
            InfeasibilityError: If no candidate reaches the network success threshold.
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

        if any(node.feasibility_conditions for node in self.dag.nodes):
            import torch

            from baybe.utils.dataframe import to_tensor

            points = to_tensor(searchspace.transform(candidates)).unsqueeze(-2)
            with torch.no_grad():
                _, probabilities = self.model.draw_network_outcomes(
                    points, self.model.num_network_samples
                )
            feasible = probabilities.mean(dim=0).squeeze(-1).cpu().numpy()
            candidates = candidates.loc[feasible >= self.feasibility_threshold]
            if candidates.empty:
                raise InfeasibilityError(
                    "No candidate reaches the network feasibility threshold "
                    f"{self.feasibility_threshold}."
                )

        recommender = BotorchRecommender(
            surrogate_model=_AcquisitionSurrogateAdapter(self.model),
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
