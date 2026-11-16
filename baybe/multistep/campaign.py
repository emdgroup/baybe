"""Multi-step campaign facade for DAG-structured experiments.

The facade owns the multi-step measurements and fits a single
:class:`~baybe.multistep.model.MultiStepModel` that is composed of one model per
DAG node. It hands that model to a stock
:class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`
directly. A recommendation sees the DAG-wide search space and the final-node
objective; upstream-only measurements still train their node models.

The campaign tracks the plan it is currently executing: the recommended node
actions that have not been measured yet and the outputs already observed within
the plan. A measurement is matched against the open actions, so callers never
have to repeat the inputs of the step they are in. Once every node of the plan
has been measured, the campaign assembles the complete DAG row and starts the
next plan.
"""

from __future__ import annotations

import gc
from collections.abc import Mapping
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from attrs import define, field
from attrs.converters import optional
from attrs.validators import instance_of

from baybe.acquisition.base import AcquisitionFunction
from baybe.acquisition.utils import convert_acqf
from baybe.exceptions import IncompatibilityError
from baybe.recommenders.base import RecommenderProtocol
from baybe.recommenders.pure.bayesian.botorch.core import BotorchRecommender
from baybe.recommenders.pure.nonpredictive.sampling import RandomRecommender
from baybe.serialization import SerialMixin
from baybe.surrogates.base import SurrogateProtocol

from ._columns import NODE
from .dag import MultiStepDAG
from .data import MultiStepData, NodeMeasurement
from .model import MultiStepModel
from .node import MultiStepNode


@define(frozen=True)
class _OpenAction:
    """A recommended node action that has not been measured yet.

    Args:
        node: The node to execute.
        inputs: The full node-model input, that is, the node's own parameter
            values and the outputs of its parents.
    """

    node: str = field()
    """The node to execute."""

    inputs: dict[str, Any] = field(converter=dict)
    """The full node-model input of the action."""


@define(slots=False)
class MultiStepCampaign(SerialMixin):
    """Stateful facade for multi-step Bayesian optimization.

    Node surrogates that produce ensemble posteriors, such as
    :class:`~baybe.surrogates.random_forest.RandomForestSurrogate`, can drive
    recommendations over discrete search spaces. Because ensemble posteriors are
    not differentiable, continuous search spaces require gradient-capable node
    surrogates.

    The facade fits a single multi-step model composed of one model per node and
    hands it to a stock
    :class:`~baybe.recommenders.pure.bayesian.botorch.core.BotorchRecommender`
    directly. The acquisition function is configurable; other recommender
    settings are not exposed.

    Args:
        dag: The DAG topology, including the optimization objective.
        data: The measurement store.
        acquisition_function: Optional acquisition function for the stock
            recommender. When omitted, BayBE selects the context default.
    """

    dag: MultiStepDAG = field(validator=instance_of(MultiStepDAG))
    """The DAG topology."""

    data: MultiStepData = field(factory=MultiStepData.empty)
    """The measurement store holding node measurements and complete rows."""

    acquisition_function: AcquisitionFunction | None = field(
        default=None, converter=optional(convert_acqf)
    )
    """The acquisition function used by the stock recommender."""

    _cached_recommendation: tuple[int, pd.DataFrame] | None = field(
        default=None, init=False, eq=False
    )
    """The cached recommendation and the batch size it was computed for."""

    _open_actions: list[_OpenAction] = field(factory=list, init=False, eq=False)
    """The recommended node actions of the current plan, in recommendation order."""

    _plan: dict[str, NodeMeasurement] = field(factory=dict, init=False, eq=False)
    """The measured steps of the plan that is currently being executed."""

    @data.validator
    def _validate_data(  # noqa: DOC101, DOC103
        self, _: Any, value: MultiStepData
    ) -> None:
        """Validate the measurement store against the DAG."""
        value.validate(self.dag)

    # --------------------------------------------------------------------------
    # Factory
    # --------------------------------------------------------------------------

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

    # --------------------------------------------------------------------------
    # Public state
    # --------------------------------------------------------------------------

    @cached_property
    def model(self) -> MultiStepModel:
        """The multi-step model owned by the campaign."""
        return self.dag.make_model()

    @property
    def recommender(self) -> RecommenderProtocol:
        """The stock recommender bound to the multi-step model."""
        return self._make_recommender(self.model)

    # --------------------------------------------------------------------------
    # Data ingestion
    # --------------------------------------------------------------------------

    def add_measurements(
        self,
        node: str,
        y: pd.DataFrame,
        x: pd.DataFrame | None = None,
    ) -> tuple[NodeMeasurement, ...]:
        """Record observed node outputs.

        Measurements of open actions advance the plan that is currently being
        executed. When ``x`` is omitted, the inputs of the open action are used,
        so the caller never has to repeat the step's inputs. Measurements that
        do not match an open action are treated as historical data.

        Args:
            node: The observed node.
            y: The observed node outputs, with exactly one column named after
                the node output.
            x: The full node-model inputs, with one column per own parameter and
                per parent output, and the same number of rows as ``y``. When
                omitted, the inputs of the open actions of ``node`` are used.

        Returns:
            The recorded measurements.

        Raises:
            IncompatibilityError: If the node, the columns, the values, or the
                match to the open actions are invalid.
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
            x = self._open_action_inputs(node_object, len(y))

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

        measurements = tuple(
            NodeMeasurement(
                node=node,
                inputs={name: _python_scalar(x[name].iloc[row]) for name in expected},
                output=_python_scalar(y[node_object.output].iloc[row]),
            )
            for row in range(len(x))
        )

        completed = self._advance_plan(measurements)
        data = self.data.add(*measurements)
        if completed is not None:
            data = data.add_complete(completed)
        self._commit(data)
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
        self._commit(self.data.add_complete(measurements[list(required)]))

    # --------------------------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------------------------

    def recommend(self, batch_size: int = 1) -> pd.DataFrame:
        """Recommend the next batch of node actions.

        The optimal point is determined over the complete DAG search space: every
        configuration of all node parameters is scored with the acquisition
        function on the shared network posterior, without any projection onto
        the currently open plan. A configuration is matched against the plan
        that is currently being executed; the nodes it already shares with the
        plan do not have to be measured again. The acquisition value is divided
        by the cost of the remaining nodes, so that an expensive node is only
        executed when the completed experiment justifies the expense. Because the
        best configuration may disagree with the measurements of the open plan,
        recommending an upstream node again, and thereby abandoning the plan, is
        part of the same decision.

        The next action is the first node of the best configuration that is not
        yet measured within the plan and whose parents are. A random initial
        design is used while no complete DAG row exists or while a node model
        cannot be fitted. In that phase the current plan is continued, because
        the first complete DAG row has to be assembled from a finished plan.

        Args:
            batch_size: The number of node actions to recommend.

        Returns:
            A dataframe with one row per recommended action, holding ``node``
            and the parameters of the node to execute.

        Raises:
            ValueError: If ``batch_size`` is smaller than 1.
        """
        if batch_size < 1:
            raise ValueError(
                f"You must at least request one recommendation per batch, but "
                f"provided {batch_size=}."
            )

        cached = self._cached_recommendation
        if cached is not None and cached[0] == batch_size:
            return cached[1].copy()

        self._fit_node_models()

        if self.data.complete_measurements.empty or self.model.missing_nodes:
            node = self._next_node(self._plan)
            result = RandomRecommender().recommend(
                batch_size=batch_size,
                searchspace=self.dag.node_by_name[node].searchspace,
            )
        else:
            node, result = self._recommend_optimum(batch_size)

        own = self.dag.owned_parameter_names(node)
        result = result[list(own)].reset_index(drop=True)
        result.insert(0, NODE, node)

        parent_outputs = {
            self.dag.node_by_name[parent].output: self._plan[parent].output
            for parent in self.dag.parents_of(node)
            if parent in self._plan
        }
        self._open_actions.extend(
            _OpenAction(
                node=node,
                inputs={
                    **{name: _python_scalar(result[name].iloc[row]) for name in own},
                    **parent_outputs,
                },
            )
            for row in range(len(result))
        )

        self._cached_recommendation = (batch_size, result)
        return result.copy()

    # --------------------------------------------------------------------------
    # Internal data handling
    # --------------------------------------------------------------------------

    def _advance_plan(
        self, measurements: tuple[NodeMeasurement, ...]
    ) -> pd.DataFrame | None:
        """Advance the current plan with matched measurements.

        Args:
            measurements: The newly recorded measurements.

        Returns:
            The complete DAG row of the plan if the measurements finish it,
            otherwise ``None``.
        """
        for measurement in measurements:
            if not self._close_open_action(measurement):
                continue
            self._start_chain(measurement)

        if set(self._plan) != set(self.dag.node_by_name):
            return None

        row = self._plan_row()
        self._plan = {}
        return row

    def _close_open_action(self, measurement: NodeMeasurement) -> bool:
        """Remove the open action matching a measurement.

        Args:
            measurement: The recorded measurement.

        Returns:
            Whether an open action was matched.
        """
        own = self.dag.owned_parameter_names(measurement.node)
        for index, action in enumerate(self._open_actions):
            if action.node != measurement.node:
                continue
            if all(action.inputs[name] == measurement.inputs[name] for name in own):
                del self._open_actions[index]
                return True
        return False

    def _commit(self, data: MultiStepData) -> None:
        """Validate a new measurement store and publish it atomically."""
        data.validate(self.dag)
        self.data = data
        self._cached_recommendation = None

    # --------------------------------------------------------------------------
    # Internal recommendation handling
    # --------------------------------------------------------------------------

    def _cost_adjusted_scores(
        self, values: np.ndarray, costs: np.ndarray
    ) -> np.ndarray:
        """Convert acquisition values into comparable utilities per unit cost.

        A candidate value refers to one complete DAG row, so its price is the
        cost of finishing the candidate plan rather than the cost of its next
        action. Finishing the current plan is cheaper than starting a new one,
        and an expensive node only justifies itself when the completed row
        promises a proportionally higher utility. Log-scaled acquisition
        functions are exponentiated before the division; non-negative functions
        already report a utility. When all candidates cost the same, the values
        are returned unchanged.

        Args:
            values: The acquisition value of each candidate.
            costs: The cost of completing the plan of each candidate.

        Returns:
            The cost-adjusted score of each candidate.

        Raises:
            IncompatibilityError: If the costs differ and the acquisition
                function can produce negative values without being log-scaled.
        """
        if np.all(costs == costs[0]):
            return values

        recommender = self._make_recommender(self.model)
        acqf = recommender._get_acquisition_function(self.dag.objective)
        if acqf.abbreviation.lstrip("q").startswith("Log"):
            utilities = np.exp(values)
        elif np.all(values[~np.isnan(values)] >= 0):
            utilities = values
        else:
            raise IncompatibilityError(
                f"Cost-aware candidate comparison requires a non-negative "
                f"acquisition function, but '{type(acqf).__name__}' can produce "
                f"negative values. Use a log-scaled or non-negative acquisition "
                f"function."
            )
        return utilities / costs

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

    def _make_recommender(self, model: SurrogateProtocol) -> BotorchRecommender:
        """Create the stock recommender bound to the given model."""
        return BotorchRecommender(
            surrogate_model=model,
            acquisition_function=self.acquisition_function,
        )

    def _next_node(self, plan: Mapping[str, NodeMeasurement]) -> str:
        """Return the node whose execution comes next in a plan.

        Args:
            plan: The measured steps of the plan.

        Returns:
            The first node in topological order that has not been measured in
            the plan yet and whose parents all have.

        Raises:
            RuntimeError: If every plan node has already been measured.
        """
        for name in self.dag.topological_order:
            if name in plan:
                continue
            if all(parent in plan for parent in self.dag.parents_of(name)):
                return name
        raise RuntimeError(
            f"Every node of the given plan of {self.__class__.__name__} has "
            f"already been measured."
        )

    def _open_action_inputs(self, node: MultiStepNode, n_rows: int) -> pd.DataFrame:
        """Resolve the node-model input of the open actions of a node.

        Args:
            node: The observed node.
            n_rows: The number of measured rows.

        Returns:
            The full node-model input of one open action per row.

        Raises:
            IncompatibilityError: If the number of open actions does not match
                the number of rows.
        """
        actions = [action for action in self._open_actions if action.node == node.name]
        if len(actions) != n_rows:
            raise IncompatibilityError(
                f"Node '{node.name}' has {len(actions)} open action(s) but "
                f"{n_rows} measurement row(s) were provided. Pass the model "
                f"input explicitly via 'x'."
            )
        return pd.DataFrame(
            [action.inputs for action in actions], columns=list(node.model_input_names)
        )

    def _plan_row(self) -> pd.DataFrame:
        """Assemble the complete DAG row of the finished plan.

        Returns:
            A single-row dataframe with one column per DAG parameter and one per
            node output.
        """
        row: dict[str, Any] = {}
        for node in self.dag.nodes:
            measurement = self._plan[node.name]
            for name in self.dag.owned_parameter_names(node.name):
                row[name] = measurement.inputs[name]
            row[node.output] = measurement.output
        columns = [
            *self.dag.parameter_by_name,
            *(node.output for node in self.dag.nodes),
        ]
        return pd.DataFrame([row], columns=columns)

    def _recommend_optimum(self, batch_size: int) -> tuple[str, pd.DataFrame]:
        """Determine the optimal configuration over the complete search space.

        Args:
            batch_size: The number of node actions to recommend.

        Returns:
            The node to execute next and the distinct parameter values of that
            node among the best-scoring configurations.

        Raises:
            IncompatibilityError: If the DAG search space is not purely discrete
                or fewer distinct actions than requested are available.
        """
        searchspace = self.dag.full_searchspace
        if not searchspace.continuous.is_empty:
            raise IncompatibilityError(
                f"{self.__class__.__name__} scores every configuration of the "
                f"DAG and therefore requires a purely discrete search space."
            )

        candidates = searchspace.discrete.exp_rep
        values = self._make_recommender(self.model).acquisition_values(
            candidates,
            searchspace,
            self.dag.objective,
            self.data.complete_measurements,
        )
        remaining = [
            self._remaining_nodes(configuration)
            for configuration in candidates.to_dict("records")
        ]
        costs = np.array(
            [
                sum(self.dag.node_by_name[name].cost for name in names)
                for names in remaining
            ],
            dtype=float,
        )
        scores = self._cost_adjusted_scores(values.to_numpy(dtype=float), costs)
        order = np.argsort(
            -np.where(np.isfinite(scores), scores, -np.inf), kind="stable"
        )

        node = remaining[order[0]][0]
        own = list(self.dag.owned_parameter_names(node))
        actions = candidates.iloc[[i for i in order if remaining[i][0] == node]][own]
        actions = actions.drop_duplicates()
        if len(actions) < batch_size:
            raise IncompatibilityError(
                f"Only {len(actions)} distinct actions of node '{node}' are "
                f"available, but {batch_size=} were requested."
            )
        return node, actions.head(batch_size)

    def _remaining_nodes(self, configuration: Mapping[Any, Any]) -> tuple[str, ...]:
        """Return the nodes that still have to be executed for a configuration.

        A node counts as done if the current plan measured it with the
        configuration's own parameter values and all its parents are done.

        Args:
            configuration: The parameter values of a complete DAG configuration.

        Returns:
            The nodes that are not done, in topological order.
        """
        done: set[str] = set()
        for name in self.dag.topological_order:
            measurement = self._plan.get(name)
            if measurement is None:
                continue
            if not all(parent in done for parent in self.dag.parents_of(name)):
                continue
            own = self.dag.owned_parameter_names(name)
            if all(measurement.inputs[p] == configuration[p] for p in own):
                done.add(name)
        return tuple(n for n in self.dag.topological_order if n not in done)

    def _start_chain(self, measurement: NodeMeasurement) -> None:
        """Continue the plan with a measurement or start a new chain.

        A measurement of a node that is already part of the plan replaces the
        earlier observation. Its descendants are dropped from the plan together
        with their open actions, because they belonged to the replaced chain.

        Args:
            measurement: The measurement that is added to the plan.
        """
        existing = self._plan.get(measurement.node)
        if existing is not None:
            if (
                existing.inputs == measurement.inputs
                and existing.output == measurement.output
            ):
                return
            chain = (measurement.node, *self.dag.descendants_of(measurement.node))
            for name in chain:
                self._plan.pop(name, None)
            self._open_actions = [
                action for action in self._open_actions if action.node not in chain
            ]
        self._validate_plan_link(measurement)
        self._plan[measurement.node] = measurement

    def _validate_plan_link(self, measurement: NodeMeasurement) -> None:
        """Validate that a measurement continues the current plan consistently.

        Args:
            measurement: The measurement that is added to the plan.

        Raises:
            IncompatibilityError: If a parent output differs from the value
                measured within the plan.
        """
        for parent in self.dag.parents_of(measurement.node):
            if parent not in self._plan:
                continue
            expected = self._plan[parent].output
            value = measurement.inputs[self.dag.node_by_name[parent].output]
            if value != expected:
                raise IncompatibilityError(
                    f"The measurement of node '{measurement.node}' uses the "
                    f"output {value!r} of node '{parent}', but the current plan "
                    f"measured {expected!r}. Pass the step's input via 'x' only "
                    f"if it belongs to a different experiment."
                )


def _python_scalar(value: Any) -> Any:
    """Convert numpy scalars to plain Python values."""
    if isinstance(value, np.generic):
        return value.item()
    return value


# Collect leftover original slotted classes processed by ``attrs.define``.
gc.collect()
