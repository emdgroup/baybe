"""Abstract base class for transfergpbo surrogates."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from attrs import define, field
from botorch.models.model import Model
from botorch.posteriors import Posterior
from torch import Tensor
from typing_extensions import override

from baybe.exceptions import ModelNotTrainedError
from baybe.parameters import TaskParameter
from baybe.parameters.base import Parameter  # Add this import
from baybe.surrogates.base import Surrogate

# Add these imports for the scaler factory
if TYPE_CHECKING:
    from botorch.models.transforms.input import InputTransform


@define
class TransferGPBOSurrogate(Surrogate, ABC):
    """Abstract base class for all transfergpbo model wrappers.

    This class handles the common BayBE integration logic for transfer learning
    models from the transfergpbo package. It translates between BayBE's search
    space representation and the tensor format expected by transfergpbo models.

    Key responsibilities:
    - Extract task information from BayBE's TaskParameter
    - Identify target task from active_values
    - Reorder tensors to match transfergpbo's expected format (task_feature=-1)
    - Handle training and prediction workflows
    """

    # Class variables
    supports_transfer_learning: ClassVar[bool] = True
    """Class variable encoding transfer learning support."""

    supports_multi_output: ClassVar[bool] = False
    """Class variable encoding multi-output compatibility."""

    # Instance variables
    input_dim: int = field()
    """Dimensionality of the input space (excluding task feature)."""

    # Private attributes for storing model state
    _model: Model = field(init=False, default=None, eq=False)
    """The actual transfergpbo model instance."""

    _target_task_id: int = field(init=False, default=None, eq=False)
    """Numeric ID of the target task."""

    _task_column_idx: int = field(init=False, default=None, eq=False)
    """Column index of task feature in BayBE's computational representation."""

    @abstractmethod
    def _create_model(self) -> Model:
        """Create the specific transfergpbo model instance.

        This method must be implemented by subclasses to instantiate
        their specific model (e.g., MHGPModel, MHGPModelStable).

        Returns:
            The created transfergpbo model instance.
        """
        pass

    def _identify_target_task(self) -> int:
        """Identify target task from TaskParameter active_values.

        Extracts the target task from the TaskParameter's active_values
        and converts the task name to its corresponding numeric ID.

        Returns:
            Numeric ID of the target task.

        Raises:
            ValueError: If no TaskParameter found in search space.
            ValueError: If not exactly one active task is specified.
        """
        # Find the TaskParameter
        task_param = None
        for param in self._searchspace.parameters:
            if isinstance(param, TaskParameter):
                task_param = param
                break

        if task_param is None:
            raise ValueError(
                "No TaskParameter found in search space. "
                "Transfer learning requires a TaskParameter."
            )

        # Get active values (target tasks)
        active_values = task_param.active_values

        if len(active_values) != 1:
            raise ValueError(
                f"Expected exactly one active task, got {len(active_values)}: {active_values}. "
                f"Transfer learning requires exactly one target task."
            )

        target_task_name = active_values[0]

        # Convert task name to numeric ID using TaskParameter's values order
        # TaskParameter uses integer encoding: values[0]→0, values[1]→1, etc.
        task_name_to_id = {name: idx for idx, name in enumerate(task_param.values)}

        if target_task_name not in task_name_to_id:
            raise ValueError(
                f"Target task '{target_task_name}' not found in TaskParameter values: "
                f"{list(task_name_to_id.keys())}"
            )

        target_task_id = task_name_to_id[target_task_name]

        return target_task_id

    def _ensure_task_feature_last(self, tensor: Tensor, task_col_idx: int) -> Tensor:
        """Reorder tensor columns so task feature is last.

        Your transfergpbo models expect task_feature=-1, but BayBE might
        place the task column at any position. This method reorders the
        columns to ensure the task feature is in the last position.

        Args:
            tensor: Input tensor with task feature at arbitrary position.
            task_col_idx: Current column index of the task feature.

        Returns:
            Tensor with task feature moved to the last column.
        """
        if task_col_idx == -1 or task_col_idx == tensor.shape[1] - 1:
            return tensor  # Already in the last position

        # Create new column order: all features except task, then task
        n_cols = tensor.shape[1]
        feature_cols = [i for i in range(n_cols) if i != task_col_idx]
        reordered_cols = feature_cols + [task_col_idx]

        return tensor[:, reordered_cols]

    def _validate_transfer_learning_context(self) -> None:
        """Validate that we have a proper transfer learning setup.

        Raises:
            ValueError: If no task parameter found in search space.
            ValueError: If input dimensions don't match expected format.
        """
        if self._searchspace.task_idx is None:
            raise ValueError(
                "No task parameter found in search space. "
                "TransferGPBOSurrogate requires a TaskParameter for transfer learning."
            )

        # Validate that we have the expected number of feature dimensions
        expected_total_dims = self.input_dim + 1  # features + task
        actual_total_dims = len(self._searchspace.comp_rep_columns)

        if actual_total_dims != expected_total_dims:
            raise ValueError(
                f"Expected {expected_total_dims} total dimensions "
                f"({self.input_dim} features + 1 task), "
                f"but got {actual_total_dims} from search space."
            )

    @staticmethod
    @override
    def _make_parameter_scaler_factory(
        parameter: Parameter,
    ) -> type["InputTransform"] | None:
        """Prevent task parameters from being normalized."""
        from botorch.models.transforms.input import Normalize

        from baybe.parameters import TaskParameter

        if isinstance(parameter, TaskParameter):
            return None  # No scaling for task parameters
        return Normalize  # Normal scaling for continuous parameters

    @override
    def _fit(self, train_x: Tensor, train_y: Tensor) -> None:
        """Fit the transfer learning model.

        This method handles the common training workflow for all transfergpbo models:
        1. Validate transfer learning context
        2. Identify target task from TaskParameter
        3. Reorder tensors to match transfergpbo format
        4. Train the model using meta_fit() and fit()

        Args:
            train_x: Training inputs in BayBE's computational representation.
                    Shape: (n_points, n_features + 1) where last column may be task indices.
            train_y: Training targets. Shape: (n_points, 1).
        """
        # Check if we receive empty data
        if train_x.shape[0] == 0 or train_y.shape[0] == 0:
            raise ValueError(
                f"Received empty training data! train_x.shape={train_x.shape}, train_y.shape={train_y.shape}"
            )

        # 1. Validate transfer learning context
        self._validate_transfer_learning_context()

        # 2. Identify target task from TaskParameter active_values
        self._target_task_id = self._identify_target_task()
        self._task_column_idx = self._searchspace.task_idx

        # 3. Reorder tensor so task feature is last (required by transfergpbo models)
        X_multi = self._ensure_task_feature_last(train_x, self._task_column_idx)

        # 4. Create model if not exists
        if self._model is None:
            self._model = self._create_model()

        # 5. Train the transfergpbo model
        # meta_fit: Train source GPs on residuals
        self._model.meta_fit(
            X_multi, train_y, task_feature=-1, target_task=self._target_task_id
        )

        # fit: Train target GP on remaining residuals
        self._model.fit(
            X_multi, train_y, task_feature=-1, target_task=self._target_task_id
        )

    @override
    def _posterior(self, candidates_comp_scaled: Tensor, /) -> Posterior:
        """Compute posterior predictions.

        This method handles prediction for candidates that may contain
        task indices for any of the available tasks (source or target).

        Args:
            candidates_comp_scaled: Candidate points in computational representation.
                                  Should include task indices in the same format as training data.

        Returns:
            Posterior distribution for the candidate points.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._model is None:
            raise ModelNotTrainedError(
                "Model must be fitted before making predictions. Call fit() first."
            )

        # Ensure task feature is in the last column (as expected by transfergpbo models)
        candidates_reordered = self._ensure_task_feature_last(
            candidates_comp_scaled, self._task_column_idx
        )

        # Get predictions from the transfergpbo model
        # The model can predict for any task based on the task indices in the input
        return self._model.posterior(candidates_reordered)

    @override
    def to_botorch(self) -> Model:
        """Return the trained transfergpbo model.

        Returns:
            The underlying transfergpbo model instance.

        Raises:
            ModelNotTrainedError: If model hasn't been trained yet.
        """
        if self._model is None:
            raise ModelNotTrainedError(
                "Model must be fitted before accessing the BoTorch model. "
                "Call fit() first."
            )
        return self._model

    def __str__(self) -> str:
        """String representation of the surrogate."""
        fields = [
            f"Input Dim: {self.input_dim}",
            f"Supports Transfer Learning: {self.supports_transfer_learning}",
            f"Model Type: {self.__class__.__name__}",
        ]

        if self._model is not None:
            fields.append("Status: Trained")
            if self._target_task_id is not None:
                fields.append(f"Target Task ID: {self._target_task_id}")
        else:
            fields.append("Status: Not Trained")

        return f"{self.__class__.__name__}({', '.join(fields)})"
