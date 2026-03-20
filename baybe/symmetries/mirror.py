"""Mirror symmetry."""

from __future__ import annotations

import gc
from collections.abc import Iterable
from typing import TYPE_CHECKING

import pandas as pd
from attrs import define, field
from attrs.validators import instance_of
from typing_extensions import override

from baybe.symmetries.base import Symmetry
from baybe.utils.augmentation import df_apply_mirror_augmentation
from baybe.utils.validation import validate_is_finite

if TYPE_CHECKING:
    from baybe.parameters.base import Parameter
    from baybe.searchspace import SearchSpace


@define(frozen=True)
class MirrorSymmetry(Symmetry):
    """Class for representing mirror symmetries.

    A mirror symmetry expresses that certain parameters can be inflected at a mirror
    point without affecting the outcome of the model. For instance, when specified
    for parameter ``x`` and mirror point ``c``, the symmetry expresses that
    $f(..., c+x, ...) = f(..., c-x, ...)$.
    """

    _parameter_name: str = field(validator=instance_of(str), alias="parameter_name")
    """The name of the single parameter affected by the symmetry."""

    # object variables
    mirror_point: float = field(
        default=0.0, converter=float, validator=validate_is_finite, kw_only=True
    )
    """The mirror point."""

    @override
    @property
    def parameter_names(self) -> tuple[str]:
        return (self._parameter_name,)

    @override
    def augment_measurements(
        self,
        measurements: pd.DataFrame,
        parameters: Iterable[Parameter] | None = None,
    ) -> pd.DataFrame:
        # See base class.

        if not self.use_data_augmentation:
            return measurements

        measurements = df_apply_mirror_augmentation(
            measurements, self._parameter_name, mirror_point=self.mirror_point
        )

        return measurements

    @override
    def validate_searchspace_context(self, searchspace: SearchSpace) -> None:
        """See base class.

        Args:
            searchspace: The searchspace to validate against.

        Raises:
            TypeError: If the affected parameter is not numerical.
        """
        super().validate_searchspace_context(searchspace)

        param = searchspace.get_parameters_by_name(self.parameter_names)[0]
        if not param.is_numerical:
            raise TypeError(
                f"In a '{self.__class__.__name__}', the affected parameter must "
                f"be numerical. However, the parameter '{param.name}' is of "
                f"type '{param.__class__.__name__}' and is not numerical."
            )


# Collect leftover original slotted classes processed by `attrs.define`
gc.collect()
