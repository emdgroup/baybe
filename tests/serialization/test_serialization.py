"""Basic serialization tests."""

import pytest
from cattrs.errors import ClassValidationError, ForbiddenExtraKeysError
from hypothesis import given

from tests.hypothesis_strategies import baybe_objects
from tests.serialization.utils import Serializable, _get_highest_baseclass


@given(baybe_objects)
def test_extra_keys(obj: Serializable):
    """Deserialization with extra keywords is forbidden."""
    cls = _get_highest_baseclass(type(obj))
    dct = obj.to_dict(add_type=True)
    dct["_some_extra_key_"] = "value"

    with pytest.raises(ClassValidationError) as ex:
        cls.from_dict(dct)
    assert all(isinstance(e, ForbiddenExtraKeysError) for e in ex.value.exceptions)
