"""
Test for save/load capability
"""

import os

from baybe.core import BayBE
from baybe.utils import add_fake_results


def test_data_consistency(
    baybe_object_batch3_iterations2,
    good_reference_values,  # pylint: disable=W0613  # TODO: work in progress
    batch_quantity,  # pylint: disable=W0613  # TODO: work in progress
    n_iterations,  # pylint: disable=W0613  # TODO: work in progress
):
    """
    Test equality after storing and loading a BayBE object.
    """
    # pylint: disable=protected-access

    # Load reference object
    baybe_obj_reference = baybe_object_batch3_iterations2

    # TODO: Serialization works, deserialization yet to be done
    baybe_obj_reference.json()
    raise NotImplementedError()

    baybe_obj_reference.save("./test.baybe")  # pylint: disable=unreachable

    # Restore BayBE object
    baybe_obj_loaded = BayBE.from_file("./test.baybe")

    # Asserts
    assert baybe_obj_reference.searchspace.discrete.metadata.equals(
        baybe_obj_loaded.searchspace.discrete.metadata
    ), "Problem directly after loading"
    assert baybe_obj_reference.searchspace.discrete.exp_rep.equals(
        baybe_obj_loaded.searchspace.discrete.exp_rep
    ), "Problem directly after loading"
    assert baybe_obj_reference.searchspace.discrete.comp_rep.equals(
        baybe_obj_loaded.searchspace.discrete.comp_rep
    ), "Problem directly after loading"
    assert baybe_obj_reference.measurements_exp.equals(
        baybe_obj_loaded.measurements_exp
    ), "Problem directly after loading"
    assert baybe_obj_reference.measurements_parameters_comp.equals(
        baybe_obj_loaded.measurements_parameters_comp
    ), "Problem directly after loading"
    assert baybe_obj_reference.measurements_targets_comp.equals(
        baybe_obj_loaded.measurements_targets_comp
    ), "Problem directly after loading"
    assert (
        baybe_obj_reference.batches_done == baybe_obj_loaded.batches_done
    ), "Problem directly after loading"
    assert (
        baybe_obj_reference.fits_done == baybe_obj_loaded.fits_done
    ), "Problem directly after loading"
    assert (
        baybe_obj_reference.config.dict() == baybe_obj_loaded.config.dict()
    ), "Problem directly after loading"
    assert baybe_obj_reference._cached_recommendation.equals(
        baybe_obj_loaded._cached_recommendation
    ), "Problem directly after loading"

    # Run some more iterations
    for k in range(n_iterations):
        rec = baybe_obj_reference.recommend(batch_quantity=batch_quantity)
        rec2 = baybe_obj_loaded.recommend(batch_quantity=batch_quantity)
        assert rec.equals(
            rec2
        ), f"Recommendations not identical after loading and running {k+1} iteration(s)"  # noqa: E501  # pylint: disable=C0301

        add_fake_results(
            rec,
            baybe_obj_reference,
            good_reference_values=good_reference_values,
        )
        rec2.Target_1 = rec.Target_1

        baybe_obj_reference.add_results(rec)
        baybe_obj_loaded.add_results(rec2)

    # Asserts
    assert baybe_obj_reference.searchspace.discrete.metadata.equals(
        baybe_obj_loaded.searchspace.discrete.metadata
    ), "Problem after continuation"
    assert baybe_obj_reference.searchspace.discrete.exp_rep.equals(
        baybe_obj_loaded.searchspace.discrete.exp_rep
    ), "Problem after continuation"
    assert baybe_obj_reference.searchspace.discrete.comp_rep.equals(
        baybe_obj_loaded.searchspace.discrete.comp_rep
    ), "Problem after continuation"
    assert baybe_obj_reference.measurements_exp.equals(
        baybe_obj_loaded.measurements_exp
    ), "Problem after continuation"
    assert baybe_obj_reference.measurements_parameters_comp.equals(
        baybe_obj_loaded.measurements_parameters_comp
    ), "Problem after continuation"
    assert baybe_obj_reference.measurements_targets_comp.equals(
        baybe_obj_loaded.measurements_targets_comp
    ), "Problem after continuation"
    assert (
        baybe_obj_reference.batches_done == baybe_obj_loaded.batches_done
    ), "Problem after continuation"
    assert (
        baybe_obj_reference.fits_done == baybe_obj_loaded.fits_done
    ), "Problem after continuation"
    assert (
        baybe_obj_reference.config.dict() == baybe_obj_loaded.config.dict()
    ), "Problem after continuation"
    assert baybe_obj_reference._cached_recommendation.equals(
        baybe_obj_loaded._cached_recommendation
    ), "Problem after continuation"

    os.remove("./test.baybe")
