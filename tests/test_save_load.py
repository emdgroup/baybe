"""
Test for save/load capability
"""

import os

from baybe.core import BayBE
from baybe.utils import add_fake_results


def test_data_consistency(
    baybe_object_batch3_iterations2, good_reference_values, batch_quantity
):
    """
    Test equality after storing and loading a BayBE object.
    """
    # Load reference object
    baybe_obj_reference = baybe_object_batch3_iterations2
    baybe_obj_reference.save("./test.baybe")

    # Restore BayBE object
    baybe_obj_loaded = BayBE.from_file("./test.baybe")

    # Asserts
    assert baybe_obj_reference.searchspace.metadata.equals(
        baybe_obj_loaded.searchspace.metadata
    ), "Problem directly after loading"
    assert baybe_obj_reference.searchspace.exp_rep.equals(
        baybe_obj_loaded.searchspace.exp_rep
    ), "Problem directly after loading"
    assert baybe_obj_reference.searchspace.comp_rep.equals(
        baybe_obj_loaded.searchspace.comp_rep
    ), "Problem directly after loading"
    assert baybe_obj_reference.measurements.equals(
        baybe_obj_loaded.measurements
    ), "Problem directly after loading"
    assert baybe_obj_reference.measured_parameters_comp.equals(
        baybe_obj_loaded.measured_parameters_comp
    ), "Problem directly after loading"
    assert baybe_obj_reference.measured_targets_comp.equals(
        baybe_obj_loaded.measured_targets_comp
    ), "Problem directly after loading"
    assert (
        baybe_obj_reference.batches_done == baybe_obj_loaded.batches_done
    ), "Problem directly after loading"
    assert (
        baybe_obj_reference.config.dict() == baybe_obj_loaded.config.dict()
    ), "Problem directly after loading"

    # Run some more iterations
    for _ in range(batch_quantity):
        rec = baybe_obj_reference.recommend(batch_quantity=batch_quantity)
        rec2 = baybe_obj_loaded.recommend(batch_quantity=batch_quantity)

        add_fake_results(
            rec,
            baybe_obj_reference,
            good_reference_values=good_reference_values,
        )
        rec2.Target_1 = rec.Target_1

        baybe_obj_reference.add_results(rec)
        baybe_obj_loaded.add_results(rec2)

    # Asserts
    assert baybe_obj_reference.searchspace.metadata.equals(
        baybe_obj_loaded.searchspace.metadata
    ), "Problem after continuation"
    assert baybe_obj_reference.searchspace.exp_rep.equals(
        baybe_obj_loaded.searchspace.exp_rep
    ), "Problem after continuation"
    assert baybe_obj_reference.searchspace.comp_rep.equals(
        baybe_obj_loaded.searchspace.comp_rep
    ), "Problem after continuation"
    assert baybe_obj_reference.measurements.equals(
        baybe_obj_loaded.measurements
    ), "Problem after continuation"
    assert baybe_obj_reference.measured_parameters_comp.equals(
        baybe_obj_loaded.measured_parameters_comp
    ), "Problem after continuation"
    assert baybe_obj_reference.measured_targets_comp.equals(
        baybe_obj_loaded.measured_targets_comp
    ), "Problem after continuation"
    assert (
        baybe_obj_reference.batches_done == baybe_obj_loaded.batches_done
    ), "Problem after continuation"
    assert (
        baybe_obj_reference.config.dict() == baybe_obj_loaded.config.dict()
    ), "Problem after continuation"

    os.remove("./test.baybe")
