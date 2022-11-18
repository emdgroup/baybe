"""
Test for save/load capability
"""

from baybe.core import BayBE, BayBEConfig
from baybe.utils import add_fake_results, add_parameter_noise


substances = {
    "Water": "O",
    "THF": "C1CCOC1",
    "DMF": "CN(C)C=O",
    "Hexane": "CCCCCC",
    "Ethanol": "CCO",
}

config_dict = {
    "project_name": "Save/Load",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": False,
    "parameters": [
        {
            "name": "Categorical_1",
            "type": "CAT",
            "values": ["very bad", "bad", "OK", "good", "very good"],
            "encoding": "INT",
        },
        {
            "name": "Num_disc_1",
            "type": "NUM_DISCRETE",
            "values": [1, 2, 3, 4],
            "tolerance": 0.3,
        },
        {
            "name": "Substance_1",
            "type": "SUBSTANCE",
            "data": substances,
            "encoding": "MORDRED",
            "decorrelate": 0.9,
        },
    ],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "Target_1",
                "type": "NUM",
                "mode": "MAX",
            },
        ],
    },
    "strategy": {
        # "surrogate_model_cls": "GP",
        # "recommender_cls": "RANDOM"
    },
}

# Define some parameter values to define rows where the fake results should be good
good_reference_values = {"Num_disc_1": [2], "Categorical_1": ["OK"]}

# Create BayBE object, add fake results and print what happens to internal data
config = BayBEConfig(**config_dict)
baybe_obj = BayBE(config)

N_ITERATIONS = 3
BATCH_QUANTITY = 3

for kIter in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1} #####")

    rec = baybe_obj.recommend(batch_quantity=BATCH_QUANTITY)

    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    add_parameter_noise(rec, baybe_obj, noise_level=0.1)
    print("### Recommended dataframe with fake results and noise:\n", rec)

    baybe_obj.add_results(rec)
# print(baybe_obj)

print("\n\n\n############################# BREAK #############################")

# Store baybe object
baybe_obj.save("./test.baybe")

# Restore baybe object
baybe_obj2 = BayBE.from_stored("./test.baybe")

# Asserts
print("Asserts checking data consistency after loading (must all be true):")
print("1 - ", baybe_obj.searchspace_metadata.equals(baybe_obj2.searchspace_metadata))
print("2 - ", baybe_obj.searchspace_exp_rep.equals(baybe_obj2.searchspace_exp_rep))
print("3 - ", baybe_obj.searchspace_comp_rep.equals(baybe_obj2.searchspace_comp_rep))
print("4 - ", baybe_obj.measurements.equals(baybe_obj2.measurements))
print(
    "5 - ",
    baybe_obj.measured_parameters_comp.equals(baybe_obj2.measured_parameters_comp),
)
print("6 - ", baybe_obj.measured_targets_comp.equals(baybe_obj2.measured_targets_comp))
print("7 - ", baybe_obj.batches_done == baybe_obj2.batches_done)
print("8 - ", baybe_obj.config.dict() == baybe_obj2.config.dict())

# Run some more iterations
for kIter2 in range(N_ITERATIONS):
    print(f"\n\n##### ITERATION {kIter+1+kIter2+1} #####")

    rec = baybe_obj.recommend(batch_quantity=BATCH_QUANTITY)
    rec2 = baybe_obj2.recommend(batch_quantity=BATCH_QUANTITY)

    add_fake_results(rec, baybe_obj, good_reference_values=good_reference_values)
    rec2.Target_1 = rec.Target_1
    print("### Recommended dataframe based on original:\n", rec)
    print("### Recommended dataframe based on loaded:\n", rec2)

    baybe_obj.add_results(rec)
    baybe_obj2.add_results(rec2)


print("\n\nMeasurements of loaded after everything ran:")
print(baybe_obj2.measurements)

print("\n\nAsserts checking final  (must all be true):")
print("1 - ", baybe_obj.searchspace_metadata.equals(baybe_obj2.searchspace_metadata))
print("2 - ", baybe_obj.searchspace_exp_rep.equals(baybe_obj2.searchspace_exp_rep))
print("3 - ", baybe_obj.searchspace_comp_rep.equals(baybe_obj2.searchspace_comp_rep))
print("4 - ", baybe_obj.measurements.equals(baybe_obj2.measurements))
print(
    "5 - ",
    baybe_obj.measured_parameters_comp.equals(baybe_obj2.measured_parameters_comp),
)
print("6 - ", baybe_obj.measured_targets_comp.equals(baybe_obj2.measured_targets_comp))
print("7 - ", baybe_obj.batches_done == baybe_obj2.batches_done)
print("8 - ", baybe_obj.config.dict() == baybe_obj2.config.dict())
