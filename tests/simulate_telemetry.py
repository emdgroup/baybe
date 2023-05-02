"""
Simulate different users and telemetry settings. This script does some calls so that
the results can be viewed on AWS CloudWatch.
"""

import os
from random import randint

from baybe.core import BayBE, BayBEConfig
from baybe.telemetry import get_user_hash
from baybe.utils import add_fake_results

dict_solvent = {
    "DMAc": r"CC(N(C)C)=O",
    "Butyornitrile": r"CCCC#N",
    "Butyl Ester": r"CCCCOC(C)=O",
    "p-Xylene": r"CC1=CC=C(C)C=C1",
}

dict_base = {
    "Potassium acetate": r"O=C([O-])C.[K+]",
    "Potassium pivalate": r"O=C([O-])C(C)(C)C.[K+]",
    "Cesium acetate": r"O=C([O-])C.[Cs+]",
    "Cesium pivalate": r"O=C([O-])C(C)(C)C.[Cs+]",
}

dict_ligand = {
    "BrettPhos": r"CC(C)C1=CC(C(C)C)=C(C(C(C)C)=C1)C2=C(P(C3CCCCC3)C4CCCCC4)C(OC)="
    "CC=C2OC",
    "Di-tert-butylphenylphosphine": r"CC(C)(C)P(C1=CC=CC=C1)C(C)(C)C",
    "(t-Bu)PhCPhos": r"CN(C)C1=CC=CC(N(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C3=CC=CC=C3",
    "Tricyclohexylphosphine": r"P(C1CCCCC1)(C2CCCCC2)C3CCCCC3",
    "PPh3": r"P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3",
    "XPhos": r"CC(C1=C(C2=CC=CC=C2P(C3CCCCC3)C4CCCCC4)C(C(C)C)=CC(C(C)C)=C1)C",
    "P(2-furyl)3": r"P(C1=CC=CO1)(C2=CC=CO2)C3=CC=CO3",
    "Methyldiphenylphosphine": r"CP(C1=CC=CC=C1)C2=CC=CC=C2",
    "1268824-69-6": r"CC(OC1=C(P(C2CCCCC2)C3CCCCC3)C(OC(C)C)=CC=C1)C",
    "JackiePhos": r"FC(F)(F)C1=CC(P(C2=C(C3=C(C(C)C)C=C(C(C)C)C=C3C(C)C)C(OC)=CC=C2OC)"
    r"C4=CC(C(F)(F)F)=CC(C(F)(F)F)=C4)=CC(C(F)(F)F)=C1",
    "SCHEMBL15068049": r"C[C@]1(O2)O[C@](C[C@]2(C)P3C4=CC=CC=C4)(C)O[C@]3(C)C1",
    "Me2PPh": r"CP(C)C1=CC=CC=C1",
}

cofig_dict = {
    "project_name": "Direct Arylation",
    "allow_repeated_recommendations": False,
    "allow_recommending_already_measured": False,
    "numerical_measurements_must_be_within_tolerance": True,
    "parameters": [
        {
            "name": "Solvent",
            "type": "SUBSTANCE",
            "data": dict_solvent,
            "encoding": "MORDRED",
        },
        {
            "name": "Base",
            "type": "SUBSTANCE",
            "data": dict_base,
            "encoding": "MORDRED",
        },
        {
            "name": "Ligand",
            "type": "SUBSTANCE",
            "data": dict_ligand,
            "encoding": "MORDRED",
        },
        {
            "name": "Temp_C",
            "type": "NUM_DISCRETE",
            "values": [90, 105, 120],
            "tolerance": 2,
        },
        {
            "name": "Concentration",
            "type": "NUM_DISCRETE",
            "values": [0.057, 0.1, 0.153],
            "tolerance": 0.005,
        },
    ],
    "objective": {
        "mode": "SINGLE",
        "targets": [
            {
                "name": "yield",
                "type": "NUM",
                "mode": "MAX",
            },
        ],
    },
}

config = BayBEConfig(**cofig_dict)


# Actual User
print(f"Actual User: {get_user_hash()}")
baybe_object = BayBE(config)
for k in range(randint(4, 6)):
    dat = baybe_object.recommend(randint(2, 3))
    add_fake_results(dat, baybe_object)
    baybe_object.add_results(dat)

# Fake User1 - 5 iterations
print("Fake User1")
os.environ["BAYBE_DEBUG_FAKE_USERHASH"] = "FAKE_USER_1"
baybe_object = BayBE(config)
for k in range(randint(2, 3)):
    dat = baybe_object.recommend(randint(3, 4))
    add_fake_results(dat, baybe_object)
    baybe_object.add_results(dat)

# Fake User2 - 2 iterations
print("Fake User2")
os.environ["BAYBE_DEBUG_FAKE_USERHASH"] = "FAKE_USER_2"
baybe_object = BayBE(config)
for k in range(randint(2, 3)):
    dat = baybe_object.recommend(randint(4, 5))
    add_fake_results(dat, baybe_object)
    baybe_object.add_results(dat)

# Fake User3 - no telemetry
print("Fake User3")
os.environ["BAYBE_DEBUG_FAKE_USERHASH"] = "FAKE_USER_3"
os.environ["BAYBE_TELEMETRY_ENABLED"] = "false"
baybe_object = BayBE(config)
for k in range(randint(5, 7)):
    dat = baybe_object.recommend(randint(2, 3))
    add_fake_results(dat, baybe_object)
    baybe_object.add_results(dat)

# Cleanup
os.environ.pop("BAYBE_DEBUG_FAKE_USERHASH")
os.environ.pop("BAYBE_TELEMETRY_ENABLED")
