"""Creating a BayBE object from a user configuration."""

from pathlib import Path
from time import perf_counter

from baybe.core import BayBE

config_json = Path("config_example.json").read_text(encoding="utf-8")

# Full object creation
t = perf_counter()
baybe = BayBE.from_config(config_json)
print("Duration of object creation", perf_counter() - t)

# Config validation only
t = perf_counter()
baybe = BayBE.validate_config(config_json)
print("Duration of config validation", perf_counter() - t)
