"""Creating a BayBE object from a user configuration."""

from pathlib import Path

from baybe.core import BayBE

config_json = Path("config_example.json").read_text(encoding="utf-8")
baybe = BayBE.from_config(config_json)
print(baybe)
