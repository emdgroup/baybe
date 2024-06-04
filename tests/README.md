# Tests
Various `pytest` tests can be run in this folder.

## PyTest
### Fast Testing
Uses small iteration number, batch size, etc., with only one variant for each.
Can be triggered as follows: 
```
pytest --fast
```

### Extensive Testing
Runs several variants of iteration numbers, batch sizes, etc., and also generally 
uses larger numbers for each. Can be triggered as follows:
```
pytest
```

### Test Options
If inspection of the test results is intended, the following options are recommended:
```
pytest -v -p no:warnings
```
This will not collect warnings and show an overview of all executed tests, together 
with their results.

If only interested in a specific test, it can be passed via the command line:
```
pytest -v -p no:warnings test_config.py
```

To show the slowest `n` tests after testing, use the option `--durations=n`. Set 
`n` to 0 if all durations should be shown:
```
pytest --durations=5
```

To get an assessment of the code coverage you can specify the following option:
```
pytest --cov=baybe
```

This will produce something like this:
```
Name                                 Stmts   Miss  Cover
--------------------------------------------------------
baybe/acquisition.py                    58      0   100%
baybe/constraints.py                   170     10    94%
baybe/campaign.py                      111      8    93%
...
--------------------------------------------------------
TOTAL                                 1941    214    89%
```

## Tox
Testing, linting and auditing can also be done via `tox`, which includes the 
possibility to test different python variants as well. 

### Environments
In `tox.ini`, we have configured several environments for running different actions 
(`fulltest`, `coretest`, `lint`, `audit`) against different versions of python (e.g. `py310`, `py311`, .
..). 
You can specify both in `tox` to call a certain combination. 

For instance 
```bash
tox -e fulltest-py310
``` 
will run pytest on baybe with all optional features in python 3.10, while 
```bash
tox -e coretest-py312
```
will run pytest on baybe without additional features in python 3.12.
```bash
tox -e lint-py312
```
will run the linters with python 3.12.

For a full overview of all available environments, type:
```
tox -l
```

### Shortcuts
In case you want to run several combinations, you can specify them like
```bash
tox -e audit-py310,audit-py311
```

If you omit the python version from the environment, `tox` will use the version 
from the command-executing environment:
```bash
tox -e coretest  # runs like '-e coretest-py310' if called from a python 3.10 environment
```

If you simply want to run all combinations, you can use
```bash
tox  # runs all environments shown via `tox -l`
```

### Local / Parallel Execution
On a local machine, the sequential execution of `tox` might take a long time. 
Thus, you can use the parallel option `-p`:
```bash
tox -p
```
which will run all environments in parallel.