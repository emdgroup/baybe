# Tests
Various `pytest` tests can be run in this folder.

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