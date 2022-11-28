# Tests
Various PyTest tests can be run in this folder.

### Fast Testing
Uses small iteration number, batch, etc. numbers and also has only one version of them. 
Is triggered by calling pytest with: 
```
pytest --fast
```
Fast testing is also triggered by a pre-commit hook.

### Extensive Testing
This will run several variants of iteration numbers, batch sizes etc and also generally 
use larger numbers for them. In addition it will run simulation tests that run a mock 
experiment with an expected outcome. Use:
```
pytest
```

### Test Options
If inspection of the test results is intended the following options are recommended:
```
pytest -v -p no:warnings
```
This will not collect warnings and show an overview of all tests that ran with thei 
result.

If only interested in a specific test it can be passed via the command line:
```
pytest -v -p no:warnings test_config.py
```

To show the slowest `n` of tests after testing use the option `--durations=n` and set 
`n` to 0 if all durations should be shown:
```
pytest --durations=5
```