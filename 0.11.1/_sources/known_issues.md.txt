# PyCharm vs. `exceptiongroup`

BayBE's (de-)serialization machinery is build upon `cattrs`, which in turn relies on
`ExceptionGroup`s to report problems in a nicely structured format when using its
[detailed validation](https://catt.rs/en/stable/validation.html#detailed-validation)
feature. However, `ExceptionGroup`s were introduced in Python 3.11 and are
therefore not usable with earlier Python versions. To
enable the feature nevertheless, `cattrs` uses the [exceptiongroup
backport](https://pypi.org/project/exceptiongroup/), which enables the same
functionality by monkeypatching `TracebackException` and installing a special
exception hook on `sys.excepthook`.

## The Problem
The changes attempted by `exceptiongroup` will only be executed if **no prior
modifications have been made**. However, PyCharm appears to make similar modifications
for its own purposes, blocking those of `exceptiongroup` and thus preventing the
exceptions from being properly thrown in detailed validation mode.

## When to encounter
The chances of encountering this problem when interacting with BayBE are rather low
as the (de-)serialization objects are usually created by BayBE itself under normal
operation, so there is little risk of them being invalid in the first place. A
potential situation where you might run into the problem is if you manually
write a BayBE configuration and try to deserialize it into a Python BayBE object.
This can happen, for example, while engineering the configuration for later API
calls and testing it locally **using PyCharm**.

## How to avoid
If you encounter the problem, you can use **any** of the following workarounds to
circumvent the problem:
* Run the code from the terminal instead of inside PyCharm
* Change PyCharm's run configuration from "Run with Python Console" to "Emulate
  terminal in output console"
* Use Python version 3.11 or higher
* Undo the monkeypatch applied by PyCharm by running the following code **at the start
  of your script**:
```python
import sys
sys.excepthook = sys.__excepthook__
```
* Manually [format the exception](https://github.com/agronholm/exceptiongroup/blob/8b8791b662c0f62a574a09f305cd204dfb0a6a05/README.rst?plain=1) thrown by the problematic code:
```python
import exceptiongroup
from cattrs import ClassValidationError

try:
    <problematic code>
except ClassValidationError as e:
    raise ValueError("".join(exceptiongroup.format_exception(e)))
```
