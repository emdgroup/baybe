# Settings

BayBE provides a variety of settings that can be configured to change its behavior.
These settings can be adjusted using environment variables or by modifying the
{data}`baybe.settings.active_settings` object in various ways, as detailed
[below](#changing-settings).

## Available Settings

For more information on which settings are available, their roles and allowed value
types, have a look at the {class}`~baybe.settings.Settings` class documentation.

## Changing Settings

BayBE offers flexible control over its settings, allowing you to adjust them in
different ways, adapted to your specific needs and use cases.
 
### Direct Assignment
You can change any specific setting by directly assigning a new value to the
corresponding attribute of the {data}`~baybe.settings.active_settings` object. For
example, to set a value for {attr}`~baybe.settings.Settings.random_seed`,
simply run:

```python
from baybe import active_settings

active_settings.random_seed = 1337
```

````{admonition} Validation
:class: note

To avoid silent bugs, BayBE automatically validates if the referenced setting
attribute exists and if the assigned value is compatible:

```python
import pytest
from baybe import active_settings

with pytest.raises(AttributeError):
    active_settings.non_existent_setting = 1337  # <-- error!

with pytest.raises(TypeError):
    active_settings.preprocess_dataframes = "not_representing_a_boolean"  # <-- error!
```
````

### Joint Activation
While you can change several settings one at a time using direct assignment, a more
convenient way is to instantiate a {class}`baybe.settings.Settings` object, which allows
to define multiple settings at once. In order for the settings to take effect,
call its :meth:`~baybe.settings.Settings.activate` method:

```python
from baybe import Settings, active_settings

assert active_settings.parallelize_simulations is True
assert active_settings.use_polars_for_constraints is True

Settings(parallelize_simulations=False, use_polars_for_constraints=False).activate()

assert active_settings.parallelize_simulations is False
assert active_settings.use_polars_for_constraints is False
```

### Delayed Activation
Adjusting settings via the {class}`~baybe.settings.Settings` class has the additional
benefit that it allows you to delay the activation of a particular settings
configuration to a later point, giving you the possibility to store and organize several
configurations in your code. For example:

```python
from baybe import Settings

slow_and_pedantic = Settings(preprocess_dataframes=True, use_fpsample=False)
fast_and_furious = Settings(preprocess_dataframes=False, use_fpsample=True)
```

You can then active these configurations in various places and in different ways:

#### Manual Activation
To *manually* activate a particular settings configuration, use its
{meth}`~baybe.settings.Settings.activate` method. The previously active settings will be
automatically remembered and can easily be restored at a later point using the
corresponding {meth}`~baybe.settings.Settings.restore_previous` method:

```python
assert active_settings.preprocess_dataframes is True
fast_and_furious.activate()
assert active_settings.preprocess_dataframes is False
fast_and_furious.restore_previous()
assert active_settings.preprocess_dataframes is True
```

```{admonition} Restoring Previous Settings
:class: important

Note that {meth}`~baybe.settings.Settings.restore_previous` restores the settings that
were active **at the time of the {meth}`~baybe.settings.Settings.activate` call**, not
the previously global active settings. The latter might potentially have changed in the
meantime, depending on execution flow:

```python
from baybe import Settings, active_settings

s_0 = Settings(random_seed=0).activate()
assert active_settings.random_seed == 0

s_42 = Settings(random_seed=42).activate()
assert active_settings.random_seed == 42

s_1337 = Settings(random_seed=1337).activate()
assert active_settings.random_seed == 1337

# At this point, the active seed is 1337, and the previous active seed was 42.
# However, the effect of "restoring settings" crucially depends on which object is used:

s_42.restore_previous()
assert active_settings.random_seed == 0  # <-- the seed before s_42 got activated

s_1337.restore_previous()
assert active_settings.random_seed == 42  # <-- the seed before s_42 got activated
```

#### Context Activation
Using {meth}`~baybe.settings.Settings.restore_previous` can be useful in special cases
where settings objects need to be passed around. However, in most cases where settings
should be activated *temporarily* within a specific scope, a more convenient approach is
to use a context manager:

```python
assert active_settings.preprocess_dataframes is True

# Within the context, the specified settings become active
with fast_and_furious:
    assert active_settings.preprocess_dataframes is False

# Outside the context, the previous settings are restored
assert active_settings.preprocess_dataframes is True
```

#### Decorator Activation
Finally, {class}`~baybe.settings.Settings` objects can also be used to decorate
callables, activating the corresponding configuration for the duration of the call:

```python
assert active_settings.preprocess_dataframes is True


@fast_and_furious
def validate_dataframes_carefully():
    assert active_settings.preprocess_dataframes is False


validate_dataframes_carefully()  # <-- the assert passes

assert active_settings.preprocess_dataframes is True
```

### Environment Variables
Settings can also be controlled via environment variables, which is particularly useful
for deploying applications in different environments without having to change any
underlying code. 

Each individual setting attribute has a corresponding environment variable, written in
uppercase letters and prefixed with `BAYBE_`. For example, the
{attr}`~baybe.settings.Settings.preprocess_dataframes` attribute is linked to the
`BAYBE_PREPROCESS_DATAFRAMES` environment variable.

When present, these variables can be used to populate settings attributes instead of
falling back to defaults, which is controlled via the `restore_environment` flag. For
further details on how they interact with other value sources and how they affect the
initialization of the {data}`~baybe.settings.active_settings` at startup time, have a
look at our [initialization precedence](#initialization-precedence) section.


## Inspecting Settings
To inspect either the current active settings or any particular settings object,
you can simply print the corresponding object:

```python
from baybe import active_settings, Settings

# Inspect current settings
print(active_settings)

# Inspect a specific configuration object
print(Settings(preprocess_dataframes=True, use_polars_for_constraints=True))
```

## Initialization Precedence
Initializing a new {class}`~baybe.settings.Settings` object follows a specific order of
precedence. Specifically, the value of each attribute is determined by whichever of the
following rules applies first:

1. If a value is passed explicitly to the {class}`~baybe.settings.Settings` constructor,
   it always takes the highest precedence.
2. If `restore_environment=True` is passed, the value of the setting's corresponding
   environment variable is used, provided it exists.
3. If `restore_defaults=True` is passed, the default value for the attribute defined
   by the class is used.
4. If none of the above applies, the current value from
   {data}`~baybe.settings.active_settings` is retained.

```{admonition} Global Settings Initialization
:class: important

For convenience, `restore_environment` is set to `True` when initializing the global
{data}`~baybe.settings.active_settings` object at package import time, so that
environment variables automatically take effect.
```


