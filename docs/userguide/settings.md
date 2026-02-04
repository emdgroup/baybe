# Settings

BayBE provides a variety of settings that can be configured to change its behavior.
These settings can be adjusted using environment variables or by modifying the
{data}`baybe.settings.active_settings` object in various ways, as detailed
[below](#changing-settings).

## Available Settings

The following settings are available:

| Setting | Description |
|--------:|-------------|
| {attr}`~baybe.settings.Settings.cache_campaign_recommendations` | Controls if {class}`~baybe.campaign.Campaign` objects cache their latest set of recommendations. |
| {attr}`~baybe.settings.Settings.cache_directory` | The directory used for persistent caching on disk. Set to ``""`` or ``None`` to disable caching. |
| {attr}`~baybe.settings.Settings.parallelize_simulation_runs` | Controls if simulation runs with [`xyzpy`](https://xyzpy.readthedocs.io/) are executed in parallel. |
| {attr}`~baybe.settings.Settings.preprocess_dataframes` | Controls if incoming user dataframes are preprocessed (i.e., dtype-converted and validated) before use. |
| {attr}`~baybe.settings.Settings.random_seed` | The used random seed. |
| {class}`use_fpsample <baybe.settings.Settings>` | Controls if [`fpsample`](https://github.com/leonardodalinky/fpsample) acceleration is to be used, if available. |
| {class}`use_polars_for_constraints <baybe.settings.Settings>` | Controls if [`polars`](https://pola.rs/) acceleration is to be used for discrete constraints, if available. |
| {attr}`~baybe.settings.Settings.use_single_precision_numpy` | Controls the floating point precision used for [`numpy`](https://numpy.org/) arrays. |
| {attr}`~baybe.settings.Settings.use_single_precision_torch` | Controls the floating point precision used for [`torch`](https://pytorch.org/) tensors. |

For more information, click on the respective link or have a look at the
{class}`~baybe.settings.Settings` class documentation.

## Changing Settings

BayBE offers flexible control over its settings, allowing you to adjust them in
different ways, adapted to your specific needs and use cases. If you want to understand
the underlying mechanism in detail, have a look [here](#activation-logic). Otherwise,
simply choose the most suitable option from the following alternatives:
 
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

~~~python
from baybe import active_settings

active_settings.non_existent_setting = 1337  # <-- error!
active_settings.preprocess_dataframes = "not_representing_a_boolean"  # <-- error!
~~~
````

### Joint Activation
While you can change several settings one at a time using direct assignment, a more
convenient way is to instantiate a {class}`baybe.settings.Settings` object, which allows
to define multiple settings at once. In order for the settings to take effect,
call its {meth}`~baybe.settings.Settings.activate` method:

```python
from baybe import Settings, active_settings

assert active_settings.parallelize_simulation_runs is True
assert active_settings.use_polars_for_constraints is True

Settings(parallelize_simulation_runs=False, use_polars_for_constraints=False).activate()

assert active_settings.parallelize_simulation_runs is False
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

(RESTORING_SETTINGS)=
```{admonition} Restoring Previous Settings
:class: important

Note that {meth}`~baybe.settings.Settings.restore_previous` restores the settings that
were active **at the time of the {meth}`~baybe.settings.Settings.activate` call**, not
the previously active settings. The latter might potentially have changed in the
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
def regret_nothing():
    assert active_settings.preprocess_dataframes is False


regret_nothing()  # <-- the assert passes

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


## Activation Logic
The activation of settings follows two simple rules that allow you to understand the
implications of modifying {class}`~baybe.settings.Settings` objects:
1. When executing code, BayBE always reads the current value of settings from the
   {data}`~baybe.settings.active_settings` object, meaning that any changes to this
   object immediately take effect.
2. An {meth}`~baybe.settings.Settings.activate` call on a
   {class}`~baybe.settings.Settings` object copies its attribute values to those of the
   {data}`~baybe.settings.active_settings` object and stores a copy of the
   previously active values in the object on which the call was made. 

This explains, for instance, why:
* Direct assignments to attributes of {data}`~baybe.settings.active_settings` take
  immediate global effect.
* Calling {meth}`~baybe.settings.Settings.activate` on a
  {class}`~baybe.settings.Settings` object is equivalent to [directly
  assigning](#direct-assignment) all of its attribute values to the
  {data}`~baybe.settings.active_settings` object one by one.
* Adjusting the attribute of any {class}`~baybe.settings.Settings` object other than
  {data}`~baybe.settings.active_settings` has no immediate effect on the 
  active settings until its {meth}`~baybe.settings.Settings.activate` method is called,
  no matter if it has been activated before or not.
* You can [restore](#restoring_settings) the settings that were active previous to an 
  {meth}`~baybe.settings.Settings.activate` call using the
  {meth}`~baybe.settings.Settings.restore_previous` method. 


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


```{admonition} Exception: Random Seed Management
:class: attention

A notable exception to the above rules applies to managing random seeds, for reasons
detailed in the [Random Seed Control](#random-seed-control) section.
```

```{admonition} Active Settings Initialization
:class: important

For convenience, `restore_environment` is set to `True` when initializing the active
{data}`~baybe.settings.active_settings` object at package import time, so that
environment variables automatically take effect.
```

## Random Seed Control
Unlike other BayBE setting attributes, whose values remain static until changed
explicitly, the states of random number generators (RNG) naturally evolve as random
numbers are drawn during code execution. Applying the same manipulation rules to
{attr}`~baybe.settings.Settings.random_seed` as we do to other settings attributes would
therefore lead to rather unexpected behavior from a user's perspective, especially since
the seed value is only used to initialize the RNG states and remains unchanged while the
latter progress.

To align RNG control with user expectations, random seed mechanics thus
slightly deviate from the otherwise general rules outlined in this user guide:
- In contrast to what is dictated by the [initialization
  precedence](#initialization-precedence), specifying a random seed via the
  `BAYBE_RANDOM_SEED` environment variable **only** affects the initialization of the
  active {data}`~baybe.settings.active_settings` object at session start, but has no
  effect on the instantiation of subsequent {class}`~baybe.settings.Settings` objects.
- Likewise, {class}`~baybe.settings.Settings` objects created by the user do not adopt
  the random seed from the {data}`~baybe.settings.active_settings` object, which
  avoids unintended RNG state resets when activating such objects.
- When [restoring](#restoring_settings) previous settings, the RNG states are only
  reverted if the {class}`~baybe.settings.Settings` object in question explicitly
  specified a random seed during its activation (i.e., the RNG state was deliberately
  controlled). If no seed was specified, the RNG state remains untouched.

