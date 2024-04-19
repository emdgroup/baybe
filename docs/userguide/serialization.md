# Serialization

BayBE is shipped with a sophisticated serialization engine that allows to unstructure
its objects into basic types and seamlessly reassemble them afterward.
This enables a variety of advanced workflows, such as:
* Persisting objects for later use
* Transmission and processing outside the Python ecosystem
* Interaction with APIs and databases
* Writing "configuration" files

Some of the common workflows are demonstrated below.

```{admonition} Terminology
:class: important
* With **serialization**, we refer to the process of breaking down structured objects
(such as [Campaigns](baybe.campaign.Campaign)) into their most basic building blocks and
subsequently converting these blocks into a format usable outside the Python ecosystem.
* With **deserialization**, we accordingly refer to the inverse operation,
i.e. reassembling the corresponding Python object from its serialized format. 
* With **roundtrip**, we refer to the successive execution of both steps.
```


## JSON de-/serialization
All BayBE objects can be conveniently serialized into an equivalent JSON 
representation by calling their
{meth}`to_json <baybe.serialization.mixin.SerialMixin.to_json>` method.
The obtained JSON string can then be deserialized via the 
{meth}`from_json <baybe.serialization.mixin.SerialMixin.from_json>` method
of the corresponding class, which yields an "equivalent copy" of the original object.

```{admonition} Equivalent copies
:class: important
Roundtrip serialization is configured such that the obtained copy of an object
is semantically equivalent to its original version and thus behaves identically.
Note, however, that some objects contain ephemeral content (i.e. internal objects such
as temporary data or cached computation results) that may be lost during the roundtrip.
Because this content will be automatically recreated on the fly when needed,
it is ignored for equality comparison with `==`.
```

For example:
```python
from baybe.parameters import CategoricalParameter

parameter = CategoricalParameter(name="setting", values=["low", "high"])
json_string = parameter.to_json()
reconstructed = CategoricalParameter.from_json(json_string)
assert parameter == reconstructed
```

This form of roundtrip serialization can be used, for instance, to persist objects
for long-term storage, but also provides an easy way to "move" existing objects between
Python sessions by executing the deserializing step in a different context
than the serialization step.

