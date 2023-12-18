{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if not fullname in ("Smiles") %}
.. auto{{ objtype }}:: {{ objname }}
{% endif %}
