{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:                              
   :special-members: __call__
   :show-inheritance:                 
   :inherited-members:

   {% block methods %}
   {% if methods %}
   .. rubric:: {{ _('Public methods') }}

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {% if item == "__init__" and "__call__" in all_methods %}
      ~{{ name }}.__call__
   {% endif %}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Public attributes and properties') }}

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}