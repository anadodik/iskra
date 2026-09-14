{{ ("``" + fullname + "``") | underline }}

.. automodule:: {{ fullname }}

{% if classes %}
.. autosummary::
   :toctree: generated

{% for item in classes %}
   {{ item }}
{%- endfor %}
{% endif %}

{% if functions %}
.. autosummary::
   :toctree: generated

{% for item in functions %}
   {{ item }}
{%- endfor %}
{% endif %}
