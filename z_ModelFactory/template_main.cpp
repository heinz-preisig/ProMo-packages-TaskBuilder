# Automatically generated, do not edit!

"""
What:    Python simulation
Author:  Andreas
Contact: arne.t.elve(at)ntnu.no
Date:    {{date}}
Model:   {{model_name}}
"""

# Import packages:
import numpy as np                                   # Numerical python library
from scipy.integrate import ode                           # Integrator in scipy
import matplotlib.pyplot as plt                             # Data illustration
from python_simulation import IndexSet, khatriRao, blockReduce   # Custom funcs

# =================================== BODY ================================== #

# INDEX SETS:
{% for index_set in indices %}
{{ index_set }}
{% endfor %}

# INITIAL STATE:
{% for state in state_variables %}
{{ state }}
{% endfor %}

# FRAMES:
{% for frame in frames %}
{{ frame }}
{% endfor %}

# CONSTANTS:
{% for var in variables %}
{{ var }}
{% endfor %}

# AUTOMATICALLY GENERATED STUFF:
# Equation selections:
{% for sel in equation_selections %}
{{ sel }}
{% endfor %}

# Network information:
{% for network in networks %}
{{ network }}
{% endfor %}

# Constant equations:
{% for eq in constant_equations %}
{{ eq }}
{% endfor %}


# INTEGRATING FUNCTION:
def derivative(t, {{ state }}):
  """
  t: time
  {{ state }}: state

  Integrating function:
  dxdt = derivative(t, {{ state }})

  Note: sequence of variables depends on integrator. Using scipy's ode requires
  time before state, while odeint needs state before time.  Also the integrator
  want  flat vectors,  no column vectors.  I, therefore, transpose  the vectors
  manually. Normal transpose is not sufficient.
  """
  # HACK: Integrator give flat state vectors while our model is column vector
  {{ state }} = [[{{ state }}i] for {{state}}i in {{ state }}]

  # EQUATIONS:
  {% for eq in equations -%}
  {{ eq }}
  {% endfor -%}

  return {{ derivative }}


# INTEGRATOR
data = {}
data['{{ state }}'] = []
data['t'] = []
r = ode(derivative).set_integrator('dop853')
r.set_initial_value({{ state }}, {{ frame_start }})
dt = 0.1
while r.successful() and r.t < {{ frame_end }}:
  {{ state }} = r.integrate(r.t+dt)
  data['{{ state }}'].append({{ state }})
  data['t'].append(r.t)

data['{{ state }}'] = np.transpose(data['{{ state }}'])[0]
for i, {{ state }}i in enumerate(data['{{ state }}']):
  plt.plot(data['t'], {{ state }}i, label = i)
plt.legend()
plt.show()
