import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np


n = 1000
estimate_pi_history = []
key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, n+1)
s = 0
for i in range(n):
    x = jax.random.uniform(subkeys[i])
    y = jnp.sqrt(1 - x**2)
    s += y
    estimate_pi_history.append(s / (i+1))

plt.plot(estimate_pi_history)
true = np.pi / 4
plt.axhline(true, color='r')
plt.savefig('pi_integral.png')
plt.show()


