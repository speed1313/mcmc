import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np


n = 10000
estimate_pi_history = []
key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, n+1)
a = 10 # Change to 10000 to see the approximation error
s = 0
for i in range(n):
    x = jax.random.uniform(subkeys[i], minval=-a, maxval=a)
    y = (1/jnp.sqrt(2*jnp.pi)) * jnp.exp(-x**2/2)
    s += y
    estimate_pi_history.append(s / (i+1)*2*a)

plt.plot(estimate_pi_history)
true = 1
plt.axhline(true, color='r')
plt.savefig('gauss.png')
plt.show()


