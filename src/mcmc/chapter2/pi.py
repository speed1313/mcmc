import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np


n = 10000
estimate_pi_history = []
key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 2*n)
s = 0
for i in range(n):
    x = jax.random.uniform(subkeys[i])
    y = jax.random.uniform(subkeys[i+1])
    if x**2 + y**2 < 1:
        s += 1
    estimate_pi_history.append(s / (i+1))

plt.plot(estimate_pi_history)
true = np.pi / 4
plt.axhline(true, color='r')
plt.savefig('pi.png')
plt.show()


