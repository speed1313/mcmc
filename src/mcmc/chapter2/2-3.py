import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np


n = 10000
estimate_pi_history = []
key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 2*n)
n_in = 1e-10
z_sum = 0
for i in range(n):
    x = jax.random.uniform(subkeys[i])
    y = jax.random.uniform(subkeys[i+1])
    if x**2 + y**2 < 1:
        n_in += 1
        z = jnp.sqrt(1 - x**2 - y**2)
        z_sum += z
    estimate_pi_history.append((z_sum / n_in) * 2 * jnp.pi)

plt.plot(estimate_pi_history)
true = 4 * jnp.pi / 3
plt.axhline(true, color='r')
file_name = __file__.split('/')[-1].replace('.py', '')
plt.savefig(f'{file_name}.png')
plt.show()

