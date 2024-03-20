import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np


n = 10000
key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 2*n)
n_in = 1e-10
z_sum = 0
z1_list = []
z2_list = []
for i in range(n):
    x = jax.random.uniform(subkeys[i])
    y = jax.random.uniform(subkeys[i+1])
    z1 = jnp.sqrt(-2 * jnp.log(x)) * jnp.cos(2 * jnp.pi * y)
    z2 = jnp.sqrt(-2 * jnp.log(x)) * jnp.sin(2 * jnp.pi * y)
    z1_list.append(z1)
    z2_list.append(z2)

plt.hist(z1_list, bins=100, alpha=0.5, label='z1')
#plt.hist(z2_list, bins=100, alpha=0.5, label='z2')
file_name = __file__.split('/')[-1].replace('.py', '')
plt.savefig(f'{file_name}.png')
plt.show()


print('<z1>', np.mean(z1_list))
print('<z1^2>', np.mean(np.array(z1_list)**2))


