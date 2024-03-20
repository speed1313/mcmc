import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

def gauss(x):
    return (1/jnp.sqrt(2*jnp.pi)) * jnp.exp(-x**2/2)

n = 100000
key = jax.random.PRNGKey(0)
key, *subkeys = jax.random.split(key, 2*n)
s = 0
max_gauss = max([gauss(x) for x in range(-10, 10)])
x_list = []
for i in range(n):
    x = jax.random.uniform(subkeys[i], minval=-5, maxval=5)
    y = jax.random.uniform(subkeys[i+1], minval=0, maxval=max_gauss)
    if y < gauss(x):
        x_list.append(x)

plt.hist(x_list, bins=100, alpha=0.5, density = True, label='x', color='b')
true_x = np.linspace(-5, 5, 100)
true_y = [gauss(x) for x in true_x]
plt.plot(true_x, true_y, label='True', color='r')
plt.savefig('reject.png')
plt.show()


