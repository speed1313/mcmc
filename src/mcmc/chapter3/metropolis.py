import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np


n = int(1e5)
estimate_pi_history = []
key = jax.random.PRNGKey(1)
key, *subkeys = jax.random.split(key, 2*n)
x = 0
step_size = 1
n_accept = 0
x_list = []
for i in range(n-1):
    backup_x = x
    action_init = 0.5 * x * x
    dx = jax.random.uniform(subkeys[2*i])
    dx = (dx - 0.5) * step_size * 2
    x = x + dx
    action_fin = 0.5 * x * x
    metropolis = jax.random.uniform(subkeys[2*i+1])
    if jnp.exp(action_init - action_fin) > metropolis:
        n_accept += 1
    else:
        x = backup_x
    x_list.append(x)

plt.hist(x_list, bins=100, alpha=0.5, label='x')

file_name = __file__.split('/')[-1].replace('.py', '')
plt.savefig(f'{file_name}.png')
plt.show()

print('<x>', np.mean(x_list))
print('<x^2>', np.mean(np.array(x_list)**2))
