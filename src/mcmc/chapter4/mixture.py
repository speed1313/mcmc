import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

def S(x):
    return -jnp.log(jnp.exp(-0.5*(x-3)**2) + jnp.exp(-0.5*(x+3)**2))


n = int(1e5)
estimate_pi_history = []
key = jax.random.PRNGKey(1)
key, *subkeys = jax.random.split(key, 2*n)
x = 0
step_size = 5
n_accept = 0
x_list = []
for i in range(n-1):
    backup_x = x
    action_init = S(x)
    dx = jax.random.uniform(subkeys[2*i])
    dx = (dx - 0.5) * step_size * 2
    x = x + dx
    action_fin = S(x)
    metropolis = jax.random.uniform(subkeys[2*i+1])
    if jnp.exp(action_init - action_fin) > metropolis:
        n_accept += 1
    else:
        x = backup_x
    x_list.append(x)


file_name = __file__.split('/')[-1].replace('.py', '')

plt.plot(x_list, label='x')
plt.xlabel('n')
plt.ylabel('x')
plt.savefig(file_name + '_x_history.png')
plt.show()


plt.hist(x_list, bins=100,density=True, alpha=0.5, label='x')
x_data = jnp.linspace(-7, 7, 100)
y_data = jnp.exp(-S(x_data)) / (2*jnp.sqrt(2 * jnp.pi))
plt.plot(x_data, y_data, label='True', color='r')


plt.savefig(f'{file_name}.png')
plt.show()

print('<x>', np.mean(x_list))
print('<x^2>', np.mean(np.array(x_list)**2))
