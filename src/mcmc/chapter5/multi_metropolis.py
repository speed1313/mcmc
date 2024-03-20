import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

def S(x, y):
    return 0.5 * (x**2 + y**2 + x * y)

n = int(1e4)
estimate_pi_history = []
key = jax.random.PRNGKey(1)
key, *subkeys = jax.random.split(key, 3*n+1)
x = 0
y = 0
step_size = 1
n_accept = 0
x_list = []
y_list = []
for i in range(n):
    backup_x = x
    backup_y = y
    action_init = S(x, y)

    dx = jax.random.uniform(subkeys[3*i])
    dy = jax.random.uniform(subkeys[3*i+1])
    dx = (dx - 0.5) * step_size * 2
    dy = (dy - 0.5) * step_size * 2
    x = x + dx
    y = y + dy
    action_fin = S(x, y)
    metropolis = jax.random.uniform(subkeys[3*i+2])
    if jnp.exp(action_init - action_fin) > metropolis:
        n_accept += 1
    else:
        x = backup_x
        y = backup_y
    x_list.append(x)
    y_list.append(y)

plt.scatter(x_list, y_list)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('xy_history.png')
plt.show()


plt.hist(x_list, bins=100,density=True, alpha=0.5, label='x')
plt.hist(y_list, bins=100,density=True, alpha=0.5, label='y')


file_name = __file__.split('/')[-1].replace('.py', '')
plt.savefig(f'{file_name}.png')
plt.show()

print('<x>', np.mean(x_list))
print('<x^2>', np.mean(np.array(x_list)**2))
print('<y>', np.mean(y_list))
print('<y^2>', np.mean(np.array(y_list)**2))
