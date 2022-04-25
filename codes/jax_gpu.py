
import jax
import jax.numpy as jnp


def my_dot(a, b):
    return jnp.dot(a, b)


try:
    print("available GPU  : ")
    print(jax.devices('gpu'))
except:
    print('no GPU found, available device : ')
    print(jax.devices())

matrix_vector = jax.vmap(my_dot, in_axes=(0, None))
matrix_matrix = jax.vmap(my_dot, in_axes=(0, None))
matrix_matrix = jax.jit(matrix_matrix)

rng = jax.random.PRNGKey(0)
k1, k2 = jax.random.split(rng, 2)

dim = 100
m1 = jax.random.normal(k1, (dim, dim))
m2 = jax.random.normal(k2, (dim, dim))

result = matrix_matrix(m1, m2)
print('test finished')
