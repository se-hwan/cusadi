import jax
import jax.dlpack
import jax.numpy as jnp
import jaxopt
import time
import torch




torch_input = torch.ones((1000, 1000), device='cuda', dtype=torch.float32)
t0 = time.time()
torch_input_dlpack = torch.utils.dlpack.to_dlpack(torch_input)
jax_input_from_torch = jax.dlpack.from_dlpack(torch_input_dlpack)
print("Time taken to move torch to jax: ", time.time() - t0)
print(jax_input_from_torch)

print(jnp.ones(3).device_buffer.device()) 



Q = 2 * jnp.array([[2.0, 0.5], [0.5, 1]])
c = jnp.array([1.0, 1.0])
A = jnp.array([[1.0, 1.0]])
b = jnp.array([1.0])
G = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
h = jnp.array([0.0, 0.0])

N = 1000
M = 5000
Q = 10*jnp.ones((N, N))
c = jnp.ones((N, 1))
A = jnp.ones((M, N))
b = jnp.ones((M, 1))
G = jnp.ones((M, N))
h = jnp.ones((M, 1))


# print(Q.__cuda_array_interface__)

qp = jaxopt.OSQP()
test_jit = jax.jit(qp.run)

t0 = time.time()
# sol = qp.run(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params
sol = test_jit(params_obj=(Q, c), params_eq=(A, b), params_ineq=(G, h)).params
print("Time taken for JaxOSQP: ", time.time() - t0)

# print(sol.primal)
# print(sol.dual_eq)
# print(sol.dual_ineq)

vec_to_move = jnp.ones((1000, 1000))

t0 = time.time()
jax_output_to_torch = jax.dlpack.to_dlpack(vec_to_move)
torch_output = torch.utils.dlpack.from_dlpack(jax_output_to_torch)
print("Time taken to move jax to torch: ", time.time() - t0)
print("torch output from dlpack: ", torch_output)


import numpy as np
import jax.numpy as jnp
import jax

def f(x):  # function we're benchmarking (works in both NumPy & JAX)
  return x.T @ (x - x.mean(axis=0))

t0 = time.time()
x_np = np.ones((1000, 1000), dtype=np.float32)  # same as JAX default dtype
print("Numpy time: ", time.time() - t0)

t0 = time.time()
x_jax = jax.device_put(x_np)  # measure JAX device transfer time
print("Transfer time: ", time.time() - t0)

t0 = time.time()
f_jit = jax.jit(f)
f_jit(x_jax).block_until_ready()
print("Jax compile time: ", time.time() - t0)

t0 = time.time()
f_jit(x_jax).block_until_ready()  # measure JAX runtime
print("Jax runtime: ", time.time() - t0)
