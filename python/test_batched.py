from casadi import *
import numpy
import torch
import time
from eval_casadi import eval_casadi_torch
import cupy

f = Function.load('test.casadi')

f.n_in()
f.n_out() 

print("Number of input cells: ", len(f.sx_in()))
for i in range(len(f.sx_in())):
  print("Input cell ", i, " has dimensions: ", f.sx_in(i))
print("Number of output cells: ", f.sx_out())
for i in range(len(f.sx_out())):
  print("Output cell ", i, " has dimensions: ", f.sx_out(i))

print("Size of work vector: ", f.sz_w())

# loop through n_in for size_in to get dimensions of inputs
# same for outputs
[n_rows_out, n_cols_out] = f.size_out(0)

# ! Check if pytorch autograd of g(x) is faster than casadi autodiffed function
# ! jacobian() vs. casadi.jacobian()?

test1 = f.sparsity_out("dgdx")
# print(test1)
# print(type(test1))
# print(dir(test1))

[sparse_row_idx, sparse_col_idx] = test1.get_crs()

N_ENVS = 4096

num_instructions = f.n_instructions()
const_instructions = [f.instruction_constant(i) for i in range(num_instructions)]
operations = [f.instruction_id(i) for i in range(num_instructions)]
output_idx = [f.instruction_output(i) for i in range(num_instructions)]
input_idx = [f.instruction_input(i) for i in range(num_instructions)]
nnz_out = f.nnz_out()

f_info = {
    'num_instructions': num_instructions,
    'const_instructions': const_instructions,
    'operations': operations,
    'output_idx': output_idx,
    'input_idx': input_idx,
    'n_out': f.n_out(),
    'nnz_out': nnz_out,
    'sz_w': f.sz_w()
}

print(output_idx)
print(len(output_idx))
exit(1)

output = [torch.zeros(N_ENVS, f_info['nnz_out'], 1, device='cuda')
                for i in range(f_info['n_out'])]
work = torch.zeros(N_ENVS, f_info['sz_w'], 1, device='cuda')

input_val_cuda = [torch.ones((N_ENVS, 192, 1), device='cuda'),
                  torch.rand((N_ENVS, 52, 1), device='cuda')]
input_val_cpu = [torch.ones((N_ENVS, 192, 1), device='cpu'),
                 torch.rand((N_ENVS, 52, 1), device='cpu')]

[out_cuda, computation_time_cuda] = eval_casadi_torch(f_info, input_val_cuda, output, work, 'cuda')
[out_cpu, computation_time_cpu] = eval_casadi_torch(f_info, input_val_cpu, output, work, 'cpu')

print("Computation time for ", N_ENVS, " environments on GPU: ", computation_time_cuda)
print("Computation time for ", N_ENVS, " environments on CPU: ", computation_time_cpu)

########################################

output_numpy = numpy.zeros((N_ENVS, test1.nnz(), 1))

time_start = time.time()
for i in range(N_ENVS):
  # input_numpy = [input_val_cpu[0][i, :, :].numpy(), input_val_cpu[1][i, :, :].numpy()]
  input_val = [numpy.ones((192, 1)), \
               numpy.random.rand(52, 1)]
  output_numpy[i, :, 0] = (f.call(input_val))[0].nonzeros()

# output_numpy = torch.from_numpy(output_numpy).to('cuda')
time_end = time.time()
print("Computation time for ", N_ENVS, " environments serially evaluated and moved to GPU: ", time_end - time_start)

print("total instructions: ", f.n_instructions())


# output_val = [torch.zeros((N_ENVS, test1.nnz(), 1), device=compute_device)]
# work = torch.zeros(N_ENVS, f.sz_w(), 1, device=compute_device)

# # For debugging
# instr = f.instructions_sx()
# print("Number of instructions: ", f.n_instructions())


# start = time.time()

# # Loop over the algorithm
# for k in range(f.n_instructions()):

#   # Get the atomic operation
#   op = f.instruction_id(k)
#   o = f.instruction_output(k)
#   i = f.instruction_input(k)

#   if(op==OP_CONST):
#     work[:, o[0]] = f.instruction_constant(k)
#   else:
#     if op==OP_INPUT:
#       work[:, o[0]] = input_val[i[0]][:, i[1]]
#     elif op==OP_OUTPUT:
#       output_val[o[0]][:, o[1]] = work[:, i[0]]
#     elif op==OP_ADD:
#       work[:, o[0]] = work[:, i[0]] + work[:, i[1]]
#     elif op==OP_SUB:
#       work[:, o[0]] = work[:, i[0]] - work[:, i[1]]
#     elif op==OP_NEG:
#       work[:, o[0]] = -work[:, i[0]]
#     elif op==OP_MUL:
#       work[:, o[0]] = work[:, i[0]] * work[:, i[1]]
#     elif op==OP_DIV:
#       work[:, o[0]] = work[:, i[0]] / work[:, i[1]]
#     elif op==OP_SIN:
#       work[:, o[0]] = torch.sin(work[:, i[0]])
#     elif op==OP_COS:
#       work[:, o[0]] = torch.cos(work[:, i[0]])
#     elif op==OP_TAN:
#       work[:, o[0]] = torch.tan(work[:, i[0]])
#     elif op==OP_SQ:
#       work[:, o[0]] = work[:, i[0]] * work[:, i[0]]
#     elif op==OP_SQRT:
#       work[:, o[0]] = torch.sqrt(work[:, i[0]])
#     else:
#       raise Exception('Unknown CasADi operation: ' + str(op))

# end = time.time()
# print('Time elapsed for batched: ', end - start)


# ########################################
# ############ SERIAL ON GPU #############
# ########################################

# input_val = [torch.ones((N_ENVS, 192, 1)), torch.rand((N_ENVS, 52, 1))]



# # Input values of the same dimensions as the above
# input_val = [numpy.ones((N_ENVS, 192, 1)), \
#              numpy.random.rand(N_ENVS, 52, 1)]

# output_serial = torch.zeros((N_ENVS, test1.nnz(), 1))
# start = time.time()
# for i in range(N_ENVS):
#   input_val_i = [input_val[0][i, :], input_val[1][i, :]]
#   output_serial[i, :] = (f.call(input_val_i))[0]

# torch.from_numpy(output_serial[i]).float().to('cuda')
# end = time.time()
# print('Time elapsed for serial: ', end - start)

# # print(eval_full)

# error = (eval_full - numpy_eval.full())

# print('------')
# print('Function: ', str(f))
# print('Sum eval: ', sum(sum(eval_full)))
# print('Sum expected: ', sum(sum(numpy_eval.full())))
# # print('Error: ', sum(error))
