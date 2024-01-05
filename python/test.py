from casadi import *
import numpy
import torch


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

# Input values of the same dimensions as the above
input_val = [numpy.ones((192, 1)), \
             numpy.random.rand(52, 1)]
numpy_eval = (f.call(input_val))[0]
print(type(numpy_eval))
# print(sum(numpy_eval.full()))


import matplotlib.pyplot
matplotlib.pyplot.spy(numpy_eval)
matplotlib.pyplot.show()

input_val = [torch.ones((192, 1), device='cuda'), torch.tensor(input_val[1] ,device='cuda')]

# Output values to be calculated of the same dimensions as the above
# output_val = [numpy.ones((357*192, 1))]
output_val = [torch.zeros((test1.nnz(), 1), device='cuda')]

# Work vector
work = torch.zeros(f.sz_w(), device='cuda')

# For debugging
instr = f.instructions_sx()
print("Number of instructions: ", f.n_instructions())

# Loop over the algorithm
for k in range(f.n_instructions()):

  # Get the atomic operation
  op = f.instruction_id(k)
  o = f.instruction_output(k)
  i = f.instruction_input(k)

  if(op==OP_CONST):
    work[o[0]] = f.instruction_constant(k)
    # print('work[', o[0], '] = ', f.instruction_constant(k))
  else:
    if op==OP_INPUT:
      work[o[0]] = input_val[i[0]][i[1]]
      # i[0] selects which SET of inputs to use (state vs. parameters vs. any other input)
      # i[1] selects within that set which VALUE to get
      # v = input_val[i[0]]
      # work[o[0]] = v[i[1]]
      # print('work[', o[0], '] = input[', i[0], '][', i[1],  ']', '            ---> ' , work[o[0]])
    elif op==OP_OUTPUT:
      # print("output idx: ", o[0], ' ', o[1])
      output_val[o[0]][o[1]] = work[i[0]]
      # v = output_val[o[0]]
      # v[o[1]] = work[i[0]]
      # output_val[o[0]] = v
      # print('output[', o[0], '][', o[1], '] = work[', i[0], ']','             ---> ', output_val[o[0]][o[1]])
    elif op==OP_ADD:
      work[o[0]] = work[i[0]] + work[i[1]]
      # print('work[', o[0], '] = work[', i[0], '] + work[', i[1], ']','        ---> ', work[o[0]])
    elif op==OP_SUB:
      work[o[0]] = work[i[0]] - work[i[1]]
      # print('work[', o[0], '] = work[', i[0], '] - work[', i[1], ']','        ---> ', work[o[0]])
    elif op==OP_NEG:
      work[o[0]] = -work[i[0]]
      # print('work[', o[0], '] = -work[', i[0], ']','                        ---> ', work[o[0]])
    elif op==OP_MUL:
      work[o[0]] = work[i[0]] * work[i[1]]
      # print('work[', o[0], '] = work[', i[0], '] * work[', i[1], ']','        ---> ', work[o[0]])
    elif op==OP_DIV:
      work[o[0]] = work[i[0]] / work[i[1]]
      # print('work[', o[0], '] = work[', i[0], '] / work[', i[1], ']','        ---> ', work[o[0]])
    elif op==OP_SIN:
      work[o[0]] = torch.sin(work[i[0]])
      # print('work[', o[0], '] = sin(work[', i[0], '])','                     ---> ', work[o[0]])
    elif op==OP_COS:
      work[o[0]] = torch.cos(work[i[0]])
      # print('work[', o[0], '] = cos(work[', i[0], '])','                     ---> ', work[o[0]])
    elif op==OP_TAN:
      work[o[0]] = torch.tan(work[i[0]])
      # print('work[', o[0], '] = tan(work[', i[0], '])','                     ---> ', work[o[0]])
    elif op==OP_SQ:
      work[o[0]] = work[i[0]] * work[i[0]]
      # print('work[', o[0], '] = work[', i[0], '] * work[', i[0], ']','        ---> ', work[o[0]])
    elif op==OP_SQRT:
      work[o[0]] = torch.sqrt(work[i[0]])
      # print('work[', o[0], '] = sqrt(work[', i[0], '])','                    ---> ', work[o[0]])
    else:
      disp_in = ["work[" + str(a) + "]" for a in i]
      debug_str = print_operator(instr[k],disp_in)
      raise Exception('Unknown operation: ' + str(op) + ' -- ' + debug_str)
  output_sum = torch.sum(output_val[0]).item()
  work_sum = torch.sum(work).item()
  if np.isnan(work_sum) or np.isnan(output_sum):
    print("Nan detected!")
    print("Instruction: ", k)
    print("Output indices: ", o)
    print("Input indices: ", i)
    print("Operation: ", op)
    exit(1)

eval_full = numpy.zeros((357, 192))
# print(sparse_row_idx)
# print(sparse_col_idx)

row_idx = test1.row()
col_idx = test1.get_col()
for i in range(test1.nnz()):
  eval_full[row_idx[i], col_idx[i]] = output_val[0][i].cpu()

matplotlib.pyplot.spy(eval_full)
matplotlib.pyplot.show()


# print(eval_full)

error = (eval_full - numpy_eval.full())

print('------')
print('Function: ', str(f))
print('Sum eval: ', sum(sum(eval_full)))
print('Sum expected: ', sum(sum(numpy_eval.full())))
# print('Error: ', sum(error))
