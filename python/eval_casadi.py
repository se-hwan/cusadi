from casadi import *
import torch
import time

def eval_casadi_torch(f_info, input_batch, output, work, compute_device='cuda'):
    num_instructions = f_info['num_instructions']
    const_instructions = f_info['const_instructions']
    operations = f_info['operations']
    output_idx = f_info['output_idx']
    input_idx = f_info['input_idx']

    start = time.time()

    # Loop over the algorithm
    for k in range(num_instructions):
        # Get the atomic operation
        op = operations[k]
        o = output_idx[k]
        i = input_idx[k]
        t0 = time.time()
        if(op==OP_CONST):
            work[:, o[0]] = const_instructions[k]
        else:
            if op==OP_INPUT:
                work[:, o[0]] = input_batch[i[0]][:, i[1]]
            elif op==OP_OUTPUT:
                output[o[0]][:, o[1]] = work[:, i[0]]
            elif op==OP_ADD:
                work[:, o[0]] = work[:, i[0]] + work[:, i[1]]
            elif op==OP_SUB:
                work[:, o[0]] = work[:, i[0]] - work[:, i[1]]
            elif op==OP_NEG:
                work[:, o[0]] = -work[:, i[0]]
            elif op==OP_MUL:
                work[:, o[0]] = work[:, i[0]] * work[:, i[1]]
            elif op==OP_DIV:
                work[:, o[0]] = work[:, i[0]] / work[:, i[1]]
            elif op==OP_SIN:
                work[:, o[0]] = torch.sin(work[:, i[0]])
            elif op==OP_COS:
                work[:, o[0]] = torch.cos(work[:, i[0]])
            elif op==OP_TAN:
                work[:, o[0]] = torch.tan(work[:, i[0]])
            elif op==OP_SQ:
                work[:, o[0]] = work[:, i[0]] * work[:, i[0]]
            elif op==OP_SQRT:
                work[:, o[0]] = torch.sqrt(work[:, i[0]])
            else:
                raise Exception('Unknown CasADi operation: ' + str(op))
        t1 = time.time()
        torch.cuda.synchronize()
        print("Operation time: ", t1-t0)
    if compute_device == 'cpu':
        output = output[0].to('cuda')
    end = time.time()
    computation_time = end - start

    return output, computation_time

# f = Function.load('test.casadi')
# N_ENVS = 4096
# input_val_cuda = [torch.ones((N_ENVS, 192, 1), device='cuda'),
#                   torch.rand((N_ENVS, 52, 1), device='cuda')]

# import torch._dynamo
# torch._dynamo.reset()
# test_torch_compile = torch.compile(eval_casadi_torch, mode='reduce-overhead')



def eval_casadi_torch_hashed(f, input_batch, compute_device='cuda'):
    
    num_envs = input_batch[0].size()[0]

    if f.class_name() == "MXFunction":
        f = f.expand()

    output = [torch.zeros(num_envs, f.nnz_out(), 1, device=compute_device)
              for _ in f.sx_out()]

    # Work vector
    work = torch.zeros(num_envs, f.sz_w(), 1, device=compute_device)

    st = time.time()
    operations = {
        OP_CONST: lambda f, k, i: f.instruction_constant(k),
        OP_INPUT: lambda f, k, i: input_batch[i[0]][:, i[1]],
        OP_ADD: lambda f, k, i: work[:, i[0]] + work[:, i[1]],
        OP_SUB: lambda f, k, i: work[:, i[0]] - work[:, i[1]],
        OP_NEG: lambda f, k, i: -work[:, i[0]],
        OP_MUL: lambda f, k, i: work[:, i[0]] * work[:, i[1]],
        OP_DIV: lambda f, k, i: work[:, i[0]] / work[:, i[1]],
        OP_SIN: lambda f, k, i: torch.sin(work[:, i[0]]),
        OP_COS: lambda f, k, i: torch.cos(work[:, i[0]]),
        OP_TAN: lambda f, k, i: torch.tan(work[:, i[0]]),
        OP_SQ: lambda f, k, i: work[:, i[0]] * work[:, i[0]],
        OP_SQRT: lambda f, k, i: torch.sqrt(work[:, i[0]])
    }
    ed = time.time()
    print("Hash table creation time: ", ed-st)

    start = time.time()
    # Loop over the algorithm
    for k in range(f.n_instructions()):
        # Get the atomic operation
        op = f.instruction_id(k)
        o = f.instruction_output(k)
        i = f.instruction_input(k)

        if op==OP_OUTPUT:
            output[o[0]][:, o[1]] = work[:, i[0]]
        else:
            work[:, o[0]] = operations[op](f, k, i)          
            # try:
            #     work[:, o[0]] = operations[op](f, k)
            # except KeyError:
            #     raise Exception('Unknown CasADi operation: ' + str(op))
    if compute_device == 'cpu':
        output = output[0].to('cuda')
    end = time.time()
    computation_time = end - start

    return output, computation_time


import torch._dynamo
torch._dynamo.reset()
test_torch_compile = torch.compile(eval_casadi_torch, mode='reduce-overhead')
