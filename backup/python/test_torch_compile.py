from casadi import *
import numpy
import torch
import time

f = Function.load('test.casadi')
N_ENVS = 4096
N_RUNS = 100
input_val_cuda = [torch.ones((N_ENVS, 192, 1), device='cuda'),
                  torch.rand((N_ENVS, 52, 1), device='cuda')]

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


class MyOperation(torch.nn.Module):
    def __init__(self, f_info, num_envs, compute_device):
        super(MyOperation, self).__init__()
        self.num_instruction = f_info['num_instructions']
        self.const_instructions = f_info['const_instructions']
        self.operations = f_info['operations']
        self.output_idx = f_info['output_idx']
        self.input_idx = f_info['input_idx']
        self.num_envs = num_envs
        self.compute_device = compute_device
        self.output = [torch.zeros(num_envs, f_info['nnz_out'], 1, device=compute_device)
                       for i in range(f_info['n_out'])]
        self.work = torch.zeros(num_envs, f_info['sz_w'], 1, device=compute_device)
    
    def forward(self, input_batch):
        for k in range(self.num_instructions):
            # Get the atomic operation
            op = self.operations[k]
            o = self.output_idx[k]
            i = self.input_idx[k]
            if(op==OP_CONST):
                self.work[:, o[0]] = self.const_instructions(k)
            else:
                if op==OP_INPUT:
                    self.work[:, o[0]] = input_batch[i[0]][:, i[1]]
                elif op==OP_OUTPUT:
                    output[o[0]][:, o[1]] = self.work[:, i[0]]
                elif op==OP_ADD:
                    self.work[:, o[0]] = self.work[:, i[0]] + self.work[:, i[1]]
                elif op==OP_SUB:
                    self.work[:, o[0]] = self.work[:, i[0]] - self.work[:, i[1]]
                elif op==OP_NEG:
                    self.work[:, o[0]] = -self.work[:, i[0]]
                elif op==OP_MUL:
                    self.work[:, o[0]] = self.work[:, i[0]] * self.work[:, i[1]]
                elif op==OP_DIV:
                    self.work[:, o[0]] = self.work[:, i[0]] / self.work[:, i[1]]
                elif op==OP_SIN:
                    self.work[:, o[0]] = torch.sin(self.work[:, i[0]])
                elif op==OP_COS:
                    self.work[:, o[0]] = torch.cos(self.work[:, i[0]])
                elif op==OP_TAN:
                    self.work[:, o[0]] = torch.tan(self.work[:, i[0]])
                elif op==OP_SQ:
                    self.work[:, o[0]] = self.work[:, i[0]] * self.work[:, i[0]]
                elif op==OP_SQRT:
                    self.work[:, o[0]] = torch.sqrt(self.work[:, i[0]])
                else:
                    raise Exception('Unknown CasADi operation: ' + str(op))
        if self.compute_device == 'cpu':
            output = output[0].to('cuda')
        return output


def eval_casadi(f_info, input_batch, num_envs, output, work, compute_device):
    num_instruction = f_info['num_instructions']
    const_instructions = f_info['const_instructions']
    operations = f_info['operations']
    output_idx = f_info['output_idx']
    input_idx = f_info['input_idx']
    num_envs = num_envs
    compute_device = compute_device
    for k in range(num_instructions):
        # Get the atomic operation
        op = operations[k]
        o = output_idx[k]
        i = input_idx[k]
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
    if compute_device == 'cpu':
        output = output[0].to('cuda')
    return output



output = [torch.zeros(N_ENVS, f_info['nnz_out'], 1, device='cuda')
                for i in range(f_info['n_out'])]
work = torch.zeros(N_ENVS, f_info['sz_w'], 1, device='cuda')

out = eval_casadi(f_info, input_val_cuda, N_ENVS, output, work, 'cuda')

t0 = time.time()
test_torch_compile = torch.compile(eval_casadi, mode='reduce-overhead')
t1 = time.time()
print("compilation time: ", t1-t0)

# test_class = MyOperation( N_ENVS, 'cuda')
# test_torch_compile = torch.jit.trace(test_class, (f, N_ENVS, 'cuda'))


t_total = 0
for i in range(N_RUNS):
    test_torch_compile(f_info, input_val_cuda, N_ENVS, output, work, 'cuda')
    print("Run: ", i, " Time taken: ", t)
    t_total += t

print("Average time: ", t_total/N_RUNS)