from casadi import *
import numpy
import torch
import time
from eval_casadi import eval_casadi_torch
import cupy as cp

start = cp.cuda.Event()
end = cp.cuda.Event()

N_RUNS = 1


# Example data on GPU
N_ENVS = 4096
a_device = torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32)
b_device = torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32)
c = b_device.clone()

torch.sum(a_device)

t_pytorch = 0
for i in range(50):
    start.record()
    c = torch.add(a_device, b_device)
    end.record()
    end.synchronize()
    t_pytorch += cp.cuda.get_elapsed_time(start, end)/1000
print("Pytorch add time: ", t_pytorch)


v1 = torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32)
v2 = torch.rand((N_ENVS, 1), device='cuda', dtype=torch.float32)
d = v1.clone()

t_pytorch = 0
for i in range(50):
    start.record()
    d = torch.add(v1, v2)
    end.record()
    end.synchronize()
    t_pytorch += cp.cuda.get_elapsed_time(start, end)/1000
print("Pytorch add time: ", t_pytorch)


casadi_filepath =os.path.join(os.path.dirname(__file__),  'test.casadi')

f = Function.load(casadi_filepath)
# f_mapped = Function.load('test_mapped.casadi')
# print(f_mapped)
# # # exit(1)
# input_numpy = [cupy.ones((192, N_ENVS)), cupy.random.rand(52, N_ENVS)]

# start_mapped_time = time.time()
# output_mapped = np.array((f_mapped.call(input_numpy))[0].nonzeros())
# # output_mapped = torch.from_numpy(output_mapped).to('cuda')
# end_mapped_time = time.time()
# print("CasADi map construct time: ", end_mapped_time - start_mapped_time)

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

output = [torch.zeros(N_ENVS, f_info['nnz_out'], 1, device='cuda')
                for i in range(f_info['n_out'])]
work = torch.zeros(N_ENVS, f_info['sz_w'], 1, device='cuda')


avg_time = 0
avg_time_cuda = 0
for i in range(N_RUNS):
    input_val_cpu = [torch.ones((N_ENVS, 192, 1), device='cpu'),
                     torch.rand((N_ENVS, 52, 1), device='cpu')]
    input_val_cuda = [torch.ones((N_ENVS, 192, 1), device='cuda'),
                      torch.rand((N_ENVS, 52, 1), device='cuda')]
    [out_cpu, t_cpu] = eval_casadi_torch(f_info, input_val_cpu, output, work, 'cpu')
    [out_cpu, t_cuda] = eval_casadi_torch(f_info, input_val_cuda, output, work, 'cuda')
    avg_time += t_cpu
    avg_time_cuda += t_cuda

print("Computation time for ", N_ENVS, " environments on CPU: ", avg_time/N_RUNS)
print("Computation time for ", N_ENVS, " environments on GPU: ", avg_time_cuda/N_RUNS)

