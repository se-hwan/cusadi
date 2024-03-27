import math
import torch
import time
from casadi import *

# Our module!
import lltm_cpp
import lltm_cuda
from python.evaluateCasADiPython import evaluateCasADiPython

assert torch.cuda.is_available()
device = torch.device("cuda")  # device object representing GPU

# Ultimately, C++ torch indexing and slicing is way too slow...
# Need to work directly with pointers


class CusADiFunction:
    f = None
    name = None
    device = None
    batch_size = 0

    def __init__(self, casadi_fn, batch_size, name="f", device='cuda'):
        self.f = casadi_fn
        self.name = name
        self.device = device
        self.batch_size = batch_size
        self.parseCasADiFunction(self.f)
        self.printCasADiFunction()
    
    def setBenchmarkingMode(self):
        # Create additional tensors for evaluation in other modes
        self.output_py = [torch.zeros(N_ENVS, self.f.nnz_out(), device='cuda') for i in range(self.f.n_out())]
        self.work_tensor_py = torch.zeros(N_ENVS, self.f.sz_w(), device='cuda')
        self.const_instr_py = [f.instruction_constant(i) for i in range(self.num_instructions)]
        self.operations_py = [f.instruction_id(i) for i in range(self.num_instructions)]
        self.output_idx_py = [f.instruction_output(i) for i in range(self.num_instructions)]
        self.input_idx_py = [f.instruction_input(i) for i in range(self.num_instructions)]

    def parseCasADiFunction(self, casadi_fn):
        self.num_instructions = casadi_fn.n_instructions()
        self.num_nonzero_out = casadi_fn.nnz_out()
        self.num_work_elem = casadi_fn.sz_w()
        operations = []
        const_instr = []
        input_idx = []
        input_idx_lengths = [0]
        output_idx = []
        for i in range(self.num_instructions):
            const_instr.append(casadi_fn.instruction_constant(i))
            operations.append(casadi_fn.instruction_id(i))
            input_idx.extend(casadi_fn.instruction_input(i))
            input_idx_lengths.append(len(casadi_fn.instruction_input(i)))
            if len(casadi_fn.instruction_output(i)) > 1:
                output_instr_i = casadi_fn.instruction_output(i)
                output_idx.append(output_instr_i[1])
            else:
                output_idx.extend(casadi_fn.instruction_output(i))

        self.operations = torch.tensor(operations, device=self.device, dtype=torch.int32)
        self.const_instr = torch.tensor(const_instr, device=self.device, dtype=torch.float32)
        self.input_idx = torch.tensor(input_idx, device=self.device, dtype=torch.int32)
        self.input_idx_lengths = torch.tensor(input_idx_lengths, device=self.device, dtype=torch.int32)
        self.output_idx = torch.tensor(output_idx, device=self.device, dtype=torch.int32)
        self.output = torch.zeros(self.batch_size, self.num_nonzero_out, device=self.device)
        self.work_tensor = torch.zeros(self.batch_size, self.num_work_elem, device=self.device)

    def printCasADiFunction(self):
        print("Received function: ", self.f)

    def evaluateWithCUDA(self, input_batch):
        # input_batch is a list of tensors
        assert len(input_batch) == self.f.n_in()
        self.output *= 0
        self.work_tensor *= 0
        lltm_cuda.evaluateCusADiFunction(self.output,
                                        self.work_tensor,
                                        input_batch,
                                        self.operations,
                                        self.output_idx,
                                        self.input_idx,
                                        self.input_idx_lengths,
                                        self.const_instr,
                                        self.num_instructions)
        return self.output # [batch_size x nnz]. How should we return this? 

    def evaluateWithCPP(self, input_batch):
        # input_batch is a list of tensors
        assert len(input_batch) == self.f.n_in()
        self.output *= 0
        self.work_tensor *= 0
        lltm_cpp.evaluateCusADiFunction(self.output,
                                        self.work_tensor,
                                        input_batch,
                                        self.operations,
                                        self.output_idx,
                                        self.input_idx,
                                        self.input_idx_lengths,
                                        self.const_instr,
                                        self.num_instructions)
        return self.output # [batch_size x nnz]. How should we return this? 
    
    def evaluateWithCasADiSerial(self, input):
        for i in range(self.batch_size):
            f.call(input[i])

    def evaluateWithPython(self, input_batch):
        self.output_py[0] *= 0
        self.work_tensor_py *= 0
        evaluateCasADiPython(self.output_py,
                             self.work_tensor_py,
                             input_batch,
                             self.operations_py,
                             self.output_idx_py,
                             self.input_idx_py,
                             self.const_instr_py,
                             self.num_instructions)


f = Function.load("test.casadi")

#### BENCHMARKING ####

N_ENVS = 4096
N_RUNS = 5

f_cpu = CusADiFunction(f, device='cpu', batch_size=N_ENVS)
f_cuda = CusADiFunction(f, device='cuda', batch_size=N_ENVS)
f_cpu.setBenchmarkingMode()
f_cuda.setBenchmarkingMode()

# * Evaluation in Python
dummy_input = [torch.ones((N_ENVS, 192), device='cuda', dtype=torch.float32),
               torch.ones((N_ENVS, 52), device='cuda', dtype=torch.float32)]
t_py_cuda = 0
for i in range(N_RUNS):
    t0 = time.time()
    out = f_cuda.evaluateWithPython(dummy_input)
    torch.cuda.synchronize()
    t_py_cuda += time.time() - t0
print("Time taken Python: ", t_py_cuda/N_RUNS, " s")


# * Evaluation in C++
dummy_input = [torch.ones((N_ENVS, 192), device='cuda', dtype=torch.float32),
               torch.ones((N_ENVS, 52), device='cuda', dtype=torch.float32)]
t_cpp_cuda = 0
for i in range(N_RUNS):
    t0 = time.time()
    out = f_cuda.evaluateWithCPP(dummy_input)
    torch.cuda.synchronize()
    t_cpp_cuda += time.time() - t0
print("Time taken C++: ", t_cpp_cuda/N_RUNS, " s")

# * Evaluation in CUDA
# dummy_input = [torch.ones((N_ENVS, 192), device='cuda', dtype=torch.float32),
#                torch.ones((N_ENVS, 52), device='cuda', dtype=torch.float32)]
# t_cpp_cuda = 0
# for i in range(N_RUNS):
#     t0 = time.time()
#     out = f_cuda.evaluateWithCUDA(dummy_input)
#     torch.cuda.synchronize()
#     t_cpp_cuda += time.time() - t0
# print("Time taken C++: ", t_cpp_cuda/N_RUNS, " s")


# # * Evaluation in C++ on CPU
# dummy_input = [torch.ones((N_ENVS, 192), device='cpu', dtype=torch.float32),
#                torch.ones((N_ENVS, 52), device='cpu', dtype=torch.float32)]
# t_cpp_cpu = 0
# for i in range(N_RUNS):
#     t0 = time.time()
#     out = f_cpu.evaluateWithCUDA(dummy_input)
#     t_cpp_cpu += time.time() - t0
# print("Time taken C++ on CPU: ", t_cpp_cpu/N_RUNS, " s")