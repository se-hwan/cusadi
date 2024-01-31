import ctypes
import torch
import time
from casadi import *
from evaluateCasADiPython import evaluateCasADiPython

# Load the shared library
libc = ctypes.CDLL('./build/libc_libs.so')
libc.evaluateCasADiFunction.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),

    ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_int),

    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,

    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int,
    ctypes.c_int]
libc.printOperationKeys()

assert torch.cuda.is_available()
device = torch.device("cuda")  # device object representing GPU

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
        self.parseFunctionParameters(self.f)
        self.prepareFunctionData()
        self.printCasADiFunction()

    def parseFunctionParameters(self, casadi_fn):
        n_instr = f.n_instructions()
        input_idx = []
        input_idx_lengths = [0]
        output_idx = []
        output_idx_lengths = [0]
        for i in range(n_instr):
            input_idx.extend(casadi_fn.instruction_input(i))
            input_idx_lengths.append(len(casadi_fn.instruction_input(i)))
            output_idx.extend(casadi_fn.instruction_output(i))
            output_idx_lengths.append(len(casadi_fn.instruction_output(i)))

        self.f_info = {
            'n_instr': n_instr,
            'n_in': f.n_in(),
            'sz_in': [f.size1_in(i) for i in range(f.n_in())],
            'n_out': f.n_out(),
            'nnz_out': [f.nnz_out(i) for i in range(f.n_out())],
            'n_w': f.sz_w(),
            'const_instr': [f.instruction_constant(i) for i in range(n_instr)],
            'operations': [f.instruction_id(i) for i in range(n_instr)],
            'input_idx': input_idx,
            'output_idx': output_idx,
            'input_idx_lengths': input_idx_lengths,
            'output_idx_lengths': output_idx_lengths}

    def setBenchmarkingMode(self):
        '''Create additional tensors for evaluation in other modes'''
        # Pytorch evaluation tensors:
        self.output_py = [torch.zeros(N_ENVS, self.f.nnz_out(), device='cuda') for i in range(self.f.n_out())]
        self.work_tensor_py = torch.zeros(N_ENVS, self.f.sz_w(), device='cuda')
        self.const_instr_py = [f.instruction_constant(i) for i in range(self.f_info['n_instr'])]
        self.operations_py = [f.instruction_id(i) for i in range(self.f_info['n_instr'])]
        self.output_idx_py = [f.instruction_output(i) for i in range(self.f_info['n_instr'])]
        self.input_idx_py = [f.instruction_input(i) for i in range(self.f_info['n_instr'])]

        # Serial CasADi evaluation Numpy arrays:

    def prepareFunctionData(self):
        # Put data on GPU with Pytorch tensors
        self.operations = torch.tensor(self.f_info['operations'], device=self.device, dtype=torch.int32).contiguous()
        self.const_instr = torch.tensor(self.f_info['const_instr'], device=self.device, dtype=torch.float32).contiguous()
        self.input_idx = torch.tensor(self.f_info['input_idx'], device=self.device, dtype=torch.int32).contiguous()
        self.input_idx_lengths = torch.tensor(self.f_info['input_idx_lengths'], device=self.device, dtype=torch.int32).contiguous()
        self.output_idx = torch.tensor(self.f_info['output_idx'], device=self.device, dtype=torch.int32).contiguous()
        self.output_idx_lengths = torch.tensor(self.f_info['output_idx_lengths'], device=self.device, dtype=torch.int32).contiguous()
        self.work_tensor = torch.zeros(self.batch_size, self.f_info['n_w'], device=self.device, dtype=torch.float32).contiguous()
        self.outputs = [torch.zeros(self.batch_size, self.f_info['nnz_out'][i], device='cuda') for i in range(self.f.n_out())]

        # Data that needs to be accessed by C must be on the CPU
        # Passing these variables by value right now, but could use pointers instead
        self.op_C = (ctypes.c_int * len(self.f_info['operations']))(*self.f_info['operations'])
        self.const_instr_C = (ctypes.c_float * len(self.f_info['const_instr']))(*self.f_info['const_instr'])
        self.input_idx_C = (ctypes.c_int * len(self.f_info['input_idx']))(*self.f_info['input_idx'])
        self.input_idx_lengths_C = (ctypes.c_int * len(self.f_info['input_idx_lengths']))(*self.f_info['input_idx_lengths'])
        self.output_idx_C = (ctypes.c_int * len(self.f_info['output_idx']))(*self.f_info['output_idx'])
        self.output_idx_lengths_C = (ctypes.c_int * len(self.f_info['output_idx_lengths']))(*self.f_info['output_idx_lengths'])
        self.output_sz_ptr = (ctypes.c_int * len(self.f_info['nnz_out']))(*self.f_info['nnz_out'])
        self.input_sz_ptr = (ctypes.c_int * len(self.f_info['sz_in']))(*self.f_info['sz_in'])

        self.work_tensor_ptr = ctypes.cast((self.work_tensor.data_ptr()), ctypes.POINTER(ctypes.c_float))
        output_ptr = [self.castAsCPointer(self.outputs[i].data_ptr()) for i in range(self.f.n_out())]
        self.output_tensors_ptr = (ctypes.POINTER(ctypes.c_float) * self.f_info['n_out'])(*output_ptr)

    def castAsCPointer(self, ptr, type='float'):
        if type == 'int':
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))
        elif type == 'float':
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))

    def printCasADiFunction(self):
        print("Received function: ", self.f)

    def evaluateVectorizedWithCUDA(self, input_batch_ptr):
        # input_batch_ptr is a list of pointers to tensors
        assert len(input_batch_ptr) == self.f.n_in()
        for i in range(self.f_info['n_out']):
            self.outputs[i] *= 0
        self.work_tensor *= 0
        libc.evaluateCasADiFunction(self.output_tensors_ptr,
                                    self.f_info['n_out'],
                                    self.output_sz_ptr,

                                    input_batch_ptr,
                                    self.f_info['n_in'],
                                    self.input_sz_ptr,
                                    
                                    self.work_tensor_ptr,
                                    self.f_info['n_w'],

                                    self.op_C,
                                    self.output_idx_C,
                                    self.output_idx_lengths_C,
                                    self.input_idx_C,
                                    self.input_idx_lengths_C,
                                    self.const_instr_C,
                                    self.f_info['n_instr'],
                                    self.batch_size)
        return self.output_tensors_ptr # [batch_size x nnz]. How should we return this? 
    
    def evaluateVectorizedWithPytorch(self, input_batch):
        evaluateCasADiPython(self.output_py,
                             self.work_tensor_py,
                             input_batch,
                             self.operations_py,
                             self.output_idx_py,
                             self.input_idx_py,
                             self.const_instr_py,
                             self.f_info['n_instr'])
        return self.output_py

    def evaluateSerialWithCasADi(self, input):
        for i in range(self.batch_size):
            f.call(input[i])

f = Function.load("test.casadi")

# Example data on GPU
N_ENVS = 4096
N_RUNS = 1

cusadi_fn = CusADiFunction(f, N_ENVS)
cusadi_fn.setBenchmarkingMode()

a_device = torch.ones((N_ENVS, 192), device='cuda', dtype=torch.float32)
b_device = torch.ones((N_ENVS, 52), device='cuda', dtype=torch.float32)
a_ptr = cusadi_fn.castAsCPointer(a_device.data_ptr())
b_ptr = cusadi_fn.castAsCPointer(b_device.data_ptr())
test_ptr = [a_ptr, b_ptr]
test_input = (ctypes.POINTER(ctypes.c_float) * len(test_ptr))(*test_ptr)
print(test_input)
print(len(test_input))
print(a_device[0, 0])


########## * CUDA vectorized benchmark * ##########
t_cuda = []
for i in range(N_RUNS):
    t0 = time.time()
    cusadi_fn.evaluateVectorizedWithCUDA(test_input)
    torch.cuda.synchronize()
    duration = time.time() - t0
    t_cuda.append(duration)
print("CUDA wall time: ", duration)


########## * Pytorch vectorized benchmark * ##########
pytorch_input = [torch.ones((N_ENVS, 192), device='cuda', dtype=torch.float32),
                 torch.ones((N_ENVS, 52), device='cuda', dtype=torch.float32)]
t_pytorch = []
for i in range(N_RUNS):
    t0 = time.time()
    cusadi_fn.evaluateVectorizedWithPytorch(pytorch_input)
    torch.cuda.synchronize()
    duration = time.time() - t0
    t_pytorch.append(duration)
print("Pytorch wall time: ", duration)


########## * Serial CPU benchmark * ##########
output_numpy = numpy.zeros((N_ENVS, f.nnz_out()))
t_serial = []
for j in range(N_RUNS):
    time_start = time.time()
    for i in range(N_ENVS):
        input_val = [numpy.ones((192, 1)), \
                    numpy.ones((52, 1))]
        output_numpy[i, :] = (f.call(input_val))[0].nonzeros()
        test = torch.from_numpy(output_numpy[i, :]).to('cuda')
    duration = time.time() - time_start
    t_serial.append(duration)
print("Computation time for ", N_ENVS, " environments serially evaluated and moved to GPU: ", duration)


# PLOTTING
import matplotlib.pyplot as plt

sparse_eval = (f.call(input_val))[0]
plt.spy(sparse_eval)
plt.title('Sparsity pattern of dg/dx')
plt.savefig('boxplot_function_evaluation_time.png')



# means = [numpy.mean(t_cuda), numpy.mean(t_pytorch), numpy.mean(t_serial)]
# stds = [numpy.std(t_cuda), numpy.std(t_pytorch), numpy.std(t_serial)]
# labels = ['CUDA', 'Pytorch', 'Serial']
# colors = ['lightgreen', 'skyblue', 'lightcoral']

# # Create a bar plot with error bars representing standard deviation
# plt.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2)
# plt.ylabel('Time (s)')
# plt.title('Evaluation time of dg/dx')

# Save the plot to a file (e.g., PNG format)
# plt.savefig('boxplot_function_evaluation_time.png')




'''

print("test end")

print("testing column addition: ")
libc.launchMyCudaFunction(c_ptr, 0, a_ptr, 3, b_ptr, 3, N_ENVS, 5)
torch.cuda.synchronize()
print(a_device[0:3, :])
print(b_device[0:3, :])
print(c_device[0:3, :])

t0 = time.time()
for i in range(N_RUNS):
    libc.launchMyCudaFunction(c_ptr, 0, a_ptr, 0, b_ptr, 0, N_ENVS, 5)
torch.cuda.synchronize()
print("CUDA wall time: ", (time.time()-t0)/N_RUNS)

t0 = time.time()
for i in range(N_RUNS):
    c_device[:, 0] = torch.add(a_device[:, 0], b_device[:, 0])
torch.cuda.synchronize()
print("Pytorch wall time: ", (time.time()-t0)/N_RUNS)
'''
