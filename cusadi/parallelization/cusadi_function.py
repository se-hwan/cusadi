import time
import torch
import numpy as np
from casadi import *

class CusadiFunction:
    # Public variables:
    fn_casadi = None
    fn_name = None
    n_in = 0
    n_out = 0
    batch_size = 0
    inputs_sparse = []
    outputs_sparse = []
    outputs_dense = []

    # Private variables:
    _device = 'cuda'
    _fn_kernel = None
    _work_tensor = []
    input_tensors = []
    output_tensors = []

    # ! Public methods:
    def __init__(self, fn_casadi, batch_size):
        assert torch.cuda.is_available()
        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self.n_in = fn_casadi.n_in()
        self.n_out = fn_casadi.n_out()
        self.batch_size = batch_size

        try:
            from .utils.jit_kernels import compile_and_load_kernels
            compile_and_load_kernels()
            import cusadi_kernels
        except:
            print("Could not import cusadi_kernels.")
            print("Generate kernels before instantiating CusadiFunction.")
            raise SystemExit
        self._fn_kernel = getattr(cusadi_kernels, self.fn_name)

        print("Loaded CasADi function: ", self.fn_casadi)
        print("Loaded library: ", self._fn_kernel)
        self._setup()

    def evaluate(self, inputs):
        self._clearTensors()
        self.input_tensors = [inputs[i].transpose(0, 1) for i in range(self.n_in)]
        self.eval_time = self._fn_kernel(self.batch_size,
                                         *self.input_tensors,
                                         *self.output_tensors,
                                         self._work_tensor)

    def test(self, n_test_envs=4096, seed=np.random.randint(0, 1e6)):
        torch.manual_seed(seed)
        # Randomized inputs
        input_tensors = [torch.rand(n_test_envs, self.fn_casadi.nnz_in(i), device=self._device, dtype=torch.float).contiguous()
                        for i in range(self.n_in)]
        print("Checking input dimensions...")
        print("    Input tensor sizes: ", [self.input_tensors[i].shape for i in range(self.n_in)])
        print("    Output tensor sizes: ", [self.output_tensors[i].shape for i in range(self.n_out)])
        print("    Work tensor size: ", self._work_tensor.shape)

        # Kernel evaluation
        start = time.time_ns()
        self.evaluate(input_tensors)
        end = time.time_ns()
        print(f"Time taken for {n_test_envs} environments (GPU): {(end - start)/1e9} seconds.")

        # CPU evaluaton
        outputs_np = [np.zeros((n_test_envs, self.fn_casadi.nnz_out(i))) for i in range(self.n_out)]
        start = time.time_ns()
        for n in range(n_test_envs):
            inputs_np = [input_tensors[i][n, :].cpu().numpy() for i in range(self.n_in)]
            for i in range(self.n_out):
                outputs_np[i][n, :] = self.fn_casadi.call(inputs_np)[i].nonzeros()
        end = time.time_ns()
        print(f"Time taken for {n_test_envs} environments (CPU): {(end - start)/1e9} seconds.")
        
        # Error calculation
        print(f"Average error for each environment:")
        for i in range(self.n_out):
            error_norm = 0
            for j in range(n_test_envs):
                output_cusadi_i = (self.output_tensors[i][j, :]).clone().cpu().numpy()
                error_norm += np.linalg.norm(output_cusadi_i - outputs_np[i][j, :])/self.fn_casadi.nnz_out(i)
            print(f"    Output {i}: Average error norm/env for {n_test_envs} envs.:", error_norm/n_test_envs)

    def getDenseOutput(self, out_idx = None):
        env_idx = torch.tensor(range(self.batch_size), device=self._device).repeat_interleave(self.fn_casadi.nnz_out(out_idx))
        row_idx = torch.tensor((self.fn_casadi.sparsity_out(out_idx).get_triplet()[0]), device=self._device) \
            .repeat(self.batch_size)
        col_idx = torch.tensor((self.fn_casadi.sparsity_out(out_idx).get_triplet()[1]), device=self._device) \
            .repeat(self.batch_size)
        dim_dense = (self.batch_size, self.fn_casadi.size1_out(out_idx), self.fn_casadi.size2_out(out_idx))
        return torch.sparse_coo_tensor(torch.vstack((env_idx, row_idx, col_idx)),
                                       self.outputs_sparse[out_idx].reshape(-1), 
                                       dim_dense).to_dense()
    
    def checkInputDimensions(self, inputs):
        self.input_CPU = [tensor[0, :].cpu().numpy() for tensor in inputs]
        try :
            out = (self.fn_casadi.call(self.input_CPU)[0]).full()
            print("CPU call successful. Tensor dimensions are correct for inputs.")
        except:
            print("Error in Casadi function call. Exiting...")
            raise SystemExit

    # * Private methods:
    # ! Need to generalize this for float type eventually
    def _setup(self):
        self.input_tensors = [torch.zeros((self.batch_size, self.fn_casadi.nnz_in(i)), device=self._device, dtype=torch.float).contiguous()
                               for i in range(self.n_in)]
        self.output_tensors = [torch.zeros((self.batch_size, self.fn_casadi.nnz_out(i)), device=self._device, dtype=torch.float).contiguous()
                                for i in range(self.n_out)]
        self._work_tensor = torch.zeros((self.fn_casadi.sz_w(), self.batch_size), device=self._device, dtype=torch.float).contiguous()
        

    def _clearTensors(self):
        for i in range(self.n_out):
            self.output_tensors[i].zero_()
        self._work_tensor.zero_()