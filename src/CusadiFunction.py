import torch
import ctypes
from casadi import *
from src import CUSADI_BUILD_DIR

class CusadiFunction:
    # Public variables:
    fn_casadi = None
    fn_name = None
    num_instances = 0
    inputs_sparse = []
    outputs_sparse = []
    outputs_dense = []

    # Private variables:
    _device = 'cuda'
    _fn_library = None
    _work_tensor = []
    _input_tensors = []
    _input_ptrs = []
    _output_tensors = []
    _output_ptrs = []
    _fn_input = []
    _fn_work = []
    _fn_output = []

    # ! Public methods:
    def __init__(self, fn_casadi, num_instances):
        assert torch.cuda.is_available()
        lib_filepath = os.path.join(CUSADI_BUILD_DIR, f"lib{fn_casadi.name()}.so")
        self.fn_casadi = fn_casadi
        self.fn_name = fn_casadi.name()
        self.num_instances = num_instances
        self._fn_library = ctypes.CDLL(lib_filepath)
        self._fn_library.evaluate.restype = ctypes.c_float
        print("Loaded CasADi function: ", self.fn_casadi)
        print("Loaded library: ", self._fn_library)
        self._setup()

    def evaluate(self, inputs):
        self._clearTensors()
        self._prepareInputTensor(inputs)
        self.eval_time = self._fn_library.evaluate(self._fn_input,
                                                   self._fn_work,
                                                   self._fn_output,
                                                   self.num_instances)
        # torch.cuda.synchronize()

    def getDenseOutput(self, out_idx = None):
        env_idx = torch.tensor(range(self.num_instances), device=self._device).repeat_interleave(self.fn_casadi.nnz_out(out_idx))
        row_idx = torch.tensor((self.fn_casadi.sparsity_out(out_idx).get_triplet()[0]), device=self._device) \
            .repeat(self.num_instances)
        col_idx = torch.tensor((self.fn_casadi.sparsity_out(out_idx).get_triplet()[1]), device=self._device) \
            .repeat(self.num_instances)
        dim_dense = (self.num_instances, self.fn_casadi.size1_out(out_idx), self.fn_casadi.size2_out(out_idx))
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
            sys.exit(1)

    # ! Private methods:
    def _setup(self):
        self._input_tensors = [torch.zeros((self.num_instances, self.fn_casadi.nnz_in(i)),
                                            device=self._device, dtype=torch.double).contiguous()
                               for i in range(self.fn_casadi.n_in())]
        self._output_tensors = [torch.zeros(self.num_instances, self.fn_casadi.nnz_out(i),
                                            device=self._device, dtype=torch.double).contiguous()
                                for i in range(self.fn_casadi.n_out())]
        self._output_tensors_dense = [torch.zeros((self.num_instances,
                                                   self.fn_casadi.size1_out(i),
                                                   self.fn_casadi.size2_out(i)),
                                      device=self._device, dtype=torch.double).contiguous()
                                      for i in range(self.fn_casadi.n_out())]
        self._work_tensor = torch.zeros((self.num_instances, self.fn_casadi.sz_w()),
                                        device=self._device, dtype=torch.double).contiguous()
        self._input_ptrs = torch.zeros(self.fn_casadi.n_in(), device='cuda', dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(self.fn_casadi.n_out(), device='cuda', dtype=torch.int64).contiguous()
        for i in range(self.fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(self.fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()
        self._fn_input = self._castAsCPointer(self._input_ptrs.data_ptr(), 'int')
        self._fn_output = self._castAsCPointer(self._output_ptrs.data_ptr(), 'int')
        self._fn_work = self._castAsCPointer(self._work_tensor.data_ptr(), 'double')
        self.inputs_sparse = self._input_tensors
        self.outputs_sparse = self._output_tensors
        self.outputs_dense = self._output_tensors_dense

    def _prepareInputTensor(self, inputs):
        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        self._fn_input = self._castAsCPointer(self._input_ptrs.data_ptr(), 'int')
        self.inputs_sparse = self._input_tensors

    def _clearTensors(self):
        # for i in range(self.fn_casadi.n_in()):
        #     self._input_tensors[i].zero_()
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()
        self._work_tensor.zero_()

    def _castAsCPointer(self, ptr, type='float'):
        if type == 'int':
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))
        elif type == 'float':
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        elif type == 'double':
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))