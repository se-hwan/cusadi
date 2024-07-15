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
        print("Loaded CasADi function: ", self.fn_casadi)
        print("Loaded library: ", self._fn_library)
        self._setup()

    def evaluate(self, inputs):
        self._clearTensors()
        self._prepareInputTensor(inputs)
        self._fn_library.evaluate(self._fn_input,
                                  self._fn_work,
                                  self._fn_output,
                                  self.num_instances)
        # torch.cuda.synchronize()

    def getDenseOutputsForEnv(self, env_idx, out_idx = None):
        if out_idx is None:
            out_idx = range(self.fn_casadi.n_out())
        out_sparse = [torch.sparse_coo_tensor(self.fn_casasdi.sparsity_out(i).get_triplet(),
                                              self._output_tensors[i][env_idx, :]) 
                      for i in out_idx]
        return out_sparse.to_dense()
    
    # ! Private methods:
    def _setup(self):
        self._input_tensors = [torch.zeros((self.num_instances, self.fn_casadi.nnz_in(i)),
                               device=self._device, dtype=torch.float32).contiguous()
                               for i in range(self.fn_casadi.n_in())]
        self._output_tensors = [torch.zeros(self.num_instances, self.fn_casadi.nnz_out(i),
                                device=self._device, dtype=torch.float32).contiguous()
                                for i in range(self.fn_casadi.n_out())]
        self._work_tensor = torch.zeros((self.num_instances, self.fn_casadi.sz_w()),
                                        device=self._device, dtype=torch.float32).contiguous()
        self._input_ptrs = torch.zeros(self.fn_casadi.n_in(), device='cuda', dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(self.fn_casadi.n_out(), device='cuda', dtype=torch.int64).contiguous()
        for i in range(self.fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(self.fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()
        self._fn_input = self._castAsCPointer(self._input_ptrs.data_ptr(), 'int')
        self._fn_output = self._castAsCPointer(self._output_ptrs.data_ptr(), 'int')
        self._fn_work = self._castAsCPointer(self._work_tensor.data_ptr(), 'float')
        self.inputs_sparse = self._input_tensors
        self.outputs_sparse = self._output_tensors

    def _prepareInputTensor(self, inputs):
        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
        for i in range(self.fn_casadi.n_in()):
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