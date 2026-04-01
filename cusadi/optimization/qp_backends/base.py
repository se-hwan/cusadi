import torch
import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from dataclasses import dataclass
from cusadi.parallelization import CASADI_FNS_DIR

@dataclass
class CudssData:
    n_envs: int
    interface: object
    Ap: torch.Tensor
    Ai: torch.Tensor
    n_rows: torch.Tensor
    n_cols: torch.Tensor
    n_nonzeros: torch.Tensor
    n_rhs: torch.Tensor
    Ax: torch.Tensor
    b: torch.Tensor
    x: torch.Tensor
    d_Ap: torch.Tensor
    d_Ai: torch.Tensor
    d_Ax: torch.Tensor
    d_b: torch.Tensor
    d_x: torch.Tensor


class QPBackend(ABC):
    casadi_fns_dir = CASADI_FNS_DIR

    @abstractmethod
    def __init__(self, problem, qp_cfg):
        pass

    @abstractmethod
    def setup(self, x_init=None, p_init=None) -> None:
        pass

    @abstractmethod
    def solve_step(self, x_eval, p_eval, solve_method=None):
        pass

    @abstractmethod
    def solve_parallelized(self, x_eval, p_eval):
        pass

    @abstractmethod
    def get_kkt_sparsity(self):
        pass

    @abstractmethod
    def _unpack_symbolics(self):
        pass

    def setup_parallelization(self,
                              linsys_method,
                              batch_size,
                              precision,
                              dynamic_batching):
        self.linsys_method = linsys_method
        self.batch_size = batch_size
        self.precision = precision
        self.dynamic_batching = dynamic_batching
        load_fns = self._check_function_signature()
        if load_fns:
            self.parallel_fns = self._load_parallel_functions()
        else:
            self.parallel_fns = self._build_parallel_functions(self.linsys_method)
            self._save_parallel_functions()
        self._setup_parallel_evaluation()
        return self.parallel_fns
    
    @abstractmethod
    def _check_function_signature(self) -> bool:
        pass

    @abstractmethod
    def _load_parallel_functions(self):
        pass

    @abstractmethod
    def _build_parallel_functions(self, linsys_method=None):
        pass

    @abstractmethod
    def _save_parallel_functions(self):
        pass

    @abstractmethod
    def _setup_parallel_evaluation(self):
        pass

    @abstractmethod
    def set_cusadi_functions(self, cusadi_fns):
        pass


    # TODO
    def _load_casadi_fns(self):
        '''
        Should check self.problem.name, and find the same functions, check against hashmap to only build when needed
        '''
        pass

    def _save_casadi_fns(self):
        '''
        Save casadi function with self.problem.name to cusadi/parallelization/casadi_fns
        '''
        pass

    def _setup_cudss_interface(self,
                               linsys_sparsity: ca.Sparsity,
                               n_envs: int,
                               precision: str):
        # JIT compile and import cuDSS interface
        try:
            from cusadi.parallelization.utils.jit_kernels import compile_and_load_cudss
            compile_and_load_cudss()
            # print("Loaded cuDSS interface.")
            import cudss
        except:
            print("Could not compile cuDSS interface.")
            raise SystemExit

        if precision == "float":
            self.cudss_interface = cudss.cudssInterface_float(n_envs)
            self.precision = {'torch': torch.float, 'np': np.float32}
        elif precision == "double":
            self.cudss_interface = cudss.cudssInterface_double(n_envs)
            self.precision = {'torch': torch.double, 'np': np.float64}
        else:
            raise ValueError(f"Unknown precision {precision}")
        
        # Load sparsity
        nnz = linsys_sparsity.nnz()
        n = linsys_sparsity.size1()

        # KKT system matrix parameters
        Ap = np.array(linsys_sparsity.colind(), dtype=np.int32)
        Ai = np.array(linsys_sparsity.row(), dtype=np.int32)
        Ap = np.tile(Ap, (n_envs, 1))
        Ai = np.tile(Ai, (n_envs, 1))
        Ap_cudss = torch.from_numpy(Ap).cuda().contiguous()
        Ai_cudss = torch.from_numpy(Ai).cuda().contiguous()
        n_rows = n * torch.ones(n_envs, dtype=torch.int32)
        n_cols = n * torch.ones(n_envs, dtype=torch.int32)
        n_nonzeros = nnz * torch.ones(n_envs, dtype=torch.int32)
        n_rhs = torch.ones(n_envs, dtype=torch.int32)

        # KKT system tensors for values
        Ax_cudss = torch.rand(
            n_envs, nnz, dtype=self.precision['torch'], device='cuda')
        b_cudss = torch.rand(
            n_envs, n, dtype=self.precision['torch'], device='cuda')
        x_cudss = torch.zeros(
            n_envs, n, dtype=self.precision['torch'], device='cuda')

        # Device pointers to tensor values
        d_Ap = torch.zeros(n_envs, dtype=torch.int64).cuda().contiguous()
        d_Ai = torch.zeros(n_envs, dtype=torch.int64).cuda().contiguous()
        d_Ax = torch.zeros(n_envs, dtype=torch.int64).cuda().contiguous()
        d_b = torch.zeros(n_envs, dtype=torch.int64).cuda().contiguous()
        d_x = torch.zeros(n_envs, dtype=torch.int64).cuda().contiguous()
        for i in range(n_envs):
            d_Ap[i] = Ap_cudss[i, :].data_ptr()
            d_Ai[i] = Ai_cudss[i, :].data_ptr()
            d_Ax[i] = Ax_cudss[i, :].data_ptr()
            d_b[i] = b_cudss[i, :].data_ptr()
            d_x[i] = x_cudss[i, :].data_ptr()

        self.cudss_data = CudssData(
            n_envs=n_envs,
            interface=self.cudss_interface,
            Ap=Ap_cudss,
            Ai=Ai_cudss,
            n_rows=n_rows,
            n_cols=n_cols,
            n_nonzeros=n_nonzeros,
            n_rhs=n_rhs,
            Ax=Ax_cudss,
            b=b_cudss,
            x=x_cudss,
            d_Ap=d_Ap,
            d_Ai=d_Ai,
            d_Ax=d_Ax,
            d_b=d_b,
            d_x=d_x,
        )

        self.cudss_data.interface.loadPointers(
            self.cudss_data.n_rows,
            self.cudss_data.n_cols,
            self.cudss_data.n_nonzeros,
            self.cudss_data.n_rhs,
            self.cudss_data.d_Ap,
            self.cudss_data.d_Ai,
            self.cudss_data.d_Ax,
            self.cudss_data.d_x,
            self.cudss_data.d_b,
        )
        self.cudss_data.interface.createMatrices()
        self.cudss_data.interface.factorizeSymbolic()
        print("cuDSS interface initialized.")