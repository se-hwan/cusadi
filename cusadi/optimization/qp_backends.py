import osqp
import torch
import numpy as np
import casadi as ca
from abc import ABC, abstractmethod
from dataclasses import dataclass
from cusadi.parallelization import CusadiFunction


@dataclass
class CudssData:
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
    def parallelize(self,
                    linsys_method,
                    batch_size,
                    precision,
                    dynamic_batching):
        pass

    @abstractmethod
    def build_parallel_fns(self, linsys_method) -> list:
        pass

    @abstractmethod
    def _unpack_symbolics(self):
        pass


    def _setup_cudss_interface(self,
                     linsys_sparsity: ca.Sparsity,
                     n_envs: int,
                     precision: str):
        print("Initializing cuDSS interface...")
        # JIT compile and import cuDSS interface
        from cusadi.parallelization.utils.jit_kernels import compile_and_load_cudss
        compile_and_load_cudss()
        print("Loaded cuDSS interface.")
        import cudss
        try:
            from cusadi.parallelization.utils.jit_kernels import compile_and_load_cudss
            compile_and_load_cudss()
            print("Loaded cuDSS interface.")
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


class OSQPBackend(QPBackend):
    # OSQP default settings
    osqp_cfg = {
        "rho": 0.01,
        "verbose": False,
        "adaptive_rho": True,
        "max_iter": 25,
        "scaling": 10,
        "check_termination": 50,
        "sigma": 1e-6,
        "alpha": 1.6,
        "warm_starting": True
        }

    def __init__(self, opti_problem, qp_cfg=None):
        self.problem = opti_problem
        qp_cfg = qp_cfg or {}
        self.osqp_cfg.update(qp_cfg)
        self.qp_setup = False
        self.solver = osqp.OSQP()
        print("OSQP backend initialized")

    def setup(self, x_init=None, p_init=None):
        self.fn_qp_matrices = self._build_qp_functions()
        if x_init is not None and p_init is not None:
            qp_data = self._compute_qp_data(x_init, p_init)
            self._setup_problem(qp_data)

    # 1. Evaluate QP matrices, solve with OSQP
    # 2. Evaluate custom OSQP functions, solve with LDL
    def solve_step(self, x_eval, p_eval, solve_method=None):
        qp_data = self._compute_qp_data(x_eval, p_eval)
        if not self.qp_setup:
            self._setup_problem(qp_data)
        else:
            self._update_problem(qp_data)
        soln = np.array([self.solver.solve().x])
        return soln.reshape(x_eval.shape[0], 1)

    def _build_qp_functions(self):
        qp_sym = self._unpack_symbolics(self.problem.opti)
        return ca.Function(
            f"qp_matrices_{self.problem.name}",
            [qp_sym["x"], qp_sym["p"]],
            [qp_sym["P_triu"], qp_sym["c"], qp_sym["A"], qp_sym["z_lb"], qp_sym["z_ub"]],
            ["x", "p"],
            ["P_triu", "c", "A", "z_lb", "z_ub"],
            self.problem.fn_opts,
        )

    def _compute_qp_data(self, x_eval, p_eval):
        P_triu, c, A, lb, ub = self.fn_qp_matrices(x_eval, p_eval)
        return {
            "P_triu": P_triu,
            "c": c,
            "A": A,
            "lb": lb,
            "ub": ub,
        }

    def _setup_problem(self, qp_data):
        self.solver.setup(
            qp_data["P_triu"].tocsc(),
            qp_data["c"].toarray(),
            qp_data["A"].tocsc(),
            qp_data["lb"].toarray(),
            qp_data["ub"].toarray(),
            **self.osqp_cfg,
        )
        self.qp_setup = True

    def _update_problem(self, qp_data):
        self.solver.update(
            Px=qp_data["P_triu"].tocsc().data,
            Ax=qp_data["A"].tocsc().data,
            q=qp_data["c"].toarray(),
            l=qp_data["lb"].toarray(),
            u=qp_data["ub"].toarray(),
        )

    def _unpack_symbolics(self, opti):
        x, p, f, g = opti.x, opti.p, opti.f, opti.g
        g_lb, g_ub = opti.lbg, opti.ubg

        P, c = ca.hessian(f, x)
        P_triu = ca.triu(P)
        A = ca.jacobian(g, x)
        z_ub = ca.MX.inf(self.problem.n_g, 1)
        z_lb = -ca.MX.inf(self.problem.n_g, 1)
        z_ub[self.problem.eq_idx] = -g[self.problem.eq_idx]
        z_lb[self.problem.eq_idx] = -g[self.problem.eq_idx]
        z_ub[self.problem.ineq_ub_idx] = g_ub[self.problem.ineq_ub_idx] - g[self.problem.ineq_ub_idx]
        z_lb[self.problem.ineq_lb_idx] = g_lb[self.problem.ineq_lb_idx] - g[self.problem.ineq_lb_idx]

        matrix_KKT = ca.MX(self.problem.n_sys, self.problem.n_sys)
        matrix_KKT[:self.problem.n_x, :self.problem.n_x] = P
        matrix_KKT[self.problem.n_x:, :self.problem.n_x] = A
        matrix_KKT[:self.problem.n_x, self.problem.n_x:] = A.T
        return {'x': x,
                'p': p,
                'f': f,
                'g': g,
                'P_triu': P_triu,
                'c': c,
                'A': A,
                'z_lb': z_lb,
                'z_ub': z_ub,
                'matrix_KKT': matrix_KKT}

    # ************** OSQP GPU PARALLELIZATION FUNCTIONS ****************** #
    def build_parallel_fns(self, linsys_method):
        self.linsys_method = linsys_method
        qp_sym = self._unpack_symbolics(self.problem.opti)
        self.fn_KKT_matrix = self.compute_KKT_matrix(qp_sym["x"],
                                                     qp_sym["p"],
                                                     qp_sym["matrix_KKT"])
        self.fn_ruiz_scaling = self.compute_ruiz_scaling(
            qp_sym["x"],
            qp_sym["p"],
            qp_sym["c"],
            qp_sym["z_lb"],
            qp_sym["z_ub"],
            qp_sym["matrix_KKT"].sparsity(),
        )
        self.fn_KKT_vector = self.compute_KKT_vector()
        self.fn_ADMM_step = self.compute_ADMM_step()

        parallel_fns = [
            self.fn_KKT_matrix,
            self.fn_ruiz_scaling,
            self.fn_KKT_vector,
            self.fn_ADMM_step
        ]
        if linsys_method == 'ldl':
            print('Symbolic LDL linear solver selected.')
            self.fn_LDL_fac, self.fn_LDL_solve = self.compute_LDL_functions(
                self.fn_ruiz_scaling.sparsity_out(0))
            parallel_fns.append(self.fn_LDL_fac)
            parallel_fns.append(self.fn_LDL_solve)
        elif linsys_method == 'cudss':
            print('cuDSS linear solver chosen. No CusADi functions generated.')
        else:
            "Unknown symbolic linear solver option. Choose 'cudss' or 'ldl'."
        return parallel_fns

    def compute_KKT_matrix(self, x, p, kkt):
            kkt_triu = ca.triu(kkt)
            return ca.Function(
                f"KKT_matrix_{self.problem.name}",
                [x, p],
                [kkt_triu],
                ["x", "p"],
                ["kkt_triu"],
                self.problem.fn_opts,
            )

    def compute_ruiz_scaling(self, x, p, c, z_lb, z_ub, kkt_sparsity):
        n_sys = self.problem.n_sys
        n_x = self.problem.n_x
        rho_norm_inv = self.problem.rho_norm_inv

        kkt_triu_sparsity = ca.triu(kkt_sparsity)
        kkt_nz = ca.MX.sym("kkt_nz", kkt_triu_sparsity.nnz(), 1)
        kkt_sym = ca.triu2symm(ca.MX(kkt_triu_sparsity, kkt_nz))
        sigma = ca.MX.sym("sigma", 1, 1)
        rho_bar = ca.MX.sym("rho_bar", 1, 1)

        S_scale = ca.MX.eye(n_sys)
        c_scale = 1

        P_kkt = kkt_sym[:n_x, :n_x]
        A_kkt = kkt_sym[n_x:, :n_x]
        P_bar = kkt_sym[:n_x, :n_x]
        A_bar = kkt_sym[n_x:, :n_x]
        q_bar = c
        M_bar = ca.MX(n_sys, n_sys)
        M_bar[:n_x, :n_x] = P_bar
        M_bar[n_x:, :n_x] = A_bar
        M_bar[:n_x, n_x:] = A_bar.T
        delta = ca.MX.ones(n_sys, 1)

        for _ in range(self.osqp_cfg['scaling']):
            for j in range(n_sys):
                M_j_inf = ca.mmax(ca.fabs(M_bar[:, j]))
                denom = ca.sqrt(M_j_inf)
                delta[j] = ca.if_else(denom <= 0, 1, 1 / denom)

            D = ca.diag(delta[:n_x])
            E = ca.diag(delta[n_x:])
            P_bar = D @ P_bar @ D
            A_bar = E @ A_bar @ D
            q_bar = D @ q_bar

            P_bar_sum = 0
            for j in range(n_x):
                P_bar_sum = P_bar_sum + ca.mmax(ca.fabs(P_bar[:, j]))
            P_bar_inf = P_bar_sum / n_x
            q_bar_inf = ca.mmax(ca.fabs(q_bar))
            gamma = 1 / ca.fmax(P_bar_inf, q_bar_inf)
            P_bar = gamma * P_bar
            q_bar = gamma * q_bar
            S_scale = ca.diag(delta) @ S_scale
            c_scale = gamma * c_scale
            M_bar[:n_x, :n_x] = P_bar
            M_bar[n_x:, :n_x] = A_bar
            M_bar[:n_x, n_x:] = A_bar.T

        S_scale = ca.diag(S_scale)
        D_scale = ca.diag(S_scale[:n_x])
        E_scale = ca.diag(S_scale[n_x:])
        P_scaled = c_scale * D_scale @ P_kkt @ D_scale
        A_scaled = E_scale @ A_kkt @ D_scale
        q_scaled = c_scale * D_scale @ c
        z_lb_scaled = E_scale @ z_lb
        z_ub_scaled = E_scale @ z_ub

        kkt_scaled = ca.MX(n_sys, n_sys)
        kkt_scaled[:n_x, :n_x] = P_scaled + sigma * ca.MX.eye(n_x)
        kkt_scaled[n_x:, :n_x] = A_scaled
        kkt_scaled[:n_x, n_x:] = A_scaled.T
        kkt_scaled[n_x:, n_x:] = -ca.diag(rho_norm_inv) * rho_bar
        kkt_scaled_triu = ca.triu(kkt_scaled)

        return ca.Function(
            f"ruiz_scaling_{self.problem.name}",
            [x, p, kkt_nz, sigma, rho_bar],
            [kkt_scaled_triu, q_scaled, D_scale, z_lb_scaled, z_ub_scaled],
            ["x", "p", "kkt_nz", "sigma", "rho_bar"],
            ["kkt_scaled_triu", "q_scaled", "D_scale", "z_lb_scaled", "z_ub_scaled"],
            self.problem.fn_opts,
        )

    def compute_ADMM_step(self):
        n_sys = self.problem.n_sys
        n_x = self.problem.n_x
        n_g = self.problem.n_g
        rho_norm = self.problem.rho_norm
        rho_norm_inv = self.problem.rho_norm_inv

        x_solve = ca.MX.sym("x_solve", n_sys, 1)
        x_k = ca.MX.sym("x_k", n_x, 1)
        y_k = ca.MX.sym("y_k", n_g, 1)
        z_k = ca.MX.sym("z_k", n_g, 1)
        z_lb = ca.MX.sym("z_lb", n_g, 1)
        z_ub = ca.MX.sym("z_ub", n_g, 1)
        alpha = ca.MX.sym("alpha", 1, 1)
        rho_bar = ca.MX.sym("rho_bar", 1, 1)
        x_substep = x_solve[:n_x]
        nu_substep = x_solve[n_x:]

        w = alpha * nu_substep + (1 - alpha) * y_k
        x_next = alpha * x_substep + (1 - alpha) * x_k
        z_next = ca.fmax(ca.fmin(rho_norm_inv * rho_bar * w + z_k, z_ub), z_lb)
        y_next = w + rho_norm * rho_bar * (z_k - z_next)
        return ca.Function(
            f"ADMM_step_{self.problem.name}",
            [x_solve, x_k, y_k, z_k, z_lb, z_ub, alpha, rho_bar],
            [x_next, y_next, z_next],
            ["x_solve", "x_k", "y_k", "z_k", "z_lb", "z_ub", "alpha", "rho_bar"],
            ["x_next", "y_next", "z_next"],
            self.problem.fn_opts,
        )

    def compute_KKT_vector(self):
        n_x = self.problem.n_x
        n_g = self.problem.n_g
        rho_norm_inv = self.problem.rho_norm_inv

        q = ca.MX.sym("q", n_x, 1)
        x_k = ca.MX.sym("x_k", n_x, 1)
        y_k = ca.MX.sym("y_k", n_g, 1)
        z_k = ca.MX.sym("z_k", n_g, 1)
        sigma = ca.MX.sym("sigma", 1, 1)
        rho_bar = ca.MX.sym("rho_bar", 1, 1)
        b_KKT = ca.vertcat(sigma * x_k - q, z_k - rho_bar * rho_norm_inv * y_k)
        return ca.Function(
            f"KKT_vector_{self.problem.name}",
            [q, x_k, y_k, z_k, sigma, rho_bar],
            [b_KKT],
            ["q", "x_k", "y_k", "z_k", "sigma", "rho_bar"],
            ["b_KKT"],
            self.problem.fn_opts,
        )
    
    def compute_LDL_functions(self, kkt_sparsity):
        kkt_sparsity_triu = ca.triu(kkt_sparsity)
        kkt_vec = ca.SX.sym('kkt_vec', kkt_sparsity.shape[0], 1)
        kkt_nz = ca.SX.sym("kkt_nz", kkt_sparsity.nnz(), 1)
        kkt_mat = ca.triu2symm(ca.triu(ca.SX(kkt_sparsity, kkt_nz)))
        [D, L, perm] = ca.ldl(kkt_mat, True)
        D_sym = ca.SX.sym('D_sym', D.nnz(), 1)
        L_sym = ca.SX.sym('L_sym', L.nnz(), 1)
        D_mat = ca.SX(D.sparsity(), D_sym)
        L_mat = ca.SX(L.sparsity(), L_sym)
        print(D.shape)
        print(D_mat.shape)
        print(L.shape)
        print(L_mat.shape)
        print(L_sym.sparsity().shape)
        kkt_soln = ca.ldl_solve(kkt_vec, D_mat, L_mat, perm)
        fn_fac = ca.Function(
            f"ldl_factorize_{self.problem.name}",
            [kkt_nz],
            [D, L, perm],
            ["kkt_nz"],
            ["D", "L", "perm"],
            self.problem.fn_opts,
        )
        fn_solve = ca.Function(
            f"ldl_solve_{self.problem.name}",
            [kkt_vec, D_sym, L_sym],
            [kkt_soln],
            ["kkt_vec", "D_nz", "L_nz"],
            ["kkt_soln"],
            self.problem.fn_opts,
        )
        return fn_fac, fn_solve





class RelaxedLogBackend(QPBackend):
    relaxed_log_cfg = {
        "alpha": 1.0,
        "max_iter": 5,
    }
    parallelized: bool = False
    cusadi_fns: dict[str, CusadiFunction] = {}

    def __init__(self, problem, qp_cfg=None):
        self.problem = problem
        self.relaxed_log_cfg.update(qp_cfg or {})

    def setup(self, x_init=None, p_init=None):
        self._build_qp_functions()
        if x_init is not None and p_init is not None:
            return self.fn_relaxed_log_soln(x_init, p_init)

    def solve_step(self, x_eval, p_eval, solve_method=None):
        dx = np.zeros((x_eval.shape[0], 1))
        if solve_method == "custom":
            raise ValueError("Only symbolic casadi LDL solver available.")
        return self.fn_relaxed_log_soln(x_eval, p_eval)

    def solve_parallelized(self, x_eval, p_eval):
        dim_x = x_eval.shape[1]
        if not self.parallelized:
            raise ModuleNotFoundError("Call .parallelize() before attempting parallel solve")

        if self.linsys_method == 'cudss':
            dx_soln = torch.zeros_like(x_eval)
            fn_KKT = self.cusadi_fns[f"relaxed_log_KKT_{self.problem.name}"]
            [A_KKT, b_KKT] = fn_KKT.evaluate([dx_soln, x_eval, p_eval])
            self.cudss_data.Ax[:, :] = A_KKT
            self.cudss_data.b[:, :] = b_KKT
            self.cudss_data.interface.factorizeNumeric()
            for _ in range(self.relaxed_log_cfg['max_iter']):
                self.cudss_data.interface.solveLinearSystem()
                soln_KKT = self.cudss_data.x[:, :]
                dx_soln += self.relaxed_log_cfg["alpha"] * soln_KKT[:, :dim_x]
                [_, b_KKT] = fn_KKT.evaluate([dx_soln, x_eval, p_eval])
                self.cudss_data.b[:, :] = b_KKT
            return dx_soln
        
        elif self.linsys_method == 'ldl':
            fn_eval = self.cusadi_fns[f'relaxed_log_soln_{self.problem.name}']
            return fn_eval.evaluate([x_eval, p_eval])[0] # dx solution to QP

    def parallelize(self,
                    linsys_method,
                    batch_size=4096,
                    precision='float',
                    dynamic_batching=True):
        from cusadi.parallelization import parallelize_functions
        solver_fns = self.build_parallel_fns(linsys_method)
        self.cusadi_fns = parallelize_functions(solver_fns,
                                                batch_size=batch_size,
                                                precision=precision,
                                                dynamic_batching=dynamic_batching)
        # Setup cudss interface with KKT sparsity pattern
        if linsys_method == "cudss":
            fn_KKT = self.cusadi_fns[f"relaxed_log_KKT_{self.problem.name}"].fn_casadi
            sparsity_KKT = fn_KKT.sparsity_out(0)
            self._setup_cudss_interface(sparsity_KKT, batch_size, precision)
        self.linsys_method = linsys_method
        self.parallelized = True

    def build_parallel_fns(self, linsys_method):
        self.linsys_method = linsys_method
        if not hasattr(self, "fn_relaxed_log_soln"):
            print('Symbolic LDL linear solver selected.')
            self._build_qp_functions()
        parallel_fns = [self.fn_relaxed_log_KKT]
        if linsys_method == 'ldl':
            parallel_fns = [self.fn_relaxed_log_soln]
        elif linsys_method == 'cudss':
            print('cuDSS linear solver chosen. No CusADi functions generated.')
        else:
            "Unknown symbolic linear solver option. Choose 'cudss' or 'ldl'."
        return parallel_fns

    def _build_qp_functions(self):
        qp_sym = self._unpack_symbolics(self.problem.opti)
        x = qp_sym['x']
        p = qp_sym['p']
        P = qp_sym['P']
        c = qp_sym['c']
        A_eq = qp_sym['A_eq']
        b_eq = qp_sym['b_eq']
        A_ineq = qp_sym['A_ineq']
        b_ineq = qp_sym['b_ineq']
        dim_x = x.shape[0]
        
        # Form relaxed barrier problem for inequality constraints
        dx = ca.MX.sym("dx", dim_x, 1)
        ineq_barrier = self._relaxed_barrier(x=(A_ineq @ dx - b_ineq),
                                                mu=0.2,
                                                delta=0.1)
        cost = 1/2 * dx.T @ P @ dx + c.T @ dx + ca.sum(ineq_barrier)
        P_newton, c_newton = ca.hessian(cost, dx)
        A_newton = A_eq
        b_newton = b_eq - A_eq @ dx

        # KKT system from equality constrained nonlinear problem
        print("Building Newton step function...")
        matrix_KKT, vector_KKT = self._newton_step_KKT_system(P_newton,
                                                              c_newton,
                                                              A_newton,
                                                              b_newton)
        self.fn_relaxed_log_KKT = ca.Function(
            f"relaxed_log_KKT_{self.problem.name}",
            [dx, x, p], [matrix_KKT, vector_KKT],
            ["dx", "x", "p"], ["matrix_KKT", "vector_KKT"],
            self.problem.fn_opts
        )

        # Solve system symbolically with casadi LDL
        print("Building iterated LDL solve function...")
        # dx_init = ca.SX.sym('dx_LDL', x.shape[0], 1)
        x_init = ca.SX.sym('x_LDL', x.shape[0], 1)
        p_init = ca.SX.sym('p_LDL', p.shape[0], 1)
        x_soln = ca.SX(dim_x, 1)
        # ! A_KKT doesn't change between QP iterations, not increasing mu
        A_KKT, b_KKT = self.fn_relaxed_log_KKT(x_soln, x_init, p_init)
        D, L, perm = ca.ldl(A_KKT, True)
        for _ in range(self.relaxed_log_cfg['max_iter']):
            soln_KKT = ca.ldl_solve(b_KKT, D, L, perm)
            x_soln += self.relaxed_log_cfg["alpha"] * soln_KKT[:dim_x]
            _, b_KKT = self.fn_relaxed_log_KKT(x_soln, x_init, p_init)

        self.fn_relaxed_log_soln = ca.Function(
            f"relaxed_log_soln_{self.problem.name}",
            [x_init, p_init], [x_soln],
            ["x", "p"], ["x_soln"],
            # [dx_init, x_init, p_init], [x_soln],
            # ["dx", "x", "p"], ["x_soln"],
            self.problem.fn_opts
        )
        print(f"Relaxed Log KKT function: {self.fn_relaxed_log_KKT.n_instructions()} instr.")
        print(f"Relaxed Log iterations function: {self.fn_relaxed_log_soln.n_instructions()} instr.")

    def _newton_step_KKT_system(self, P, c, A, b):
        dim_x = P.shape[0]
        dim_g = A.shape[0]
        A_KKT = ca.MX(dim_x + dim_g, dim_x + dim_g)
        A_KKT[:dim_x, :dim_x] = P
        A_KKT[:dim_x, dim_x:] = A.T
        A_KKT[dim_x:, :dim_x] = A
        A_KKT += 1e-8 * ca.MX.eye(dim_x + dim_g)
        b_KKT = ca.vertcat(-c, b)
        return A_KKT, b_KKT

    def _relaxed_barrier(self, x, mu=0.02, delta=0.1):
        log_barrier = -mu * ca.log(-x + 1e-8)
        quadratic_penalty = (mu / 2) * (((-x - 2 * delta) / delta) ** 2 - 1) \
                            - mu * ca.log(delta)
        barrier = ca.if_else(x <= -delta, log_barrier, quadratic_penalty)
        return barrier

    def _unpack_symbolics(self, opti):
        x, p, f, g = opti.x, opti.p, opti.f, opti.g
        g_lb, g_ub = opti.lbg, opti.ubg
        g_eq = g[self.problem.eq_idx]

        P, c = ca.hessian(f, x)
        P_triu = ca.triu(P)

        A_eq = ca.jacobian(g_eq, x)
        b_eq = -g[self.problem.eq_idx]
        A_rows = []
        b_terms = []
        for idx in self.problem.ineq_idx:
            if idx in self.problem.ineq_ub_idx:
                g_i = g[idx] - g_ub[idx]
                A_rows.append(ca.jacobian(g_i, x))
                b_terms.append(-g_i)
            if idx in self.problem.ineq_lb_idx:
                g_i = g_lb[idx] - g[idx]
                A_rows.append(ca.jacobian(g_i, x))
                b_terms.append(-g_i)
        if A_rows:
            A_ineq = ca.vertcat(*A_rows)
            b_ineq = ca.vertcat(*b_terms)
        else:
            A_ineq = ca.MX(0, self.problem.n_x)
            b_ineq = ca.MX(0, 1)
        return {'x': x,
                'p': p,
                'f': f,
                'g': g,
                'P': P,
                'P_triu': P_triu,
                'c': c,
                'A_eq': A_eq,
                'b_eq': b_eq,
                'A_ineq': A_ineq,
                'b_ineq': b_ineq}
