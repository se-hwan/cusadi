import osqp
import torch
import numpy as np
import casadi as ca

from cusadi.parallelization import CusadiFunction
from .base import QPBackend

class ADMMBackend(QPBackend):
    # OSQP default settings
    osqp_cfg = {"rho": 0.1,
                "verbose": False,
                "adaptive_rho": False,
                "max_iter": 25,
                "scaling": 2,
                "check_termination": 100,
                "sigma": 1e-6,
                "alpha": 1.6,
                "warm_starting": False}

    def __init__(self, opti_problem, qp_cfg=None):
        self.problem = opti_problem
        qp_cfg = qp_cfg or {}
        self.osqp_cfg = dict(type(self).osqp_cfg)
        self.osqp_cfg.update(qp_cfg)
        self.parallelization_ready = False
        self.cusadi_fns = {}
        self.qp_setup = False
        self.solver = osqp.OSQP()

        # Normalized rho vector in OSQP
        self.rho_norm = np.ones(self.problem.n_g)
        self.rho_norm[self.problem.eq_idx] = 1e3*self.rho_norm[self.problem.eq_idx]
        self.rho_norm_inv = 1/self.rho_norm.copy()
        print("OSQP backend initialized")

    def setup(self, x_init=None, p_init=None):
        self.fn_osqp_matrices = self.build_OSQP_function()
        if x_init is not None and p_init is not None:
            P_triu, c, A, lb, ub = self.fn_osqp_matrices(x_init, p_init)
            self.solver.setup(P_triu.tocsc(),
                              c.toarray(),
                              A.tocsc(),
                              lb.toarray(),
                              ub.toarray(),
                              **self.osqp_cfg)
            self.qp_setup = True

    def update_settings(self, osqp_settings):
        self.osqp_cfg.update(osqp_settings)
        if self.qp_setup:
            self.solver.update(**osqp_settings)

    def solve_step(self, x_eval, p_eval, solve_method=None):
        if solve_method is None:
            return self.solve_with_osqp(x_eval, p_eval)
        elif solve_method == 'custom':
            return self.solve_with_custom(x_eval, p_eval)

    def solve_with_osqp(self, x_eval, p_eval):
        if not self.qp_setup:
            self.setup(x_eval, p_eval)
        else:
            P_triu, c, A, lb, ub = self.fn_osqp_matrices(x_eval, p_eval)
            self.solver.update(Px=P_triu.tocsc().data,
                               Ax=A.tocsc().data,
                               q=c.toarray(),
                               l=lb.toarray(),
                               u=ub.toarray())
        soln = np.array([self.solver.solve().x])
        return soln.reshape(x_eval.shape[0], 1)

    def solve_with_custom(self, x_eval, p_eval):
        if not hasattr(self, "fn_KKT_matrix"):
            self._build_parallel_functions('ldl')

        x_k = np.zeros((self.problem.n_x, 1))
        y_k = np.zeros((self.problem.n_g, 1))
        z_k = np.zeros((self.problem.n_g, 1))
        
        A_KKT = self.fn_KKT_matrix(x_eval, p_eval)
        [A_KKT_scaled_triu, q_scaled, D_scaled, lb_scaled, ub_scaled] = \
            self.fn_ruiz_scaling(A_KKT.nonzeros(), x_eval, p_eval, self.osqp_cfg['sigma'], self.osqp_cfg['rho'])
        D_ldl, L_ldl, _ = self.fn_LDL_fac(A_KKT_scaled_triu.nonzeros())
        for _ in range(self.osqp_cfg['max_iter']):
            b_KKT = self.fn_KKT_vector(q_scaled,
                                         x_k,
                                         y_k,
                                         z_k,
                                         self.osqp_cfg['sigma'],
                                         self.osqp_cfg['rho'])
            soln_KKT = self.fn_LDL_solve(b_KKT, D_ldl, L_ldl.nonzeros())
            x_next, y_next, z_next = self.fn_ADMM_step(
                soln_KKT,
                x_k,
                y_k,
                z_k,
                lb_scaled,
                ub_scaled,
                self.osqp_cfg['alpha'],
                self.osqp_cfg['rho'])
            x_k[:] = x_next
            y_k[:] = y_next
            z_k[:] = z_next
        return (D_scaled * x_k).toarray()

    def build_OSQP_function(self):
        qp_sym = self._unpack_symbolics(self.problem.opti)
        return ca.Function(
            f"qp_matrices_{self.problem.name}",
            [qp_sym["x"], qp_sym["p"]],
            [qp_sym["P_triu"], qp_sym["c"], qp_sym["A"], qp_sym["z_lb"], qp_sym["z_ub"]],
            ["x", "p"],
            ["P_triu", "c", "A", "z_lb", "z_ub"],
            self.problem.fn_opts,
        )


    # ************** GPU PARALLELIZATION METHODS ****************** #
    def solve_parallelized(self, x_eval, p_eval, kkt_params=None):
        if not self.parallelization_ready:
            raise ModuleNotFoundError("Call .parallelize() before attempting parallel solve")
        
        input_batch_size = x_eval.shape[0]
        static_batching = not getattr(self.gpu_fn_KKT, "dynamic_batching", True)
        if static_batching and input_batch_size != self.gpu_fn_KKT.batch_size:
            raise RuntimeError(
                "Static-batch solver received a different number of environments than the "
                "compiled kernel expects. "
                f"Kernel batch size is {self.gpu_fn_KKT.batch_size}, runtime batch size is {input_batch_size}."
            )
        if kkt_params is not None:
            self.rho_tensor = kkt_params['rho']
            self.sigma_tensor = kkt_params['sigma']
            self.alpha_tensor = kkt_params['alpha']

        self.gpu_x_k.zero_()
        self.gpu_y_k.zero_()
        self.gpu_z_k.zero_()
        [A_KKT] = self.gpu_fn_KKT.evaluate([x_eval, p_eval])
        [A_KKT_scaled_triu, q_scaled, D_scaled, lb_scaled, ub_scaled] = \
            self.gpu_fn_scaling.evaluate(
                [A_KKT, x_eval, p_eval, self.sigma_tensor, self.rho_tensor])
        if self.linsys_method == 'cudss':
            self.cudss_data.Ax[:, :] = A_KKT_scaled_triu
            self.cudss_data.interface.factorizeNumeric()
        elif self.linsys_method == 'ldl':
            [D_ldl, L_ldl, _] = self.gpu_fn_ldl_fac.evaluate([A_KKT_scaled_triu])
        for _ in range(self.osqp_cfg['max_iter']):
            [b_KKT] = self.gpu_fn_KKT_vector.evaluate([q_scaled,
                                                        self.gpu_x_k,
                                                        self.gpu_y_k,
                                                        self.gpu_z_k,
                                                        self.sigma_tensor,
                                                        self.rho_tensor])
            if self.linsys_method == 'cudss':
                self.cudss_data.b[:, :] = b_KKT
                self.cudss_data.interface.solveLinearSystem()
                soln_KKT = self.cudss_data.x.clone()
            elif self.linsys_method == 'ldl':
                [soln_KKT] = self.gpu_fn_ldl_solve.evaluate([b_KKT, D_ldl, L_ldl])
            [x_next, y_next, z_next] = self.gpu_fn_ADMM_step.evaluate([
                soln_KKT,
                self.gpu_x_k,
                self.gpu_y_k,
                self.gpu_z_k,
                lb_scaled,
                ub_scaled,
                self.alpha_tensor,
                self.rho_tensor])
            self.gpu_x_k[:, :] = x_next
            self.gpu_y_k[:, :] = y_next
            self.gpu_z_k[:, :] = z_next
        return D_scaled * self.gpu_x_k

    def _check_function_signature(self):
        import inspect, hashlib
        src = (inspect.getsource(self.compute_KKT_matrix) +
               inspect.getsource(self.compute_ruiz_scaling) +
               inspect.getsource(self.compute_KKT_vector) +
               inspect.getsource(self.compute_ADMM_step) +
               inspect.getsource(self.compute_LDL_functions))
        self._fn_hash = hashlib.md5(src.encode()).hexdigest()
        hash_file = self.casadi_fns_dir / f"{self.problem.name}.hash"
        if not hash_file.exists():
            return False
        return hash_file.read_text().strip() == self._fn_hash

    def _load_parallel_functions(self):
        d = self.casadi_fns_dir
        name = self.problem.name
        self.fn_KKT_matrix = ca.Function.load(str(d / f"KKT_matrix_{name}.casadi"))  # type: ignore[arg-type]
        self.fn_ruiz_scaling = ca.Function.load(str(d / f"ruiz_scaling_{name}.casadi"))  # type: ignore[arg-type]
        self.fn_KKT_vector = ca.Function.load(str(d / f"KKT_vector_{name}.casadi"))  # type: ignore[arg-type]
        self.fn_ADMM_step = ca.Function.load(str(d / f"ADMM_step_{name}.casadi"))  # type: ignore[arg-type]
        fns = [self.fn_KKT_matrix, self.fn_ruiz_scaling, self.fn_KKT_vector, self.fn_ADMM_step]
        if self.linsys_method == "ldl":
            self.fn_LDL_fac = ca.Function.load(str(d / f"ldl_factorize_{name}.casadi"))  # type: ignore[arg-type]
            self.fn_LDL_solve = ca.Function.load(str(d / f"ldl_solve_{name}.casadi"))  # type: ignore[arg-type]
            fns += [self.fn_LDL_fac, self.fn_LDL_solve]
        return fns

    def _save_parallel_functions(self):
        d = self.casadi_fns_dir
        name = self.problem.name
        self.fn_KKT_matrix.save(str(d / f"KKT_matrix_{name}.casadi"))
        self.fn_ruiz_scaling.save(str(d / f"ruiz_scaling_{name}.casadi"))
        self.fn_KKT_vector.save(str(d / f"KKT_vector_{name}.casadi"))
        self.fn_ADMM_step.save(str(d / f"ADMM_step_{name}.casadi"))
        if self.linsys_method == "ldl":
            self.fn_LDL_fac.save(str(d / f"ldl_factorize_{name}.casadi"))
            self.fn_LDL_solve.save(str(d / f"ldl_solve_{name}.casadi"))
        (d / f"{name}.hash").write_text(self._fn_hash)

    def _build_parallel_functions(self, linsys_method):
        lin_solver = self.linsys_method if linsys_method is None else linsys_method
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
        if lin_solver == 'ldl':
            print('Symbolic LDL linear solver selected.')
            self.fn_LDL_fac, self.fn_LDL_solve = self.compute_LDL_functions(
                self.fn_ruiz_scaling.sparsity_out(0))
            parallel_fns.append(self.fn_LDL_fac)
            parallel_fns.append(self.fn_LDL_solve)
        elif lin_solver == 'cudss':
            print('cuDSS linear solver chosen. No CasADi functions generated.')
        else:
            "Unknown symbolic linear solver option. Choose 'cudss' or 'ldl'."
        for f in parallel_fns:
            print(f"{f.name()}: {f.n_instructions()} instructions.")
        return parallel_fns

    def _setup_parallel_evaluation(self):
        # Instantiate variables for GPU solves
        torch_precision = torch.float if self.precision == 'float' else torch.double
        self.gpu_x_k = torch.zeros(self.batch_size,
                                   self.problem.n_x,
                                   device='cuda',
                                   dtype=torch_precision)
        self.gpu_y_k = torch.zeros(self.batch_size,
                                   self.problem.n_g,
                                   device='cuda',
                                   dtype=torch_precision)
        self.gpu_z_k = torch.zeros(self.batch_size,
                                   self.problem.n_g,
                                   device='cuda',
                                   dtype=torch_precision)
        torch_ones = torch.ones(self.batch_size, 1, dtype=torch_precision, device='cuda')
        self.rho_tensor = self.osqp_cfg['rho'] * torch_ones
        self.sigma_tensor = self.osqp_cfg['sigma'] * torch_ones
        self.alpha_tensor = self.osqp_cfg['alpha'] * torch_ones

    def set_cusadi_functions(self, cusadi_fns):
        # Store functions to be used in GPU-parallized solves
        self.gpu_fn_KKT = cusadi_fns[f"KKT_matrix_{self.problem.name}"]
        self.gpu_fn_scaling = cusadi_fns[f"ruiz_scaling_{self.problem.name}"]
        self.gpu_fn_KKT_vector = cusadi_fns[f"KKT_vector_{self.problem.name}"]
        self.gpu_fn_ADMM_step = cusadi_fns[f"ADMM_step_{self.problem.name}"]
        if self.linsys_method == "cudss":
            sparsity_KKT = self.gpu_fn_scaling.fn_casadi.sparsity_out(0)
            self._setup_cudss_interface(sparsity_KKT, self.batch_size, self.precision)
        elif self.linsys_method == "ldl":
            self.gpu_fn_ldl_fac = self.cusadi_fns[f"ldl_factorize_{self.problem.name}"]
            self.gpu_fn_ldl_solve = self.cusadi_fns[f"ldl_solve_{self.problem.name}"]
        else:
            raise ValueError(f"Unknown linsys method: {self.linsys_method}")
        self.parallelization_ready = True
        return self.cusadi_fns


    # ************* SYMBOLIC SOLVER EXPRESSIONS *********** #
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
        n_g = self.problem.n_g

        kkt_triu_sparsity = ca.triu(kkt_sparsity)
        kkt_nz = ca.MX.sym("kkt_nz", kkt_triu_sparsity.nnz(), 1)
        kkt_sym = ca.triu2symm(ca.MX(kkt_triu_sparsity, kkt_nz))
        sigma = ca.MX.sym("sigma", 1, 1)
        rho_bar = ca.MX.sym("rho_bar", 1, 1)
        rho = self.rho_norm * rho_bar
        rho_inv = 1 / rho

        S_scale = ca.MX.eye(n_sys)
        delta = ca.MX(n_sys, 1)
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

        for _ in range(self.osqp_cfg['scaling']):
            for j in range(n_sys):
                M_j_inf = ca.mmax(ca.fabs(M_bar[:, j]))
                denom = ca.sqrt(M_j_inf)
                delta[j] = ca.if_else(denom <= 0, 1, 1/denom)
            D = ca.diag(delta[:n_x])
            E = ca.diag(delta[n_x:])
            P_bar = D @ P_bar @ D
            A_bar = E @ A_bar @ D
            q_bar = D @ q_bar

            P_inf_norm = ca.vertcat(*[ca.mmax(ca.fabs(P_bar[:, j])) for j in range(n_x)])
            # P_bar_inf = ca.sum(P_inf_norm) / n_x
            P_bar_inf = ca.mmax(P_inf_norm) 
            q_bar_inf = ca.mmax(ca.fabs(q_bar))
            gamma = 1 / (ca.fmax(P_bar_inf, q_bar_inf))
            P_bar = gamma * P_bar
            q_bar = gamma * q_bar
            S_scale = ca.diag(delta) @ S_scale
            c_scale = gamma * c_scale
            M_bar = ca.MX(n_sys, n_sys)
            M_bar[:n_x, :n_x] = P_bar
            M_bar[n_x:, :n_x] = A_bar
            M_bar[:n_x, n_x:] = A_bar.T

        S_scale = ca.diag(S_scale)
        D_scale = ca.diag(S_scale[:n_x])
        E_scale = ca.diag(S_scale[n_x:])
        P_scaled = c_scale * D_scale @ P_kkt @ D_scale
        A_scaled = E_scale @ A_kkt @ D_scale
        q_scaled = c_scale @ D_scale @ c
        z_lb_scaled = E_scale @ z_lb
        z_ub_scaled = E_scale @ z_ub

        kkt_scaled = ca.MX(n_sys, n_sys)
        kkt_scaled[:n_x, :n_x] = P_scaled + sigma * ca.MX.eye(n_x)
        kkt_scaled[n_x:, :n_x] = A_scaled
        kkt_scaled[:n_x, n_x:] = A_scaled.T
        kkt_scaled[n_x:, n_x:] = -ca.diag(rho_inv)
        kkt_scaled_triu = ca.triu(kkt_scaled)
        D_scale = ca.diag(D_scale)

        return ca.Function(
            f"ruiz_scaling_{self.problem.name}",
            [kkt_nz, x, p, sigma, rho_bar],
            [kkt_scaled_triu, q_scaled, D_scale, z_lb_scaled, z_ub_scaled],
            ["kkt_nz", "x", "p", "sigma", "rho_bar"],
            ["kkt_scaled_triu", "q_scaled", "D_scale", "z_lb_scaled", "z_ub_scaled"],
            self.problem.fn_opts,
        )

    def compute_ADMM_step(self):
        n_sys = self.problem.n_sys
        n_x = self.problem.n_x
        n_g = self.problem.n_g

        x_solve = ca.MX.sym("x_solve", n_sys, 1)
        x_k = ca.MX.sym("x_k", n_x, 1)
        y_k = ca.MX.sym("y_k", n_g, 1)
        z_k = ca.MX.sym("z_k", n_g, 1)
        z_lb = ca.MX.sym("z_lb", n_g, 1)
        z_ub = ca.MX.sym("z_ub", n_g, 1)
        alpha = ca.MX.sym("alpha", 1, 1)
        rho_bar = ca.MX.sym("rho_bar", 1, 1)
        rho = self.rho_norm * rho_bar
        rho_inv = 1 / rho
        x_substep = x_solve[:n_x]
        nu_substep = x_solve[n_x:]

        w = alpha * nu_substep + (1 - alpha) * y_k
        x_next = alpha * x_substep + (1 - alpha) * x_k
        z_next = ca.fmax(ca.fmin(rho_inv * w + z_k, z_ub), z_lb)
        y_next = w + rho * (z_k - z_next)

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

        q = ca.MX.sym("q", n_x, 1)
        x_k = ca.MX.sym("x_k", n_x, 1)
        y_k = ca.MX.sym("y_k", n_g, 1)
        z_k = ca.MX.sym("z_k", n_g, 1)
        sigma = ca.MX.sym("sigma", 1, 1)
        rho_bar = ca.MX.sym("rho_bar", 1, 1)
        rho = self.rho_norm * rho_bar
        rho_inv = 1 / rho

        b_KKT = ca.vertcat(sigma * x_k - q, z_k - rho_inv * y_k)
        return ca.Function(
            f"KKT_vector_{self.problem.name}",
            [q, x_k, y_k, z_k, sigma, rho_bar],
            [b_KKT],
            ["q", "x_k", "y_k", "z_k", "sigma", "rho_bar"],
            ["b_KKT"],
            self.problem.fn_opts,
        )
    
    def compute_LDL_functions(self, kkt_sparsity_triu):
        kkt_vec = ca.SX.sym('kkt_vec', kkt_sparsity_triu.shape[0], 1)
        kkt_nz = ca.SX.sym("kkt_nz", kkt_sparsity_triu.nnz(), 1)
        kkt_mat = ca.triu2symm(ca.SX(kkt_sparsity_triu, kkt_nz))
        [D, L, perm] = ca.ldl(kkt_mat, True)
        D_sym = ca.SX.sym('D_sym', D.nnz(), 1)
        L_sym = ca.SX.sym('L_sym', L.nnz(), 1)
        D_mat = ca.SX(D.sparsity(), D_sym)
        L_mat = ca.SX(L.sparsity(), L_sym)
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
    

    # ************* UTILITY METHODS *********** #
    def get_kkt_sparsity(self):
        if not hasattr(self, "fn_ruiz_scaling"):
            self._build_parallel_functions('ldl')
        kkt_triu_sparsity = self.fn_ruiz_scaling.sparsity_out(0)
        kkt_sparsity = ca.triu2symm(ca.SX(kkt_triu_sparsity))
        return kkt_sparsity