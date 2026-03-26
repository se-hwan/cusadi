import torch
import numpy as np
import casadi as ca
from cusadi.parallelization import CusadiFunction
from .base import QPBackend


class BarrierBackend(QPBackend):
    relaxed_log_cfg = {
        "alpha": 1.0,
        "max_iter": 20,
        "mu_init": 0.05,
        "delta_init": 0.1,
    }

    def __init__(self, problem, qp_cfg=None):
        self.problem = problem
        self.relaxed_log_cfg = dict(type(self).relaxed_log_cfg)
        self.relaxed_log_cfg.update(qp_cfg or {})
        self.parallelization_ready = False
        self.cusadi_fns = {}

    def setup(self, x_init=None, p_init=None):
        self._build_qp_functions()

    def solve_step(self, x_eval, p_eval):
        dim_x = x_eval.shape[0]
        dx_soln = np.zeros_like(x_eval)
        mu = self.relaxed_log_cfg['mu_init']
        delta = self.relaxed_log_cfg['delta_init']
        for _ in range(self.relaxed_log_cfg['max_iter']):
            b_KKT = self.fn_relaxed_log_KKT_vec(dx_soln, x_eval, p_eval, mu, delta)
            D_KKT, L_KKT = self.fn_relaxed_log_LDL_fac(dx_soln, x_eval, p_eval, mu, delta)
            soln_KKT = self.fn_relaxed_log_LDL_solve(b_KKT, D_KKT, L_KKT.nonzeros())
            dx_soln += self.relaxed_log_cfg["alpha"] * soln_KKT[:dim_x]
            delta /= 4.0
            mu /= 2.0
        return dx_soln.toarray().reshape(dim_x, 1)


    # ************** GPU PARALLELIZATION METHODS ****************** #
    def solve_parallelized(self, x_eval, p_eval):
        dim_x = x_eval.shape[1]
        if not self.parallelization_ready:
            raise ModuleNotFoundError("Call .parallelize() before attempting parallel solve")

        dx_soln = torch.zeros_like(x_eval)
        for _ in range(self.relaxed_log_cfg['max_iter']):
            [A_KKT, b_KKT] = self.fn_KKT.evaluate([dx_soln, x_eval, p_eval])
            if self.linsys_method == 'cudss':
                self.cudss_data.Ax[:, :] = A_KKT
                self.cudss_data.b[:, :] = b_KKT
                self.cudss_data.interface.factorizeNumeric()
                self.cudss_data.interface.solveLinearSystem()
                soln_KKT = self.cudss_data.x[:, :]
            elif self.linsys_method == 'ldl':
                [D, L, _] = self.fn_LDL_fac.evaluate([A_KKT])
                [soln_KKT] = self.fn_LDL_solve.evaluate([b_KKT, D, L])
            dx_soln += self.relaxed_log_cfg["alpha"] * soln_KKT[:, :dim_x]
        return dx_soln

    def _check_function_signature(self):
        import inspect, hashlib
        src = inspect.getsource(self.compute_KKT_system) + inspect.getsource(self.compute_KKT_factorization)
        self._fn_hash = hashlib.md5(src.encode()).hexdigest()
        hash_file = self.casadi_fns_dir / f"{self.problem.name}.hash"
        if not hash_file.exists():
            return False
        return hash_file.read_text().strip() == self._fn_hash

    def _load_parallel_functions(self):
        d = self.casadi_fns_dir
        name = self.problem.name
        self.fn_relaxed_log_KKT = ca.Function.load(str(d / f"relaxed_log_KKT_{name}.casadi"))  # type: ignore[arg-type]
        fns = [self.fn_relaxed_log_KKT]
        if self.linsys_method == "ldl":
            self.fn_relaxed_log_LDL_fac = ca.Function.load(str(d / f"relaxed_log_KKT_fac_{name}.casadi"))  # type: ignore[arg-type]
            self.fn_relaxed_log_LDL_solve = ca.Function.load(str(d / f"relaxed_log_KKT_solve_{name}.casadi"))  # type: ignore[arg-type]
            fns += [self.fn_relaxed_log_LDL_fac, self.fn_relaxed_log_LDL_solve]
        return fns

    def _save_parallel_functions(self):
        d = self.casadi_fns_dir
        name = self.problem.name
        self.fn_relaxed_log_KKT.save(str(d / f"relaxed_log_KKT_{name}.casadi"))
        if self.linsys_method == "ldl":
            self.fn_relaxed_log_LDL_fac.save(str(d / f"relaxed_log_KKT_fac_{name}.casadi"))
            self.fn_relaxed_log_LDL_solve.save(str(d / f"relaxed_log_KKT_solve_{name}.casadi"))
        (d / f"{name}.hash").write_text(self._fn_hash)

    def _build_parallel_functions(self, linsys_method="ldl"):
        lin_solver = self.linsys_method if linsys_method is None else linsys_method
        qp_sym = self._unpack_symbolics(self.problem.opti)
        
        # KKT system from equality constrained nonlinear problem
        self.fn_relaxed_log_KKT = self.compute_KKT_system(qp_sym['x'],
                                                          qp_sym['p'],
                                                          qp_sym['P'],
                                                          qp_sym['c'],
                                                          qp_sym['A_eq'],
                                                          qp_sym['b_eq'],
                                                          qp_sym['A_ineq'],
                                                          qp_sym['b_ineq'])
        parallel_fns = [self.fn_relaxed_log_KKT]
        if lin_solver == "ldl":
            self.fn_relaxed_log_LDL_fac, self.fn_relaxed_log_LDL_solve = \
                self.compute_KKT_factorization(self.fn_relaxed_log_KKT.sparsity_out(0))
            parallel_fns.append(self.fn_relaxed_log_LDL_fac)
            parallel_fns.append(self.fn_relaxed_log_LDL_solve)
        return parallel_fns

    def _setup_parallel_evaluation(self):
        pass

    def set_cusadi_functions(self, cusadi_fns):
        self.fn_KKT = cusadi_fns[f"relaxed_log_KKT_{self.problem.name}"]
        if self.linsys_method == "cudss":
            # Setup cudss interface with KKT sparsity pattern
            sparsity_KKT = self.fn_KKT.sparsity_out(0)
            self._setup_cudss_interface(sparsity_KKT, self.batch_size, self.precision)
        elif self.linsys_method == "ldl":
            self.fn_LDL_fac = cusadi_fns[f"relaxed_log_KKT_fac_{self.problem.name}"]
            self.fn_LDL_solve = cusadi_fns[f"relaxed_log_KKT_solve_{self.problem.name}"]
        self.parallelization_ready = True
        return self.cusadi_fns


    # ************* SYMBOLIC SOLVER EXPRESSIONS *********** #
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

    def compute_KKT_system(self, x, par, P, c, A_eq, b_eq, A_ineq, b_ineq):
        dim_x = P.shape[0]
        dx = ca.MX.sym("dx", dim_x, 1)
        mu = ca.MX.sym("mu", 1, 1)
        delta = ca.MX.sym("delta", 1, 1)
        ineq_barrier = self._relaxed_barrier(x=(A_ineq @ dx - b_ineq),
                                                mu=mu,
                                                delta=delta)
        cost = 1/2 * dx.T @ P @ dx + c.T @ dx + ca.sum(ineq_barrier)
        P_newton, c_newton = ca.hessian(cost, dx)
        A_newton = A_eq
        b_newton = b_eq - A_eq @ dx

        dim_eq = A_eq.shape[0]
        A_KKT = ca.MX(dim_x + dim_eq, dim_x + dim_eq)
        A_KKT[:dim_x, :dim_x] = P_newton
        A_KKT[:dim_x, dim_x:] = A_newton.T
        A_KKT[dim_x:, :dim_x] = A_newton
        A_KKT += 1e-6 * ca.MX.eye(dim_x + dim_eq)
        A_KKT_triu = ca.triu(A_KKT)
        b_KKT = ca.vertcat(-c_newton, b_newton)
        return ca.Function(
            f"relaxed_log_KKT_{self.problem.name}",
            [dx, x, par, mu, delta], [A_KKT_triu, b_KKT],
            ["dx", "x", "p", "mu", "delta"], ["A_KKT", "b_KKT"],
            self.problem.fn_opts
        )

    def _relaxed_barrier(self, x, mu=0.02, delta=0.1):
        log_barrier = -mu * ca.log(-x + 1e-8)
        quadratic_penalty = (mu / 2) * (((-x - 2 * delta) / delta) ** 2 - 1) \
                            - mu * ca.log(delta)
        barrier = ca.if_else(x <= -delta, log_barrier, quadratic_penalty)
        return barrier

    def compute_KKT_factorization(self, kkt_sparsity_triu):
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
            f"relaxed_log_KKT_fac_{self.problem.name}",
            [kkt_nz],
            [D, L, perm],
            ["kkt_nz"],
            ["D", "L", "perm"],
            self.problem.fn_opts,
        )
        fn_solve = ca.Function(
            f"relaxed_log_KKT_solve_{self.problem.name}",
            [kkt_vec, D_sym, L_sym],
            [kkt_soln],
            ["kkt_vec", "D_nz", "L_nz"],
            ["kkt_soln"],
            self.problem.fn_opts,
        )
        return fn_fac, fn_solve
    
        # # Solve system symbolically with casadi LDL
        # dx_init = ca.SX.sym('dx_LDL', x.shape[0], 1)
        # x_init = ca.SX.sym('x_LDL', x.shape[0], 1)
        # p_init = ca.SX.sym('p_LDL', p.shape[0], 1)
        # mu = ca.SX.sym("mu", 1, 1)
        # delta = ca.SX.sym("delta", 1, 1)
        # A_KKT_triu, _ = self.fn_relaxed_log_KKT(dx_init, x_init, p_init, mu, delta)
        # A_KKT = ca.triu2symm(A_KKT_triu)
        # D, L, perm = ca.ldl(A_KKT, True)
        # self.fn_relaxed_log_LDL_fac = ca.Function(
        #     f"relaxed_log_KKT_fac_{self.problem.name}",
        #     [dx_init, x_init, p_init, mu, delta], [D, L],
        #     ["dx", "x", "p", "mu", "delta"], ["D", "L"],
        #     self.problem.fn_opts
        # )
        # b_sym = ca.SX.sym('b_sym', dim_sys, 1)
        # D_sym = ca.SX.sym('D_sym', D.nnz(), 1)
        # L_sym = ca.SX.sym('L_sym', L.nnz(), 1)
        # D_mat = ca.SX(D.sparsity(), D_sym)
        # L_mat = ca.SX(L.sparsity(), L_sym)
        # soln_KKT = ca.ldl_solve(b_sym, D_mat, L_mat, perm)

        # self.fn_relaxed_log_LDL_solve = ca.Function(
        #     f"relaxed_log_KKT_solve_{self.problem.name}",
        #     [b_sym, D_sym, L_sym], [soln_KKT],
        #     ["b", "D", "L"], ["soln_KKT"],
        #     self.problem.fn_opts
        # )

    # ************* UTILITY METHODS *********** #
    def get_kkt_sparsity(self):
        if not hasattr(self, "fn_relaxed_log_KKT_mat"):
            self._build_parallel_functions('ldl')
        kkt_triu_sparsity = self.fn_relaxed_log_KKT.sparsity_out(0)
        kkt_sparsity = ca.triu2symm(ca.SX(kkt_triu_sparsity))
        return kkt_sparsity