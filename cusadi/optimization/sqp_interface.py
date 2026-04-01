import casadi as ca
from .qp_backends import QPBackend, ADMMBackend, BarrierBackend


class SQPInterface:
    line_search = None
    max_iter = 1
    alpha = 1.0

    def __init__(self, opti_problem, qp_solver="osqp", sqp_cfg=None, qp_cfg=None,
                 x_init=None, p_init=None):
        self.problem = opti_problem
        self.qp_solver = qp_solver

        sqp_cfg = sqp_cfg or {}
        for key, value in sqp_cfg.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.fn_qp_matrices = None
        self.qp_backend: QPBackend

        self._setup_qp_backend(self.problem, qp_cfg, x_init, p_init)

    # TODO: resolve this similar behavior
    def solve(self, x_eval, p_eval, solve_method=None):
        x_soln = x_eval.copy()
        for _ in range(self.max_iter):
            dx = self.qp_backend.solve_step(x_soln, p_eval, solve_method)
            x_soln += self.alpha * dx # ? Line search functionality here
        return x_soln

    def solve_parallelized(self, x_eval, p_eval):
        x_soln = x_eval.clone()
        for _ in range(self.max_iter):
            dx = self.qp_backend.solve_parallelized(x_soln, p_eval)
            x_soln += self.alpha * dx
        return x_soln

    def setup_parallelization(self,
                              linsys_method,
                              batch_size=4096,
                              precision='float',
                              dynamic_batching=True):
        return self.qp_backend.setup_parallelization(linsys_method,
                                                     batch_size,
                                                     precision,
                                                     dynamic_batching)

    def set_cusadi_functions(self, cusadi_fns):
        return self.qp_backend.set_cusadi_functions(cusadi_fns)

    def _setup_qp_backend(self, problem, qp_cfg,  x_init, p_init):
        if self.qp_solver == "osqp":
            self.qp_backend = ADMMBackend(problem, qp_cfg)
        elif self.qp_solver == "relaxed_log":
            self.qp_backend = BarrierBackend(problem, qp_cfg)
        else:
            raise ValueError(f"Unknown solver: {self.qp_solver}")
        self.qp_backend.setup(x_init=x_init, p_init=p_init)