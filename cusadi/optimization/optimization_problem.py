import casadi as ca
import numpy as np
import yaml
import matplotlib.pyplot as plt
from dataclasses import dataclass
from cusadi.utils.symbolic import *
from .solver_registry import get_solver


@dataclass
class Variable:
    sym: ca.SX | ca.MX
    dim: tuple
    idx: np.ndarray

@dataclass
class Parameter:
    sym: ca.SX | ca.MX
    dim: tuple
    dim_flat: int
    idx: np.ndarray
    init: np.ndarray
    init_vec: np.ndarray


class OptimizationProblem:
    opti = ca.Opti()
    name = "problem"
    n_x = 0     # Num. decision variables
    n_g = 0     # Num. constraints
    n_sys = 0   # Dim. of full system (n_x + n_g)
    n_par = 0   # Num. parameters

    cost = None
    variables = {}
    parameters = {}
    eq_constraints = {}
    ineq_constraints = {}
    solver = ""
    post_processed = False

    def __init__(self, name=None):
        # Clear all previous definitions
        self.opti = ca.Opti()
        self.name = name
        self.n_x = 0
        self.n_g = 0
        self.n_sys = 0
        self.n_par = 0
        self.cost = None
        self.variables = {}
        self.parameters = {}
        self.eq_constraints = {}
        self.ineq_constraints = {}
        self.fn_opts = get_function_options()
        self.cg_opts = get_codegen_options()

    def __repr__(self):
        return self.get_summary()

    def __str__(self):
        return self.get_summary()
    
    def get_summary(self):
        summary_text =   "OptimizationProblem instance: \n"
        summary_text += f"  Number of decision variables:     {self.n_x}\n"
        summary_text += f"  Number of parameters:             {self.n_par}\n"
        summary_text += f"  Number of constraints:            {self.n_g}\n"
        summary_text += f"  Dimension of KKT matrix:          {self.n_sys}\n"
        summary_text += f"  Sets of equality constraints:     {len(self.eq_constraints)}\n"
        summary_text += f"  Sets of inequality constraints:   {len(self.ineq_constraints)}"
        return summary_text

    # ----------------------  FORMULATION  ---------------------------- #
    def configure_problem(self, yaml_file):
        with open(yaml_file, 'r') as f:
            ocp_cfg = yaml.safe_load(f)
        # Set solver options
        solver_opts = ocp_cfg.get('solver_options', {})
        if solver_opts:
            print("Setting solver options:")
            for key, value in solver_opts.items():
                print(f"  {key}: {value}")
            self.opti.solver('ipopt', solver_opts)
        else:
            self.opti.solver('ipopt')

    def set_objective(self, expr):
        self.opti.minimize(expr)
        self.cost = expr

    def add_variable(self, dim_1, dim_2, label):
        assert label not in self.variables, \
            f"Variable with label '{label}' already exists."
        opti_var = self.opti.variable(dim_1, dim_2)
        idx_start = self.n_x
        idx_end = self.n_x + dim_1*dim_2
        self.n_x += dim_1*dim_2

        self.variables[label] = Variable(
            sym=opti_var,
            dim=(dim_1*dim_2),
            idx=np.arange(idx_start, idx_end)
        )
        return opti_var

    def add_parameter(self, dim_1, dim_2, label, init_value=None):
        assert label not in self.parameters, \
            f"Parameter with label '{label}' already exists."
        opti_par = self.opti.parameter(dim_1, dim_2)
        idx_start = self.n_par
        idx_end = self.n_par + dim_1*dim_2
        self.n_par += dim_1*dim_2

        if init_value is None:
            init_value = np.zeros((dim_1, dim_2))
        self.opti.set_value(opti_par, init_value)

        self.parameters[label] = Parameter(
            sym=opti_par,
            dim=(dim_1, dim_2),
            dim_flat=(dim_1*dim_2),
            idx=np.arange(idx_start, idx_end),
            init=init_value,
            init_vec=init_value.flatten(order='F')
        )
        return opti_par

    def set_parameter(self, label, init_value):
        self.parameters[label].init = init_value
        return self.parameters[label]

    def get_parameter_vec(self):
        for param in self.parameters.values():
            self.opti.set_value(param.sym, param.init)
        return self.opti.value(self.opti.p)

    # ? Add lagrange multipliers tracking?
    def add_equality_constraint(self, expr, label):
        assert label not in self.eq_constraints, \
            f"Equality constraint with label '{label}' already exists."
        # Add constraint and log to dictionary
        idx_start = self.opti.ng
        self.opti.subject_to(expr == ca.MX(expr.shape[0], 1))
        self.n_g = self.opti.ng
        g_OCP = self.opti.g
        g_expr = g_OCP[idx_start:self.n_g]

        self.eq_constraints[label] = {
            'expr': g_expr,
            'dim': self.n_g - idx_start,
            'idx': np.arange(idx_start, self.n_g),
        }

    # ? Add lagrange multipliers tracking?
    def add_inequality_constraint(self, expr, label, ub=None, lb=None):
        assert label not in self.ineq_constraints, \
            f"Inequality constraint with label '{label}' already exists."
        assert not (ub is None and lb is None), \
            f"Provide at least one of least one of 'ub' or 'lb' for constraint '{label}'."
        if lb is not None:
            expr = lb <= expr
        if ub is not None:
            expr = expr <= ub
        idx_start = self.opti.ng
        self.opti.subject_to(expr)
        self.n_g = self.opti.ng
        g_OCP = self.opti.g
        g_expr = g_OCP[idx_start:self.n_g]

        self.ineq_constraints[label] = {
            'expr': g_expr,
            'dim': self.n_g - idx_start,
            'idx': np.arange(idx_start, self.n_g),
            'lb': True if lb is not None else False,
            'ub': True if ub is not None else False
        }

    def post_process(self):
        assert self.n_x == self.opti.nx, \
            f"Number of decision variables mismatch: {self.n_x} (logged) \
              vs {self.opti.nx} (opti stack)."
        assert self.n_par == self.opti.np, \
            f"Number of parameters mismatch: {self.n_par} (logged) vs \
              {self.opti.p} (opti stack). \
              Redundant parameters may have been defined."
        
        print("Post-processing problem...")
        self.n_sys = self.n_x + self.n_g
        self.ineq_idx = [i.item() for v in self.ineq_constraints.values()
                    for i in v['idx']]
        self.eq_idx = [i.item() for v in self.eq_constraints.values()
                  for i in v['idx']]
        self.ineq_ub_idx = [i.item() for v in self.ineq_constraints.values()
                            if v['ub'] for i in v['idx']]
        self.ineq_lb_idx = [i.item() for v in self.ineq_constraints.values()
                            if v['lb'] for i in v['idx']]
        self.post_processed = True

    def compute_cost(self, x_eval, p_eval):
        if not hasattr(self, 'fn_cost'):
            self.fn_cost = ca.Function(f'{self.name}_cost',
                                       [self.opti.x, self.opti.p],
                                       [self.cost],
                                       ['x', 'p'],
                                       ['f'],
                                       self.fn_opts
                                       )
        return self.fn_cost(x_eval, p_eval)

    def build_parameter_yaml(self):
        with open(f"{self.name}.yaml", "w") as f:
            for name, param in self.parameters.items():
                flat = param.init.flatten(order='F').tolist()
                f.write(f"{name}: {flat}\n")

    # TODO: Convenient parameter getting and setting from YAML

    # ------------------------  ANALYSIS  ----------------------------- #
    def plot_sparsity_patterns(self):
        x = self.opti.x
        f = self.opti.f
        g = self.opti.g
        n_kkt = self.n_x + self.n_g
        kkt = ca.MX(n_kkt, n_kkt)
        P, _ = ca.hessian(f, x)
        A = ca.jacobian(g, x)
        kkt[:self.n_x, :self.n_x] = P.sparsity()
        kkt[self.n_x:, :self.n_x] = A.sparsity()
        kkt[:self.n_x, self.n_x:] = A.T.sparsity()
        P_sparsity_percentage = 100 * kkt.sparsity().nnz() / (n_kkt*n_kkt)
        plt.figure()
        plt.title(f"KKT: {kkt.sparsity()} ({P_sparsity_percentage:.2f} %)")
        plt.spy(kkt.sparsity())

    # ------------------------  SOLVING  ------------------------------ #
    def setup_solver(self, name='ipopt', **solver_opts):
        if not self.post_processed:
            self.post_process()
        solver_class = get_solver(name, **solver_opts)
        self.solver = solver_class(self, **solver_opts)
        return self.solver

    def solve(self, x_0, p):
        return self.solver.solve(x_0, p)

    # ---------------------  PARALLELIZATION ----------------------------- #
    def parallelize(self,
                    linsys_method='ldl',
                    batch_size=4096,
                    precision='float',
                    dynamic_batching=True):
        from cusadi.parallelization import CusadiFunction, parallelize_functions
        if not self.post_processed:
            self.post_process()
        if not hasattr(self, "solver"):
            print("Solver not instantiated.")
            print("Call configure_solver('sqp') first.")
            raise AssertionError
        if not hasattr(self.solver, "parallelize"):
            raise AssertionError(f"GPU code-generation not available for {self.solver}.")

        cusadi_fns = self.solver.parallelize(
            linsys_method, batch_size, precision, dynamic_batching)
        return cusadi_fns