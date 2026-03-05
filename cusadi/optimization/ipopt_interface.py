import osqp
import casadi as ca
import numpy as np
from cusadi.utils.symbolic import *

class IPOPTInterface:
    def __init__(self, problem, verbose=True, solver_cfg={}):
        self.problem = problem
        self.opti = problem.opti
        [self.p_opts, self.s_opts] = get_solver_options(
            'ipopt', verbose, solver_cfg)
        self.opti.solver('ipopt', self.p_opts, self.s_opts)

    def solve(self, x_eval, params):
        self.opti.set_initial(self.opti.x, x_eval)
        if params:
            self.opti.set_value(self.opti.p, params)
        soln = self.opti.solve()
        return soln.value(self.opti.x)