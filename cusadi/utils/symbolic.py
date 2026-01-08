
import numpy as np
import casadi as ca

def auto_concat(items: list, symbolic=False):
    if symbolic:
        return ca.vertcat(*items)
    else:
        return np.hstack(items)

def auto_array(items: list, symbolic=False):
    if symbolic:
        return ca.DM(items)
    else:
        return np.array(items)

def get_function_options(custom_options: dict={}):
    # https://web.casadi.org/python-api/#function
    options = {
        'jit': False,
        'jit_cleanup': True,
        'print_time': False,
        'cse': True,
        'post_expand': True}
    for key, value in custom_options.items():
        options[key] = value
    return options

def get_codegen_options(custom_options: dict={}):
    # https://web.casadi.org/python-api/#function
    options = {
        'verbose': False,
        'mex': False,
        'main': False,
        'cpp': False,
        'casadi_real': 'double',
        'casadi_int': 'long long int',
        'with_header': False,
        'with_mem': False,
        'indent': 2}
    for key, value in custom_options.items():
        options[key] = value
    return options

def get_solver_options(solver, custom_options: dict={}, verbose=True):
    if solver == 'ipopt':
        p_opts = {'expand': True}
        s_opts = {
            'max_iter': 2000, 
            'max_cpu_time': 100.0, 
            'tol': 1e-4,  # 1e-6
            'acceptable_tol': 1e-4,  # 1e-4
            'constr_viol_tol': 1e-4,  # 1e-6
            'acceptable_iter': 5,  # 15
            'nlp_scaling_method': 'gradient-based', # {'gradient-based', 'none', 'equilibration-based'};
            'nlp_scaling_max_gradient': 50,  # 100
            'bound_relax_factor': 1e-6,  # 1e-8
            'fixed_variable_treatment': 'relax_bounds',  # {'make_parameter', 'make_constraint', 'relax_bounds'};
            'bound_frac': 1e-2,  # 1e-2
            'bound_push': 1e-2,  # 1e-2
            'mu_strategy': 'adaptive',  # {'monotone', 'adaptive'}; # adaptive seems to work well
            'mu_oracle': 'quality-function',  # {'quality-function', 'probing', 'loqo'};
            'fixed_mu_oracle': 'quality-function',  # {'average_compl', 'quality-function', 'probing', 'loqo'};
            'adaptive_mu_globalization': 'obj-constr-filter',  # {'obj-constr-filter', 'kkt-error', 'never-monotone-mode'};
            'mu_init': 1e-1,  # [1e-1 1e-2 1]
            'alpha_for_y': 'full',  # {'primal', 'bound-mult', 'min', 'max', 'full', 'min-dual-infeas', 'safer-min-dual-infeas', 'primal-and-full'}; # full seems best
            'alpha_for_y_tol': 1e1,  # 1e1
            'recalc_y': 'no',  # {'no', 'yes'};
            'max_soc': 4,  # 4
            'accept_every_trial_step': 'no',  # {'no', 'yes'}
            'linear_solver': 'ma27',  # {'ma27', 'mumps', 'ma57', 'ma77', 'ma86'} # ma27 seems to work well
            'linear_system_scaling': 'slack-based',  # {'mc19', 'none', 'slack-based'};
            'linear_scaling_on_demand': 'yes',  # {'yes', 'no'};
            'max_refinement_steps': 10,  # 10
            'min_refinement_steps': 1, # 1
            'file_print_level': 5,
            'print_level': 5,
            'print_frequency_iter': 5,
        } 
        if not verbose:
            s_opts['print_level'] = 0
            s_opts['print_timing_statistics'] = 'no'
            p_opts['ipopt.print_level'] = 0
            p_opts['print_time'] = 0
            p_opts['ipopt.sb'] = 'yes'
        for key, value in custom_options.items():
            s_opts[key] = value
    return p_opts, s_opts