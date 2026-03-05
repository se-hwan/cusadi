from .ipopt_interface import IPOPTInterface
from .sqp_interface import SQPInterface

def get_solver(name, **solver_opts):
    solver_registry = {
        'ipopt': {
            'class': IPOPTInterface,
            'msg': "IPOPT solver set. No GPU code-generation available."
        },
        'sqp': {
            'class': SQPInterface,
            'msg': f"SQP solver set with QP solver: {solver_opts.get('qp_solver', 'osqp')}."
        }
    }
    if name not in solver_registry:
        valid_solvers = ", ".join(solver_registry.keys())
        raise KeyError(f"Solver '{name}' unknown. Choose from: {valid_solvers}")

    config = solver_registry[name]
    print(config['msg'])
    print("Evaluate by calling solve([x_init], [params]).")
    return config['class']