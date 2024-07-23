import os
import argparse
import torch
from casadi import *
from src import *


def main(args):
    fn_filepath = os.path.join(CUSADI_FUNCTION_DIR, f"{args.fn_name}.casadi")
    f = casadi.Function.load(fn_filepath)
    print("Evaluating function:", f.name())
    print("Function has %d arguments" % f.n_in())
    print("Function has %d outputs" % f.n_out())

    input_tensors = [torch.rand(args.n_envs, f.nnz_in(i), device='cuda', dtype=torch.double).contiguous()
                     for i in range(f.n_in())]

    fn_cusadi = CusadiFunction(f, args.n_envs)
    fn_cusadi.evaluate(input_tensors)

    output_numpy = [numpy.zeros((args.n_envs, f.nnz_out(i))) for i in range(f.n_out())]
    for n in range(args.n_envs):
        inputs_np = [input_tensors[i][n, :].cpu().numpy() for i in range(f.n_in())]
        for i in range(f.n_out()):
            output_numpy[i][n, :] = f.call(inputs_np)[i].nonzeros()

    print(f"Evaluating with {args.n_envs} environments.")
    print(f"Average error for each environment:")
    for i in range(f.n_out()):
        error_norm = numpy.linalg.norm(fn_cusadi.outputs_sparse[i].cpu().numpy() - output_numpy[i])/args.n_envs
        print(f"Output {i} error norm:", error_norm)

def printParserArguments(parser, args):
    # Print out all arguments, descriptions, and default values in a formatted manner
    print(f"\n{'Argument':<10} {'Description':<80} {'Default':<10} {'Current Value':<10}")
    print("=" * 120)
    for action in parser._actions:
        if action.dest == 'help':
            continue
        arg_strings = ', '.join(action.option_strings)
        description = action.help or 'No description'
        default = action.default if action.default is not argparse.SUPPRESS else 'No default'
        current_value = getattr(args, action.dest, default)
        print(f"{arg_strings:<10} {description:<80} {default:<10} {current_value:<10}")
    print()

def setupParser():
    parser = argparse.ArgumentParser(description='Script to evaluate Cusadi function and check error')
    parser.add_argument('--fn', type=str, dest='fn_name', default='test',
                        help='Function name in cusadi/casadi_functions, defaults to "test"')
    parser.add_argument('--num_envs', type=int, dest='n_envs', default=4000,
                        help='Number of instances to evaluate in parallel, default to 4000')
    return parser


if __name__ == "__main__":
    parser = setupParser()
    args = parser.parse_args()
    printParserArguments(parser, args)
    main(args)











