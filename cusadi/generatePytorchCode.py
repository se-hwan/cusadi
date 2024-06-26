import textwrap
from casadi import *
from cusadi import *

def generatePytorchCode(f, filepath=None):
    print("Generating Pytorch code for CasADi function: ", f.name())
    if filepath is None:
        codegen_filepath = os.path.join(CUSADI_ROOT_DIR, "codegen", f"{f.name()}.py")
    else:
        codegen_filepath = filepath
    codegen_file = open(codegen_filepath, "w+")
    codegen_strings = {}

    # * Parse CasADi function
    n_instr = f.n_instructions()
    n_in = f.n_in()
    n_out = f.n_out()
    nnz_in = [f.nnz_in(i) for i in range(n_in)]
    nnz_out = [f.nnz_out(i) for i in range(n_out)]
    n_w = f.sz_w()

    INSTR_LIMIT = n_instr  
    input_idx = [f.instruction_input(i) for i in range(INSTR_LIMIT)]
    output_idx = [f.instruction_output(i) for i in range(INSTR_LIMIT)]
    operations = [f.instruction_id(i) for i in range(INSTR_LIMIT)]
    const_instr = [f.instruction_constant(i) for i in range(INSTR_LIMIT)]

    # * Codegen for const declarations and indices
    codegen_strings['header'] = "# ! AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
    codegen_strings['includes'] = textwrap.dedent(
    '''
    import torch

    ''')
    codegen_strings["nnz_in"] = f"nnz_in = [{','.join(map(str, nnz_in))}]\n"
    codegen_strings["nnz_out"] = f"nnz_out = [{','.join(map(str, nnz_out))}]\n"
    codegen_strings["n_w"] = f"n_w = {n_w}\n\n"

    # * Codegen for Pytorch
    str_operations = "@torch.compile\n"
    str_operations += f"def evaluate_{f.name()}(outputs, inputs, work):"

    for k in range(INSTR_LIMIT):
        op = operations[k]
        o_idx = output_idx[k]
        i_idx = input_idx[k]
        if op == OP_CONST:
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], const_instr[k])
        elif op == OP_INPUT:
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0], i_idx[1])
        elif op == OP_OUTPUT:
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], o_idx[1], i_idx[0])
        elif op == OP_SQ:
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0], i_idx[0])
        elif OP_PYTORCH_DICT[op].count("%d") == 3:
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0], i_idx[1])
        elif OP_PYTORCH_DICT[op].count("%d") == 2:
            str_operations += OP_PYTORCH_DICT[op] % (o_idx[0], i_idx[0])
        else:
            raise Exception('Unknown CasADi operation: ' + str(op))

    codegen_strings['pytorch_operations'] = str_operations

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("Pytorch codegen complete for CasADi function: ", f.name())
