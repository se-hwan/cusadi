
import torch
from casadi import *

def evaluateCasADiPython(output,
                         work_tensor,
                         input_batch,
                         operations,
                         output_idx,
                         input_idx,
                         const_instr,
                         num_instructions):
    for k in range(num_instructions):
        # Get the atomic operation
        op = operations[k]
        o = output_idx[k]
        i = input_idx[k]
        if(op==OP_CONST):
            work_tensor[:, o[0]] = const_instr[k]
        else:
            if op==OP_INPUT:
                work_tensor[:, o[0]] = input_batch[i[0]][:, i[1]]
            elif op==OP_OUTPUT:
                output[o[0]][:, o[1]] = work_tensor[:, i[0]]
            elif op==OP_ADD:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] + work_tensor[:, i[1]]
            elif op==OP_SUB:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] - work_tensor[:, i[1]]
            elif op==OP_NEG:
                work_tensor[:, o[0]] = -work_tensor[:, i[0]]
            elif op==OP_MUL:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] * work_tensor[:, i[1]]
            elif op==OP_DIV:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] / work_tensor[:, i[1]]
            elif op==OP_SIN:
                work_tensor[:, o[0]] = torch.sin(work_tensor[:, i[0]])
            elif op==OP_COS:
                work_tensor[:, o[0]] = torch.cos(work_tensor[:, i[0]])
            elif op==OP_TAN:
                work_tensor[:, o[0]] = torch.tan(work_tensor[:, i[0]])
            elif op==OP_SQ:
                work_tensor[:, o[0]] = work_tensor[:, i[0]] * work_tensor[:, i[0]]
            elif op==OP_SQRT:
                work_tensor[:, o[0]] = torch.sqrt(work_tensor[:, i[0]])
            else:
                raise Exception('Unknown CasADi operation: ' + str(op))
    return output