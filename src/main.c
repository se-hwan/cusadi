#include <include/main.h>

void evaluateCasADiFunction(float *outputs[],
                            const int num_outputs,
                            const int* output_sizes,

                            const float *inputs[],
                            const int num_inputs,
                            const int* input_sizes,

                            float *work,
                            const int num_work,

                            const int *operations,
                            const int *output_idx,
                            const int *output_idx_lengths,
                            const int *input_idx,
                            const int *input_idx_lengths,
                            const float *const_instr,
                            const int num_instr,
                            const int batch_size) {
    // printf("Evaluating CasADi function\n");
    // printf("Number of outputs: %d\n", num_outputs);
    // printf("Output size: %d\n", output_sizes[0]);
    // printf("Number of inputs: %d\n", num_inputs);
    // printf("Input size: %d\n", input_sizes[0]);
    // printf("Input size: %d\n", input_sizes[1]);
    // printf("Number of instructions: %d\n", num_instr);
    // printf("Batch size: %d\n", batch_size);
    // printf("Work vector size: %d\n", num_work);

    // printf("Work vector test: %f\n", work[0]);
    // printf("Outputs vector test: %f\n", outputs[0]);
    // printf("Outputs vector test: %p\n", outputs[0]);
    
    // printf("Inputs vector test: %p\n", inputs[0]);

    struct CasADiFunctionParameters f_info = {
        .o_instr = 0,
        .i_instr = 0,
    };

    for (int k = 0; k < num_instr; k++) {
        f_info.op = operations[k];
        f_info.o_idx = output_idx[f_info.o_instr];
        f_info.i_idx = input_idx[f_info.i_instr];

        switch (f_info.op) {
            case CONST:
                cu_assign_const(work, f_info.o_idx, num_work,
                                const_instr[k],
                                batch_size);
                break;
            case INPUT:
                cu_assign(work, f_info.o_idx, num_work,
                          inputs[f_info.i_idx], input_idx[f_info.i_instr + 1], input_sizes[f_info.i_idx],
                          batch_size);
                break;
            case OUTPUT:
                cu_assign(outputs[f_info.o_idx], output_idx[f_info.o_instr + 1], output_sizes[f_info.o_idx],
                          work, f_info.i_idx, num_work,
                          batch_size);
                break;
            case ADD:
                cu_add(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       work, input_idx[f_info.i_instr + 1], num_work,
                       batch_size);
                break;
            case SUB:
                cu_sub(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       work, input_idx[f_info.i_instr + 1], num_work,
                       batch_size);
                break;
            case MUL:
                cu_mul(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       work, input_idx[f_info.i_instr + 1], num_work,
                       batch_size);
                break;
            case DIV:
                cu_div(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       work, input_idx[f_info.i_instr + 1], num_work,
                       batch_size);
                break;
            case NEG:
                cu_neg(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       batch_size);
            case SIN:
                cu_sin(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       batch_size);
                break;
            case COS:
                cu_cos(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       batch_size);
                break;
            case TAN:
                cu_tan(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       batch_size);
                break;
            case SQRT:
                cu_sqrt(work, f_info.o_idx, num_work,
                        work, f_info.i_idx, num_work,
                        batch_size);
                break;
            case SQ:
                cu_sq(work, f_info.o_idx, num_work,
                      work, f_info.i_idx, num_work,
                      batch_size);
                break;
            case EXP:
                cu_exp(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       batch_size);
                break;
            case LOG:
                cu_log(work, f_info.o_idx, num_work,
                       work, f_info.i_idx, num_work,
                       batch_size);
                break;
            default:
                printf("Operation not implemented\n");
                printf("Current operation: %d\n", f_info.op);
                printf("Current output index: %d\n", f_info.o_idx);
                printf("Current input index: %d\n", f_info.i_idx);
                exit(1);
        }
        f_info.o_instr += output_idx_lengths[k + 1];
        f_info.i_instr += input_idx_lengths[k + 1];
    }
}

void printFloatColumn(const float *array, int rows, int col) {
    printf("Printing column of float array\n");
    for (int i = 0; i < rows; i++) {
        printf("%f\n", array[col*rows + i]);
    }
}

void printIntList(const int* array, int size) {
    printf("Printing int list\n");
    for (int i = 0; i < size; i++) {
        printf("%d\n", array[i]);
    }
}

void printIntListOfLists(const int **array, int *sizes, int num_arrays) {
    for (int i = 0; i < num_arrays; i++) {
        int size = sizes[i];
        printf("Array %d: ", i + 1);
        for (int j = 0; j < size; j++) {
            printf("%d ", array[i][j]);
        }
        printf("\n");
    }
}

void printFloatList(const float* array, int size) {
    printf("Printing float list\n");
    for (int i = 0; i < size; i++) {
        printf("%f\n", array[i]);
    }
}

void launchMyCudaFunction(float *f_out, const unsigned int col_f_out,
                          const float *f_in1, const unsigned int col_f_in1,
                          const float *f_in2, const unsigned int col_f_in2,
                          const unsigned int num_rows, const unsigned int num_cols) {
    printf("Launching CUDA function\n");
    // op_add(f_out, col_f_out, f_in1, col_f_in1, f_in2, col_f_in2, num_rows, num_cols);
}

void printOperationKeys() {
    printf("Operation keys:\n");
    printf("    INPUT: %d\n", INPUT);
    printf("    CONSTANT: %d\n", CONST);
    printf("    ADD: %d\n", ADD);
    printf("    SUB: %d\n", SUB);
    printf("    MUL: %d\n", MUL);
    printf("    DIV: %d\n", DIV);
    printf("    SIN: %d\n", SIN);
    printf("    COS: %d\n", COS);
    printf("    TAN: %d\n", TAN);
    printf("    SQRT: %d\n", SQRT);
    printf("    SQ: %d\n", SQ);
    printf("    POW: %d\n", POW);
    printf("    EXP: %d\n", EXP);
    printf("    LOG: %d\n", LOG);
}