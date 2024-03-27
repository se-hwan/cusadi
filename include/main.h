#ifndef MAIN_H
#define MAIN_H
#include <stdio.h>
#include <stdlib.h>

extern void cu_assign_const(float *out, const int col_out, const int sz_out,
                            const float in,
                            const int batch_size);
extern void cu_assign(float *out, const int col_out, const int sz_out,
                      const float *in, const int col_in, const int sz_in,
                      const int batch_size);
extern void cu_neg(float *out, const int col_out, const int sz_out,
                      const float *in, const int col_in, const int sz_in,
                      const int batch_size);
extern void cu_add(float *out, const int col_out, const int sz_out,
                   const float *in1, const int col_in1, const int sz_in1,
                   const float *in2, const int col_in2, const int sz_in2,
                   const int batch_size);
extern void cu_sub(float *out, const int col_out, const int sz_out,
                        const float *in1, const int col_in1, const int sz_in1,
                        const float *in2, const int col_in2, const int sz_in2,
                        const int batch_size);
extern void cu_mul(float *out, const int col_out, const int sz_out,
                        const float *in1, const int col_in1, const int sz_in1,
                        const float *in2, const int col_in2, const int sz_in2,
                        const int batch_size);
extern void cu_div(float *out, const int col_out, const int sz_out,
                      const float *in1, const int col_in1, const int sz_in1,
                      const float *in2, const int col_in2, const int sz_in2,
                      const int batch_size);
extern void cu_sin(float *out, const int col_out, const int sz_out,
                   const float *in, const int col_in, const int sz_in,
                   const int batch_size);
extern void cu_cos(float *out, const int col_out, const int sz_out,
                   const float *in, const int col_in, const int sz_in,
                   const int batch_size);
extern void cu_tan(float *out, const int col_out, const int sz_out,
                   const float *in, const int col_in, const int sz_in,
                   const int batch_size);
extern void cu_exp(float *out, const int col_out, const int sz_out,
                   const float *in, const int col_in, const int sz_in,
                   const int batch_size);
extern void cu_log(float *out, const int col_out, const int sz_out,
                   const float *in, const int col_in, const int sz_in,
                   const int batch_size);
extern void cu_sqrt(float *out, const int col_out, const int sz_out,
                    const float *in, const int col_in, const int sz_in,
                    const int batch_size);
extern void cu_sq(float *out, const int col_out, const int sz_out,
                  const float *in, const int col_in, const int sz_in,
                  const int batch_size);

void evaluateCasADiFunction(float *outputs[],
                            const int num_outputs,
                            const int* output_sizes,

                            const float *inputs[],
                            const int num_inputs,
                            const int* input_sizes,

                            float *work,
                            const int num_work,

                            const int operations[],
                            const int *output_idx,
                            const int *output_idx_lengths,
                            const int *input_idx,
                            const int *input_idx_lengths,
                            const float *const_instr,
                            const int num_instr,
                            const int batch_size);
void printOperationKeys();
void printIntList(const int* array, int size);
void printIntListOfLists(const int **array, int *sizes, int num_arrays);
void printFloatList(const float* array, int size);
void launchMyCudaFunction(float *f_out, const unsigned int col_f_out,
                          const float *f_in1, const unsigned int col_f_in1,
                          const float *f_in2, const unsigned int col_f_in2,
                          const unsigned int num_rows, const unsigned int num_cols);


struct CasADiFunctionParameters {
    int op;
    int o_idx;
    int o_instr;
    int i_idx;
    int i_instr;
    int num_outputs;
    int num_inputs;
    int sz_w;
};

enum CasADiOperations {
    INPUT = 45,
    OUTPUT = 46,
    CONST = 44,
    ADD = 1,
    SUB = 2,
    MUL = 3,
    DIV = 4,
    NEG = 5,
    EXP = 6,
    LOG = 7,
    POW = 8,
    SQRT = 10,
    SQ = 11,
    SIN = 13,
    COS = 14,
    TAN = 15
};

#endif