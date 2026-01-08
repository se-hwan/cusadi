#pragma once
#include <torch/extension.h>


extern "C" void launch_ADMM_step_kernel(
    const int batch_size,
    const float* input_0,
    const float* input_1,
    const float* input_2,
    const float* input_3,
    const float* input_4,
    const float* input_5,
    float* output_0,
    float* output_1,
    float* output_2,
    float* work);

void ADMM_step_binding(
    const int batch_size,
    const torch::Tensor& input_0,
    const torch::Tensor& input_1,
    const torch::Tensor& input_2,
    const torch::Tensor& input_3,
    const torch::Tensor& input_4,
    const torch::Tensor& input_5,
    torch::Tensor& output_0,
    torch::Tensor& output_1,
    torch::Tensor& output_2,
    torch::Tensor& work) {
    launch_ADMM_step_kernel(
        batch_size,
        input_0.data_ptr<float>(),
        input_1.data_ptr<float>(),
        input_2.data_ptr<float>(),
        input_3.data_ptr<float>(),
        input_4.data_ptr<float>(),
        input_5.data_ptr<float>(),
        output_0.data_ptr<float>(),
        output_1.data_ptr<float>(),
        output_2.data_ptr<float>(),
        work.data_ptr<float>());
}

extern "C" void launch_tau_kernel(
    const int batch_size,
    const float* input_0,
    const float* input_1,
    const float* input_2,
    float* output_0,
    float* work);

void tau_binding(
    const int batch_size,
    const torch::Tensor& input_0,
    const torch::Tensor& input_1,
    const torch::Tensor& input_2,
    torch::Tensor& output_0,
    torch::Tensor& work) {
    launch_tau_kernel(
        batch_size,
        input_0.data_ptr<float>(),
        input_1.data_ptr<float>(),
        input_2.data_ptr<float>(),
        output_0.data_ptr<float>(),
        work.data_ptr<float>());
}

