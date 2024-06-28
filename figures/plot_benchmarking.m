clear; clc; close all;

%% Load data
benchmark_data = load('../data/benchmark_data.mat');

%% Parse data
N_ENVS = benchmark_data.N_ENVS_SWEEP;
N_INSTR = benchmark_data.N_INSTRUCTIONS;
n_1e1_data = benchmark_data.n_1e1;
n_1e2_data = benchmark_data.n_1e2;
n_1e3_data = benchmark_data.n_1e3;
n_1e4_data = benchmark_data.n_1e4;
n_1e5_data = benchmark_data.n_1e5;
n_1e6_data = benchmark_data.n_1e6;
instr_data = {n_1e1_data, n_1e2_data, n_1e3_data, n_1e4_data, n_1e5_data, n_1e6_data};

for i = 1:length(instr_data)
    instr_data{i}.t_baseline = mean(instr_data{i}.parallel_cpu, 2); % CHECK THIS DIMENSION! should be NUM_ENVS_SWEEP x 1
    instr_data{i}.cusadi_speedup = instr_data{i}.cusadi ./ instr_data{i}.t_baseline; % CHECK THIS DIMENSION! should be NUM_ENVS_SWEEP x 1
    instr_data{i}.pytorch_speedup = instr_data{i}.pytorch ./ instr_data{i}.t_baseline;
    instr_data{i}.serial_cpu_speedup = instr_data{i}.serial_cpu ./ instr_data{i}.t_baseline;
end



%% Plots
figure; hold on;
xline(1, 'k--');
plot(N_ENVS, mean(instr_data{1}.cusadi_speedup, 2), 'LineWidth', 1.5);
plot(N_ENVS, mean(instr_data{1}.pytorch_speedup, 2), 'LineWidth', 1.5);
plot(N_ENVS, mean(instr_data{1}.serial_cpu_speedup, 2), 'LineWidth', 1.5);

xlabel("Number of environments")
ylabel("Evaluation time (s)")