clear; clc; close all;

%% Load data
benchmark_data = load('../data/benchmark_data.mat');

%% Parse data
N_ENVS = cast(benchmark_data.N_ENVS_SWEEP, 'single');
N_INSTR = cast(benchmark_data.N_INSTRUCTIONS, 'single'); N_INSTR = N_INSTR(1:end-1);
n_1e1_data = benchmark_data.n_1e1;
n_1e2_data = benchmark_data.n_1e2;
n_1e3_data = benchmark_data.n_1e3;
n_1e4_data = benchmark_data.n_1e4;
n_1e5_data = benchmark_data.n_1e5;
% n_1e6_data = benchmark_data.n_1e6;
instr_data = {n_1e1_data, n_1e2_data, n_1e3_data, n_1e4_data, n_1e5_data};

for i = 1:length(instr_data)
    instr_data{i}.t_baseline = mean(instr_data{i}.serial_cpu, 2); % CHECK THIS DIMENSION! should be NUM_ENVS_SWEEP x 1
    instr_data{i}.cusadi_stats.mean = mean(instr_data{i}.cusadi, 2);
    instr_data{i}.pytorch_stats.mean = mean(instr_data{i}.pytorch, 2);
    instr_data{i}.serial_cpu_stats.mean = mean(instr_data{i}.serial_cpu, 2);
    instr_data{i}.parallel_cpu_stats.mean = mean(instr_data{i}.parallel_cpu, 2);
    instr_data{i}.cusadi_stats.std = std(instr_data{i}.cusadi, 0, 2);
    instr_data{i}.pytorch_stats.std = std(instr_data{i}.pytorch, 0, 2);
    instr_data{i}.serial_cpu_stats.std = std(instr_data{i}.serial_cpu, 0, 2);
    instr_data{i}.parallel_cpu_stats.std = std(instr_data{i}.parallel_cpu, 0, 2);
    instr_data{i}.cusadi_speedup = instr_data{i}.t_baseline ./ instr_data{i}.cusadi;
    instr_data{i}.pytorch_speedup = instr_data{i}.t_baseline ./ instr_data{i}.pytorch;
    instr_data{i}.serial_cpu_speedup = instr_data{i}.t_baseline ./ instr_data{i}.serial_cpu;
    instr_data{i}.parallel_cpu_speedup = instr_data{i}.t_baseline ./ instr_data{i}.parallel_cpu;
end

%% Plots
instr_idx = 5;

% Time plot
figure; hold on;
errorbar(log10(N_ENVS), instr_data{instr_idx}.cusadi_stats.mean, instr_data{instr_idx}.cusadi_stats.std, 'ro-', 'LineWidth', 1.5);
errorbar(log10(N_ENVS), instr_data{instr_idx}.pytorch_stats.mean, instr_data{instr_idx}.pytorch_stats.std, 'go-', 'LineWidth', 1.5);
errorbar(log10(N_ENVS), instr_data{instr_idx}.serial_cpu_stats.mean, instr_data{instr_idx}.serial_cpu_stats.std, 'bo-', 'LineWidth', 1.5);
errorbar(log10(N_ENVS), instr_data{instr_idx}.parallel_cpu_stats.mean, instr_data{instr_idx}.parallel_cpu_stats.std, 'co-', 'LineWidth', 1.5);
% yline(1, 'k--');
% axis([0, log10(10000), 0, 0.02])
legend("CusADi (ours)", "Pytorch", "CPU (Serial)", "CPU (Parallel)")
xlabel("Number of environments")
ylabel("Evaluation time (s)")

% Speedup plot
figure; hold on;

num_env_labels = {'1', '5', '10', '50', '100', '250', '500', '1000', '5000', '10000'};
num_instr_labels = {'1E1', '1E2', '1E3', '1E4', '1E5'};
legend_entries = {'CusADi (ours)', 'Pytorch', 'CPU (Parallel)'};

speedup_data = {};
tmp_cusadi = []; tmp_pytorch = []; tmp_parallel_cpu = [];
for i=1:numel(instr_data)
    tmp_cusadi = [tmp_cusadi, instr_data{i}.t_baseline./instr_data{i}.cusadi_stats.mean];
    tmp_pytorch = [tmp_pytorch, instr_data{i}.t_baseline./instr_data{i}.pytorch_stats.mean];
    tmp_parallel_cpu = [tmp_parallel_cpu, instr_data{i}.t_baseline./instr_data{i}.parallel_cpu_stats.mean];
end
speedup_data.cusadi = tmp_cusadi;
speedup_data.pytorch = tmp_pytorch;
speedup_data.parallel_cpu = tmp_parallel_cpu;

% % Example standard deviation data
% std_data = []
% for i=1:numel(instr_data)
%     tmp = [instr_data{i}.t_baseline./instr_data{i}.cusadi_stats.std;
%            instr_data{i}.t_baseline./instr_data{i}.pytorch_stats.std;
%            instr_data{i}.t_baseline./instr_data{i}.parallel_cpu_stats.std];
%     std_data = [std_data, tmp];
% end
% std_data = std_data*0;

% Number of categories and datasets
num_categories = numel(num_env_labels);
num_datasets = numel(num_instr_labels);
bar_width = 0.17;

markers = {'o', '+', '*', 'square', 'diamond'};

for i = 1:num_datasets
    bar_positions = (1:num_categories) + (i - (num_datasets + 1)/2) * bar_width;
    bar(bar_positions, speedup_data.cusadi(:, i), bar_width, 'FaceColor', 'g');
    bar(bar_positions, speedup_data.pytorch(:, i), bar_width, 'FaceColor', '#EB5036');
    bar(bar_positions, speedup_data.parallel_cpu(:, i), bar_width, 'b');
    % errorbar(bar_positions, speedup_data(:, 2), std_data(i, :), '.k');
    marker_pos = max(speedup_data.cusadi(:, i), max(speedup_data.pytorch(:, i), speedup_data.parallel_cpu(:, i)));
    plot(bar_positions, marker_pos + 50, markers{i}, 'MarkerEdgeColor', 'k', 'MarkerSize', 5);
end

qw{1} = plot(nan, 'o', 'MarkerEdgeColor', 'k');
qw{2} = plot(nan, '+', 'MarkerEdgeColor', 'k');
qw{3} = plot(nan, '*', 'MarkerEdgeColor', 'k');
qw{4} = plot(nan, 'square', 'MarkerEdgeColor', 'k');
qw{5} = plot(nan, 'diamond', 'MarkerEdgeColor', 'k');

% Set x-axis labels and ticks
set(gca, 'XTick', 1:num_categories, 'XTickLabel', num_env_labels);
bar_handle{1} = bar(nan, 'g');
bar_handle{2} = bar(nan, 'FaceColor', '#EB5036');
bar_handle{3} = bar(nan, 'b');
leg_1 = legend([bar_handle{:}], legend_entries, 'Location', 'NorthWest');
xlabel('Number of function evaluations');
ylabel('Speedup (x)');
% set(gca, 'FontSize', 11)
axis([3, 11, 0, 1400])
ah1 = axes('position',get(gca,'position'),'visible','off');
leg_2 = legend(ah1, [qw{:}], num_instr_labels, 'location', 'West');
title(leg_1, 'Evaluation')
title(leg_2, 'Num. instr.')
% set(gca, 'FontSize', 10)