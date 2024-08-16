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
instr_data = {n_1e1_data, n_1e2_data, n_1e3_data, n_1e4_data, n_1e5_data};
num_env_labels = arrayfun(@num2str, benchmark_data.N_ENVS_SWEEP, 'UniformOutput', false);
num_instr_labels = {'1E1', '1E2', '1E3', '1E4', '1E5'};
legend_entries = {'  CusADi (ours)', '  Pytorch', '  CPU (parallel)'};
colors = ['g', '#EB5036', '#332288'];

for i = 1:numel(instr_data)
    instr_data{i}.cusadi_stats.mean = mean(instr_data{i}.cusadi, 2);
    instr_data{i}.pytorch_stats.mean = mean(instr_data{i}.pytorch, 2);
    instr_data{i}.serial_cpu_stats.mean = mean(instr_data{i}.serial_cpu, 2);
    instr_data{i}.parallel_cpu_stats.mean = mean(instr_data{i}.parallel_cpu, 2);
    instr_data{i}.serial_cpu_transfer_stats.mean = mean(instr_data{i}.serial_cpu_transfer, 2);
    instr_data{i}.parallel_cpu_transfer_stats.mean = mean(instr_data{i}.parallel_cpu_transfer, 2);
    instr_data{i}.cusadi_stats.std = std(instr_data{i}.cusadi, 0, 2);
    instr_data{i}.pytorch_stats.std = std(instr_data{i}.pytorch, 0, 2);
    instr_data{i}.serial_cpu_stats.std = std(instr_data{i}.serial_cpu, 0, 2);
    instr_data{i}.parallel_cpu_stats.std = std(instr_data{i}.parallel_cpu, 0, 2);
    instr_data{i}.serial_cpu_transfer_stats.std = std(instr_data{i}.serial_cpu_transfer, 0, 2);
    instr_data{i}.parallel_cpu_transfer_stats.std = std(instr_data{i}.parallel_cpu_transfer, 0, 2);
end

% SPEED UP RELATIVE TO CPU
speedup_data = {};
tmp_cusadi = []; tmp_pytorch = []; tmp_parallel_cpu = []; tmp_parallel_cpu_transfer = [];
for i=1:numel(instr_data)
    instr_data{i}.t_baseline = mean(instr_data{i}.serial_cpu, 2);
    instr_data{i}.cusadi_speedup = instr_data{i}.t_baseline ./ instr_data{i}.cusadi;
    instr_data{i}.pytorch_speedup = instr_data{i}.t_baseline ./ instr_data{i}.pytorch;
    instr_data{i}.serial_cpu_speedup = instr_data{i}.t_baseline ./ instr_data{i}.serial_cpu;
    instr_data{i}.parallel_cpu_speedup = instr_data{i}.t_baseline ./ instr_data{i}.parallel_cpu;
    instr_data{i}.serial_cpu_transfer_speedup = instr_data{i}.t_baseline ./ instr_data{i}.serial_cpu_transfer;
    instr_data{i}.parallel_cpu_transfer_speedup = instr_data{i}.t_baseline ./ instr_data{i}.parallel_cpu_transfer;
    tmp_cusadi = [tmp_cusadi, instr_data{i}.t_baseline./instr_data{i}.cusadi_stats.mean];
    tmp_pytorch = [tmp_pytorch, instr_data{i}.t_baseline./instr_data{i}.pytorch_stats.mean];
    tmp_parallel_cpu = [tmp_parallel_cpu, instr_data{i}.t_baseline./instr_data{i}.parallel_cpu_stats.mean];
    tmp_parallel_cpu_transfer = [tmp_parallel_cpu_transfer, instr_data{i}.t_baseline./instr_data{i}.parallel_cpu_transfer_stats.mean];
end
speedup_data.cusadi = tmp_cusadi;
speedup_data.pytorch = tmp_pytorch;
speedup_data.parallel_cpu = tmp_parallel_cpu;
speedup_data.parallel_cpu_transfer = tmp_parallel_cpu_transfer;

% Number of categories and datasets
num_categories = numel(num_env_labels);
num_datasets = numel(num_instr_labels);
bar_width = 0.17;
spacing = 1:1:1*num_categories;
xlim_range = [9.43, num_categories + 0.5];
ylim_range = [-4, 5];

% Speedup plot
sp1 = subplot(1, 2, 1); hold on;


% Set y-axis limits for centered plotting
ylim(ylim_range);  % This sets the range to show values from 10^-3 to 10^3
xlim(xlim_range);

% Customize y-axis tick labels to reflect original values centered around 1
yticks = -5:5;  % Corresponds to log10 values from 10^-3 to 10^3
yticklabels = arrayfun(@(x) sprintf('10^{%d}', x), yticks, 'UniformOutput', false);
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels);

markers = {'o', '+', '*', 'square', 'diamond'};

for i = 1:5
    bar_positions = (spacing) + (i - (num_datasets + 1)/2) * bar_width;
    % bar_positions = (spacing) + 5*i* (bar_width+0.2);
    t1 = bar(bar_positions, log10(speedup_data.cusadi(:, i)), bar_width, 'FaceColor', 'g', 'EdgeAlpha', 0.2);
    t3 = bar(bar_positions, log10(speedup_data.pytorch(:, i)), 0.75*bar_width, 'FaceColor', '#EB5036', 'EdgeAlpha', 0.2);
    t2 = bar(bar_positions, log10(speedup_data.parallel_cpu(:, i)), 0.5*bar_width, 'FaceColor', '#332288', 'EdgeAlpha', 0.2);

    marker_pos = max(max(log10(speedup_data.cusadi(:, i)), ...
                     max(log10(speedup_data.pytorch(:, i)), log10(speedup_data.parallel_cpu(:, i)))), 0);
    plot(bar_positions, marker_pos + 0.25, markers{i}, 'MarkerEdgeColor', 'k', 'MarkerSize', 5);
end
grid on;
ax = gca;
ax.XGrid = 'off';


qw{1} = plot(nan, 'o', 'MarkerEdgeColor', 'k');
qw{2} = plot(nan, '+', 'MarkerEdgeColor', 'k');
qw{3} = plot(nan, '*', 'MarkerEdgeColor', 'k');
qw{4} = plot(nan, 'square', 'MarkerEdgeColor', 'k');
qw{5} = plot(nan, 'diamond', 'MarkerEdgeColor', 'k');

% Set x-axis labels and ticks
set(gca, 'XTick', spacing, 'XTickLabel', num_env_labels);
bar_handle{1} = bar(nan, 'g');
bar_handle{2} = bar(nan, 'FaceColor', '#EB5036');
bar_handle{3} = bar(nan, 'FaceColor', '#332288');
leg_1 = legend([bar_handle{:}], legend_entries);
leg_1.Position = [0.349462059914672,0.143848920863309,0.11525,0.188848920863309];
xlabel('Batch size');
ylabel('Speedup (x)');
% set(gca, 'FontSize', 11)
% axis([3, 11, 0, 1200])
ah1 = axes('position',get(gca,'position'),'visible','off');
leg_2 = legend(ah1, [qw{:}], num_instr_labels);
leg_2.Position = [0.141293532338311,0.605946897158533,0.067661691542289,0.298897572589411];
title(leg_1, 'Evaluation')
title(leg_2, 'Num. instr.')
% set(gca, 'FontSize', 10)



%% SPEEDUP RELATIVE TO CPU + TRANSFER TIME

speedup_data = {};
tmp_cusadi = []; tmp_pytorch = []; tmp_parallel_cpu = []; tmp_parallel_cpu_transfer = [];
for i=1:numel(instr_data)
    instr_data{i}.t_baseline = mean(instr_data{i}.serial_cpu_transfer, 2);
    instr_data{i}.cusadi_speedup = instr_data{i}.t_baseline ./ instr_data{i}.cusadi;
    instr_data{i}.pytorch_speedup = instr_data{i}.t_baseline ./ instr_data{i}.pytorch;
    instr_data{i}.serial_cpu_speedup = instr_data{i}.t_baseline ./ instr_data{i}.serial_cpu;
    instr_data{i}.parallel_cpu_speedup = instr_data{i}.t_baseline ./ instr_data{i}.parallel_cpu;
    instr_data{i}.serial_cpu_transfer_speedup = instr_data{i}.t_baseline ./ instr_data{i}.serial_cpu_transfer;
    instr_data{i}.parallel_cpu_transfer_speedup = instr_data{i}.t_baseline ./ instr_data{i}.parallel_cpu_transfer;
    tmp_cusadi = [tmp_cusadi, instr_data{i}.t_baseline./instr_data{i}.cusadi_stats.mean];
    tmp_pytorch = [tmp_pytorch, instr_data{i}.t_baseline./instr_data{i}.pytorch_stats.mean];
    tmp_parallel_cpu = [tmp_parallel_cpu, instr_data{i}.t_baseline./instr_data{i}.parallel_cpu_stats.mean];
    tmp_parallel_cpu_transfer = [tmp_parallel_cpu_transfer, instr_data{i}.t_baseline./instr_data{i}.parallel_cpu_transfer_stats.mean];
end
speedup_data.cusadi = tmp_cusadi;
speedup_data.pytorch = tmp_pytorch;
speedup_data.parallel_cpu = tmp_parallel_cpu;
speedup_data.parallel_cpu_transfer = tmp_parallel_cpu_transfer;


sp2 = subplot(1, 2, 2); hold on;

% Number of categories and datasets
num_categories = numel(num_env_labels);
num_datasets = numel(num_instr_labels);
bar_width = 0.17;

% Set y-axis limits for centered plotting
ylim(ylim_range);  % This sets the range to show values from 10^-3 to 10^3
xlim(xlim_range);

% Customize y-axis tick labels to reflect original values centered around 1
yticks = -5:5;  % Corresponds to log10 values from 10^-3 to 10^3
yticklabels = arrayfun(@(x) sprintf('10^{%d}', x), yticks, 'UniformOutput', false);
set(gca, 'YTick', yticks, 'YTickLabel', yticklabels);

markers = {'o', '+', '*', 'square', 'diamond'};

for i = 1:num_datasets
    bar_positions = (1:num_categories) + (i - (num_datasets + 1)/2) * bar_width;
    t1 = bar(bar_positions, log10(speedup_data.cusadi(:, i)), bar_width, 'FaceColor', 'g', 'EdgeAlpha', 0.2);
    t3 = bar(bar_positions, log10(speedup_data.pytorch(:, i)), 0.75*bar_width, 'FaceColor', '#EB5036', 'EdgeAlpha', 0.2);
    t2 = bar(bar_positions, log10(speedup_data.parallel_cpu_transfer(:, i)), 0.5*bar_width, 'FaceColor', '#332288', 'EdgeAlpha', 0.2);

    marker_pos = max(max(log10(speedup_data.cusadi(:, i)), ...
                     max(log10(speedup_data.pytorch(:, i)), log10(speedup_data.parallel_cpu_transfer(:, i)))), 0);
    plot(bar_positions, marker_pos + 0.25, markers{i}, 'MarkerEdgeColor', 'k', 'MarkerSize', 5);
end
grid on;
ax = gca;
ax.XGrid = 'off';


qw{1} = plot(nan, 'o', 'MarkerEdgeColor', 'k');
qw{2} = plot(nan, '+', 'MarkerEdgeColor', 'k');
qw{3} = plot(nan, '*', 'MarkerEdgeColor', 'k');
qw{4} = plot(nan, 'square', 'MarkerEdgeColor', 'k');
qw{5} = plot(nan, 'diamond', 'MarkerEdgeColor', 'k');

% Set x-axis labels and ticks
set(gca, 'XTick', 1:num_categories, 'XTickLabel', num_env_labels);
% bar_handle{1} = bar(nan, 'g');
% bar_handle{2} = bar(nan, 'FaceColor', '#EB5036');
% bar_handle{3} = bar(nan, 'FaceColor','#332288');
% leg_1 = legend([bar_handle{:}], legend_entries, 'Location', 'NorthWest');
xlabel('Batch size');
ylabel('Speedup (x)');
ah1 = axes('position',get(gca,'position'),'visible','off');
% leg_2 = legend(ah1, [qw{:}], num_instr_labels, 'location', 'southeast');
title(leg_1, 'Evaluation')
title(leg_2, 'Num. instr.')
set(gcf,'position',[1000, 2000, 2000, 600])
exportgraphics(gcf, 'test.pdf','ContentType','vector')