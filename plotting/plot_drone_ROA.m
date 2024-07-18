clear; clc; close all;

%% Load and parse data
data = load('../data/drone_ROA_data.mat');
t = data.t;
data_F_lim_10 = data.F_lim_0;
data_F_lim_20 = data.F_lim_1;
data_F_lim_30 = data.F_lim_2;
data_F_lim_40 = data.F_lim_3;
data_F_lim_50 = data.F_lim_4;

idx_success_10 = find(data_F_lim_10.successes);
idx_success_20 = find(data_F_lim_20.successes);
idx_success_30 = find(data_F_lim_30.successes);
idx_success_40 = find(data_F_lim_40.successes);
idx_success_50 = find(data_F_lim_50.successes);
successes_10 = double([data_F_lim_10.init_lin_mtm(idx_success_10); data_F_lim_10.init_ang_mtm(idx_success_10)]');
successes_20 = double([data_F_lim_20.init_lin_mtm(idx_success_20); data_F_lim_20.init_ang_mtm(idx_success_20)]');
successes_30 = double([data_F_lim_30.init_lin_mtm(idx_success_30); data_F_lim_30.init_ang_mtm(idx_success_30)]');
successes_40 = double([data_F_lim_40.init_lin_mtm(idx_success_40); data_F_lim_40.init_ang_mtm(idx_success_40)]');
successes_50 = double([data_F_lim_50.init_lin_mtm(idx_success_50); data_F_lim_50.init_ang_mtm(idx_success_50)]');

%% Plotting
colors = [
    0, 26, 35;
    49, 73, 60;
    122, 158, 126;
    179, 239, 178;
    % 200, 255, 200;
    255, 255, 255;
]./255;

figure; hold on;
[k_10] = convhull(successes_10);
[k_20] = convhull(successes_20);
[k_30] = convhull(successes_30);
[k_40] = convhull(successes_40);
[k_50] = convhull(successes_50);
scatter(successes_50(:, 1), successes_50(:, 2), 'MarkerFaceColor', colors(1, :), 'MarkerEdgeColor', 'None', 'MarkerFaceAlpha', 1);
scatter(successes_40(:, 1), successes_40(:, 2), 'MarkerFaceColor', colors(2, :), 'MarkerEdgeColor', 'None', 'MarkerFaceAlpha', 1);
scatter(successes_30(:, 1), successes_30(:, 2), 'MarkerFaceColor', colors(3, :), 'MarkerEdgeColor', 'None', 'MarkerFaceAlpha', 1);
scatter(successes_20(:, 1), successes_20(:, 2), 'MarkerFaceColor', colors(4, :), 'MarkerEdgeColor', 'None', 'MarkerFaceAlpha', 1);
scatter(successes_10(:, 1), successes_10(:, 2), 'MarkerFaceColor', colors(5, :), 'MarkerEdgeColor', 'None', 'MarkerFaceAlpha', 1);

qw{1} = plot(nan, 'o', 'MarkerFaceColor', colors(1, :), 'MarkerEdgeColor', 'k');
qw{2} = plot(nan, 'o', 'MarkerFaceColor', colors(2, :), 'MarkerEdgeColor', 'k');
qw{3} = plot(nan, 'o', 'MarkerFaceColor', colors(3, :), 'MarkerEdgeColor', 'k');
qw{4} = plot(nan, 'o', 'MarkerFaceColor', colors(4, :), 'MarkerEdgeColor', 'k');
qw{5} = plot(nan, 'o', 'MarkerFaceColor', colors(5, :), 'MarkerEdgeColor', 'k');
legend([qw{:}], "50 N", "40 N", "30 N", "20 N", "10 N", 'Location', 'NorthEast');

xlabel('Linear momentum (kg m/s)');
ylabel('Angular momentum (kg m^2/s)');