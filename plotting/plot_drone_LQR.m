clear; clc; close all;

%% Load and parse data
data = load('../data/drone_LQR_data.mat');
t = data.t;

traj_baseline = data.baseline.traj_baseline;
traj_Q_x = data.Q_x.traj_Q_x;
traj_R = data.R.traj_R;
traj_mass = data.mass.traj_mass;
F_lim = data.F_lim(1);
Q_default = data.Q_default(1, :);
R_default = data.R_default(1, :);
mass_default = data.mass_default(1, :);

N_TRAJ = length(traj_baseline(:, 1, 1));
N_ENVS = length(traj_baseline(1, :, 1));
N_ENVS_PER_IC = N_ENVS/3;

idx_IC_1 = 1;
idx_IC_2 = N_ENVS/3 + 1;
idx_IC_3 = 2*N_ENVS/3 + 1;

%% Plotting
subplot(1, 3, 1); hold on;
left_color = [0, 1, 0];
right_color = [1, 0, 0];
cmap = interp1([0, 1], [left_color; right_color], linspace(0, 1, N_ENVS_PER_IC));
for i = 1:N_ENVS_PER_IC
    plot(traj_Q_x(:, i, 1), traj_Q_x(:, i, 2), '-', 'Color', cmap(i, :))
    plot(traj_Q_x(:, i+idx_IC_2-1, 1), ...
         traj_Q_x(:, i+idx_IC_2-1, 2), '-', 'Color', cmap(i, :))
    plot(traj_Q_x(:, i+idx_IC_3-1, 1), ...
         traj_Q_x(:, i+idx_IC_3-1, 2), '-', 'Color', cmap(i, :))
end
plot(traj_Q_x(1, :, 1), traj_Q_x(1, :, 2), 'ko')
plot(traj_Q_x(end, :, 1), traj_Q_x(end, :, 2), 'kx')
plot(traj_baseline(:, idx_IC_1, 1), traj_baseline(:, idx_IC_1, 2), 'k-', 'LineWidth', 2)
plot(traj_baseline(:, idx_IC_2, 1), traj_baseline(:, idx_IC_2, 2), 'k-', 'LineWidth', 2)
plot(traj_baseline(:, idx_IC_3, 1), traj_baseline(:, idx_IC_3, 2), 'k-', 'LineWidth', 2)
title('$$Q_x$$', 'Interpreter', 'latex')
axis([-2.5, 1.5, -2.5, 2.5])
xlabel('x (m)'); ylabel('y (m)');
leg_Q_x{1} = plot(nan, '-', 'Color', left_color, 'LineWidth', 2);
leg_Q_x{2} = plot(nan, '-', 'Color', right_color, 'LineWidth', 2);
leg_Q_x{3} = plot(nan, 'k-', 'LineWidth', 2);
legend([leg_Q_x{:}], "$$Q_x = 0.05$$", "$$Q_x = 20$$", "$$Q_x = 1$$", 'Location', 'NorthEast', 'Interpreter', 'latex');

subplot(1, 3, 2); hold on;
left_color = [1, 0.5, 0];
right_color = [0, 0, 1];
cmap = interp1([0, 1], [left_color; right_color], linspace(0, 1, N_ENVS_PER_IC));
for i = 1:N_ENVS_PER_IC
    plot(traj_R(:, i, 1), traj_R(:, i, 2), '-', 'Color', cmap(i, :))
    plot(traj_R(:, i+idx_IC_2-1, 1), ...
         traj_R(:, i+idx_IC_2-1, 2), '-', 'Color', cmap(i, :))
    plot(traj_R(:, i+idx_IC_3-1, 1), ...
         traj_R(:, i+idx_IC_3-1, 2), '-', 'Color', cmap(i, :))
end
plot(traj_R(1, :, 1), traj_R(1, :, 2), 'ko')
plot(traj_R(end, :, 1), traj_R(end, :, 2), 'kx')
plot(traj_baseline(:, idx_IC_1, 1), traj_baseline(:, idx_IC_1, 2), 'k-', 'LineWidth', 2)
plot(traj_baseline(:, idx_IC_2, 1), traj_baseline(:, idx_IC_2, 2), 'k-', 'LineWidth', 2)
plot(traj_baseline(:, idx_IC_3, 1), traj_baseline(:, idx_IC_3, 2), 'k-', 'LineWidth', 2)
title('$$R$$', 'Interpreter', 'latex')
axis([-2.5, 1.5, -2.5, 2.5])
xlabel('x (m)'); ylabel('y (m)');
leg_R{1} = plot(nan, '-', 'Color', left_color, 'LineWidth', 2);
leg_R{2} = plot(nan, '-', 'Color', right_color, 'LineWidth', 2);
leg_R{3} = plot(nan, 'k-', 'LineWidth', 2);
legend([leg_R{:}], "$$R = 0.01$$", "$$R = 100$$", "$$R = 1$$", 'Location', 'NorthEast', 'Interpreter', 'latex');

subplot(1, 3, 3); hold on;
left_color = [1, 1, 0];
right_color = [0.5, 0, 1];
cmap = interp1([0, 1], [left_color; right_color], linspace(0, 1, N_ENVS_PER_IC));
for i = 1:N_ENVS_PER_IC
    plot(traj_mass(:, i, 1), traj_mass(:, i, 2), '-', 'Color', cmap(i, :))
    plot(traj_mass(:, i+idx_IC_2-1, 1), ...
         traj_mass(:, i+idx_IC_2-1, 2), '-', 'Color', cmap(i, :))
    plot(traj_mass(:, i+idx_IC_3-1, 1), ...
         traj_mass(:, i+idx_IC_3-1, 2), '-', 'Color', cmap(i, :))
end
plot(traj_mass(1, :, 1), traj_mass(1, :, 2), 'ko')
plot(traj_mass(end, :, 1), traj_mass(end, :, 2), 'kx')
plot(traj_baseline(:, idx_IC_1, 1), traj_baseline(:, idx_IC_1, 2), 'k-', 'LineWidth', 2)
plot(traj_baseline(:, idx_IC_2, 1), traj_baseline(:, idx_IC_2, 2), 'k-', 'LineWidth', 2)
plot(traj_baseline(:, idx_IC_3, 1), traj_baseline(:, idx_IC_3, 2), 'k-', 'LineWidth', 2)
title('$$m$$', 'Interpreter', 'latex')
axis([-2.5, 1.5, -2.5, 2.5])
xlabel('x (m)'); ylabel('y (m)');
leg_mass{1} = plot(nan, '-', 'Color', left_color, 'LineWidth', 2);
leg_mass{2} = plot(nan, '-', 'Color', right_color, 'LineWidth', 2);
leg_mass{3} = plot(nan, 'k-', 'LineWidth', 2);
legend([leg_mass{:}], "$$m = 0.25$$", "$$m = 4$$", "$$m = 1$$", 'Location', 'NorthEast', 'Interpreter', 'latex');


figure; hold on;
plot(traj_R(1:20:end, :, 1), traj_R(1:20:end, :, 4), 'r.-')