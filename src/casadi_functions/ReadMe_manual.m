
%% Load casadi files
CASADI_finiteStateMachine_FK = casadi.Function.load('CASADI_finiteStateMachine_FK.casadi');
CASADI_swingControl = casadi.Function.load('CASADI_swingControl.casadi');
CASADI_srb_nmpc_codegen_iter5 = casadi.Function.load('CASADI_srb_nmpc_codegen_iter5.casadi');
CASADI_srb_nmpc_codegen_iter10 = casadi.Function.load('CASADI_srb_nmpc_codegen_iter10.casadi');

eye_3 = [1,0,0; 0,1,0; 0,0,1];
horizon = 12; % mpc # of horizon = 12 (embedded in the code)

%% Finite State Machine
clock = 0; % simulation clock
bool_walk = 1; % 0: standing, 1: walking
v_d = [1;0;0]; % desired body velocity 
wZ_d = 0.0; % desired body yaw rate
gc = [0;0;0.6; 1;0;0;0; zeros(5,1); zeros(5,1)]; % [pBody; quatBody; encoders]
gv = [zeros(3,1); zeros(3,1); zeros(5,1); zeros(5,1)]; % [vBody; wBody (in body-frame); d_encoders]
dt_fsm = 0.001; % fsm time-step
Tst = 0.2; % stance time
Tsw = 0.1; % swing time
dt_mpc = 0.025; % mpc time-step
zc = 0.6; % desired body height
FSM = [0;0]; % 0: Stand, 1: Stance, 2: Swing
Ta = [0;0]; % Phase start time
Tb = [1;1]; % Phase end time
pf_trans_w_vec = [-0.0057; -0.0804; 0; -0.0057; 0.0804; 0]; % feet position (transition state)
Rf_trans_w_vec = [eye_3(:); eye_3(:)]; % feet orientation (transition state)
R_d = eye_3(:); % desired body orientaion
p_d = [-0.0057; 0; zc];
fixed_params_fsm = [dt_fsm;Tst;Tsw;dt_mpc;zc];
memo_params = [FSM;Ta;Tb;pf_trans_w_vec;Rf_trans_w_vec;R_d;p_d];

[FSM,s,x0,xd,pf,Rf,vf,wf,J_R,J_rot_R,pf_trans_w,Rf_trans_w,contact_pred,Rf_yaw,memo_params] = ...
    CASADI_finiteStateMachine_FK(clock,bool_walk,v_d,wZ_d,gc,gv,memo_params,fixed_params_fsm);

%% Swing Leg Controller
p_cap_max = 0.3; % foot limit in xy-direction
gain_fb = 1; % on/off Raibert heuristic feedback term
gain_fb_cent = 0; % on/off Raibert heuristic centrifugal term
bd2hip = [-0.005653;-0.082;-0.05735; -0.005653;0.082;-0.05735]; % body to hip yaw (in body frame)
dpz = 0.05; % swing height
hip_offset_y = 0; % desired foot offset in y-axis
Kp_sw = [3500;3500;3500];
Kd_sw = [40;40;40];
Kp_sw_rot = [70;70;70];
Kd_sw_rot = [0.8;0.8;0.8];
fixed_params_sw = [...
    Tst;Tsw;zc;p_cap_max;gain_fb;gain_fb_cent;bd2hip; ...
    dpz;hip_offset_y;Kp_sw;Kd_sw;Kp_sw_rot;Kd_sw_rot];

[tau_sw,pFootFrame_b] = ...
    CASADI_swingControl(FSM,s,x0,xd,pf,Rf,vf,wf,J_R,J_rot_R,pf_trans_w,Rf_trans_w,fixed_params_sw);

%% SRB-NMPC Controller
g = 9.81;
I = [0.4626; 0.0014; 0.0040; 0.0014; 0.3037; -0.0017; 0.0040; -0.0017; 0.2454];
m = 24.8885;
u_nominal = repmat([0;0;m*g/2;0;0;0;0;0;m*g/2;0;0;0],[horizon,1]); % [grf_right; grm_right; grf_left; grm_left]
Qx = [1;1;1; 5;5;50; 0.01;0.01;0.01; 0.2;0.2;0.1];  % MPC weight (state)
Qu = 1e-5*ones(12,1); % MPC weight (control input)
% Rf_yaw = [0;0]; % feet yaw angles (can be obatined from FSM function)
X_heel = 0.031295129;
X_toe = 0.076138708;
mu = 0.6;
fmax = 2000;
fmin = 1;
mu_initial = 100;
fixed_params_mpc = [...
    dt_mpc;g;I;m;
    Tst;Tsw;zc;p_cap_max;gain_fb;gain_fb_cent;bd2hip;
    Qx;Qu;1;
    X_heel;X_toe;mu;fmax;fmin]; % 54x1

parmas_reinitialize = [...
    x0;xd;pf;contact_pred;pFootFrame_b;Rf_yaw;0]; % 77x1
u_k = CASADI_srb_nmpc_codegen_iter5(...
    u_nominal,mu_initial,parmas_reinitialize,fixed_params_mpc);
% u_k = CASADI_srb_nmpc_codegen_iter10(...
%     u_nominal,mu_initial,parmas_reinitialize,fixed_params_mpc);
opti_u_penalty = reshape(u_k,[12,horizon]);
u_nominal = reshape([opti_u_penalty(:,2:end), opti_u_penalty(:,end)],12*horizon,1);