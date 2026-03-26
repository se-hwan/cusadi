import casadi as ca
import math
import matplotlib.pyplot as plt
import numpy as np
from cusadi.models import PinocchioModel
from cusadi.optimization import OptimizationProblem
from cusadi.visualization import Visualizer3D
from cusadi.controllers.utils import ActuatorSpec

ACTUATOR_SPECS = {
    "hip_yaw": ActuatorSpec(1.6841e-4, 6.0),
    "hip_abad": ActuatorSpec(1.6841e-4, 6.0),
    "hip_pitch": ActuatorSpec(5.548e-4, 6.0),
    "knee": ActuatorSpec(5.548e-4, 12.0),
    "ankle": ActuatorSpec(1.6841e-4, 12.0),
    "shoulder_pitch": ActuatorSpec(1.6841e-4, 6.0),
    "shoulder_abad": ActuatorSpec(1.6841e-4, 6.0),
    "shoulder_yaw": ActuatorSpec(1.6841e-4, 6.0),
    "elbow": ActuatorSpec(1.6841e-4, 9.0),
}

HUMANOID_ACTUATOR_ORDER = [
    "hip_yaw", "hip_abad", "hip_pitch", "knee", "ankle",
    "hip_yaw", "hip_abad", "hip_pitch", "knee", "ankle",
    "shoulder_pitch", "shoulder_abad", "shoulder_yaw", "elbow",
    "shoulder_pitch", "shoulder_abad", "shoulder_yaw", "elbow",
]

class MITHumanoidModelPredictiveController:
    problem_built: bool = False

    def __init__(self, urdf_filepath):
        self.urdf_filepath = urdf_filepath
        self.model = PinocchioModel(self.urdf_filepath, is_floating=True)
        self.last_soln = None

    def set_parameters(self, parameter_vec, field: str = None, val=None, env_ids=None):
        """Write values into a parameter vector at the indices of the named field.

        Args:
            parameter_vec: torch.Tensor (N_envs, n_pars), or np.ndarray (n_pars,) / (n_pars, 1)
            field:    parameter name (key in self.formulation.parameters);
                      if None, fills parameter_vec with each parameter's default (init) values
            val:      value(s) to write — must be broadcastable to the field's flat size
            env_ids:  row indices to update (tensor case only); None means all envs
        """
        try:
            import torch
            is_torch = isinstance(parameter_vec, torch.Tensor)
        except ImportError:
            is_torch = False

        if field is None:
            for param in self.formulation.parameters.values():
                idx = param.idx
                val_flat = param.init_vec                      # already col-major flattened
                if is_torch:
                    val_t = torch.as_tensor(val_flat, dtype=parameter_vec.dtype,
                                            device=parameter_vec.device)
                    idx_t = torch.as_tensor(idx, dtype=torch.long, device=parameter_vec.device)
                    if env_ids is None:
                        parameter_vec[:, idx_t] = val_t
                    else:
                        env_t = torch.as_tensor(env_ids, dtype=torch.long, device=parameter_vec.device)
                        parameter_vec[env_t[:, None], idx_t[None, :]] = val_t
                else:
                    parameter_vec.flat[idx] = val_flat
            return parameter_vec

        idx = self.formulation.parameters[field].idx          # contiguous int array

        if is_torch:
            idx_t = torch.as_tensor(idx, dtype=torch.long, device=parameter_vec.device)
            n_rows = parameter_vec.shape[0]
            if env_ids is None:
                env_t = torch.arange(n_rows, dtype=torch.long, device=parameter_vec.device)
            else:
                env_t = torch.as_tensor(env_ids, dtype=torch.long, device=parameter_vec.device)

            if isinstance(val, torch.Tensor):
                val_t = val.to(dtype=parameter_vec.dtype, device=parameter_vec.device)
                if val_t.ndim == 1:
                    val_t = val_t.unsqueeze(0)
                elif val_t.ndim > 2:
                    val_t = val_t.transpose(-2, -1).reshape(val_t.shape[0], -1)
                if val_t.shape[0] == 1 and env_t.numel() > 1:
                    val_t = val_t.expand(env_t.numel(), -1)
                elif val_t.shape[0] != env_t.numel():
                    raise ValueError(
                        f"Tensor value for parameter '{field}' has batch dimension {val_t.shape[0]} "
                        f"but expected 1 or {env_t.numel()}."
                    )
                if val_t.shape[1] != len(idx):
                    raise ValueError(
                        f"Tensor value for parameter '{field}' has flattened size {val_t.shape[1]} "
                        f"but expected {len(idx)}."
                    )
                parameter_vec[env_t[:, None], idx_t[None, :]] = val_t
            else:
                val_flat = np.asarray(val).flatten(order='F')         # col-major, matching CasADi layout
                val_t = torch.as_tensor(val_flat, dtype=parameter_vec.dtype,
                                        device=parameter_vec.device)
                parameter_vec[env_t[:, None], idx_t[None, :]] = val_t
        else:
            val_flat = np.asarray(val).flatten(order='F')         # col-major, matching CasADi layout
            parameter_vec.flat[idx] = val_flat
        return parameter_vec

    def set_controller_constants(self):
        self.fn_opts = {'cse': True, 'post_expand': True}
        self.N_HORIZON = 10
        self.N_CONTACT = 4
        self.N_F = self.N_CONTACT * 3
        self.N_FB = 6
        self.N_Q_LEG = 5
        self.N_Q_ARM = 4
        self.N_Q = self.model.cpin_model.nq # num. gen. coord.
        self.N_V = self.model.cpin_model.nv # num. gen. vel.
        self.N_STAGE = 2 * self.N_V + self.N_F
        self.N_DV = self.N_STAGE * self.N_HORIZON # num. decision vars
        self.dq_idx = np.arange(self.N_V)
        self.v_idx = self.dq_idx + self.N_V
        self.F_idx = 2*self.N_V + np.arange(self.N_F)
        self.dt = 1.0/30
        self.t_horizon = self.N_HORIZON * self.dt
        self.right_toe_idx = [0, 1, 2]
        self.left_toe_idx = [3, 4, 5]
        self.right_heel_idx = [6, 7, 8]
        self.left_heel_idx = [9, 10, 11]
        self.contact_x_idx = [0, 3, 6, 9]
        self.contact_y_idx = [1, 4, 7, 10]
        self.contact_z_idx = [2, 5, 8, 11]
        self.collision_radius = 0.15
        self.offset_lateral_stance = 0.2
        self.q_nominal = np.array([
            0.0, 0.0, 0.62, 0.0, 0.0, 0.0, 1.0,
            0.0, -0.1, -0.724757, 1.412282, -0.68752,
            0.0,  0.1, -0.724757, 1.412282, -0.68752,
            0.0, -0.1, 0.0, 0.0,
            0.0, 0.1, 0.0, 0.0
            ])
        self.q_nominal_traj = np.tile(self.q_nominal, (self.N_HORIZON, 1)).T
        self.q_max = self.model.pin_model.upperPositionLimit
        self.q_min = self.model.pin_model.lowerPositionLimit
        self.qd_lim = self.model.pin_model.velocityLimit
        self.tau_lim = self.model.pin_model.effortLimit
        joint_gear_ratio = np.array(
            [ACTUATOR_SPECS[name].gear_ratio for name in HUMANOID_ACTUATOR_ORDER],
            dtype=float,
        )
        joint_rotor_inertia = np.array(
            [ACTUATOR_SPECS[name].rotor_inertia for name in HUMANOID_ACTUATOR_ORDER],
            dtype=float,
        )
        joint_armature = np.array(
            [ACTUATOR_SPECS[name].armature for name in HUMANOID_ACTUATOR_ORDER],
            dtype=float,
        )
        self.rotor_inertia = np.concatenate([np.zeros(6), joint_rotor_inertia])
        self.gear_ratio = np.concatenate([np.zeros(6), joint_gear_ratio])
        self.armature = np.concatenate([np.zeros(6), joint_armature])
        self.model.set_model_parameter("rotorInertia", self.rotor_inertia)
        self.model.set_model_parameter("rotorGearRatio", self.gear_ratio)
        self.model.set_model_parameter("armature", self.armature)
        all_frames = self.model.get_frame_names()
        self.end_eff_frames = ['right_toe', 'left_toe',
                               'right_heel', 'left_heel']
        self.model.set_end_effector_frames(self.end_eff_frames)

    def build(self):
        self.formulation = OptimizationProblem('mit_humanoid_mpc')
        self.set_controller_constants()
        self.build_bezier_swing_fn()

        # Decision variables
        X_opt = self.formulation.add_variable(self.N_STAGE, self.N_HORIZON, 'X_opt')
        dq_opt = X_opt[self.dq_idx, :]
        v_opt = X_opt[self.v_idx, :]
        F_opt = X_opt[self.F_idx, :]

        # Parameters
        q_0 = self.formulation.add_parameter(self.N_Q, 1, 'q_0', init_value=self.q_nominal)
        v_0 = self.formulation.add_parameter(self.N_V, 1, 'v_0')
        q_nom = self.formulation.add_parameter(self.N_Q, 1, 'q_nom', init_value=self.q_nominal)
        q_des = self.formulation.add_parameter(self.N_Q, self.N_HORIZON, 'q_des', init_value=self.q_nominal_traj)
        v_des = self.formulation.add_parameter(self.N_V, self.N_HORIZON, 'v_des')
        F_des = self.formulation.add_parameter(self.N_F, self.N_HORIZON, 'F_des')
        phi_traj = self.formulation.add_parameter(self.N_CONTACT, self.N_HORIZON, 'phi_traj')
        contact_traj = self.formulation.add_parameter(self.N_CONTACT, self.N_HORIZON, 'contact_traj', init_value=np.ones((self.N_CONTACT, self.N_HORIZON)))
        Q_fb = self.formulation.add_parameter(self.N_FB, 1, 'Q_fb', init_value=np.array([0, 0, 1000, 300, 600, 300]))
        Q_leg = self.formulation.add_parameter(self.N_Q_LEG, 1, 'Q_leg', init_value=np.array([100, 100, 50, 25, 20]))
        Q_arm = self.formulation.add_parameter(self.N_Q_ARM, 1, 'Q_arm', init_value=np.array([5, 5, 5, 5]))
        Qv_fb = self.formulation.add_parameter(self.N_FB, 1, 'Qd_fb', init_value=np.array([100, 100, 100, 800, 800, 400]))
        Qv_leg = self.formulation.add_parameter(self.N_Q_LEG, 1, 'Qd_leg', init_value=np.array([5, 5, 0.12, 0.12, 0.12]))
        Qv_arm = self.formulation.add_parameter(self.N_Q_ARM, 1, 'Qd_arm', init_value=np.array([2.5, 2.5, 2.5, 2.5]))
        Qa_fb = self.formulation.add_parameter(self.N_FB, 1, 'Qa_fb', init_value=np.array([0] * self.N_FB))
        Qa_leg = self.formulation.add_parameter(self.N_Q_LEG, 1, 'Qa_leg', init_value=np.array([0] * self.N_Q_LEG))
        Qa_arm = self.formulation.add_parameter(self.N_Q_ARM, 1, 'Qa_arm', init_value=np.array([0] * self.N_Q_ARM))
        R_F = self.formulation.add_parameter(self.N_F, 1, 'R_F', init_value=np.array([1e-5] * self.N_F))

        Q_q = ca.diag(ca.vertcat(Q_fb, Q_leg, Q_leg, Q_arm, Q_arm))
        Q_v = ca.diag(ca.vertcat(Qv_fb, Qv_leg, Qv_leg, Qv_arm, Qv_arm))
        Q_a = ca.diag(ca.vertcat(Qa_fb, Qa_leg, Qa_leg, Qa_arm, Qa_arm))

        # Useful expressions
        q_traj = self.model.get_integrated_states(q_nom, dq_opt)
        v_traj = v_opt
        v_traj_appended = ca.horzcat(v_0, v_traj)
        a_traj = (v_traj_appended[:, 1:] - v_traj_appended[:, :-1]) / self.dt
        F_traj = F_opt

        [h_contact_des_traj, v_foot_des_traj, a_foot_des_traj] = self.fn_swing_traj(
            phi_traj, 0.02, -0.02, 0.05)
        fwd_kin_traj = self.model.get_forward_kinematics(q=q_traj,
                                                         v=v_traj,
                                                         frames=self.end_eff_frames)
        cost = 0
        for k in range(self.N_HORIZON):
            # ! Cost
            q_k = q_traj[:, k]
            v_k = v_traj[:, k]
            a_k = a_traj[:, k]
            F_k = F_traj[:, k]
            q_des_k = q_des[:, k]
            v_des_k = v_des[:, k]
            F_des_k = F_des[:, k]

            q_err_k = self.model.get_state_error(q_k, q_des_k)
            v_err_k = v_des_k - v_k
            F_err_k = F_des_k - F_k
            cost += q_err_k.T @ Q_q @  q_err_k * self.dt
            cost += v_err_k.T @ Q_v @ v_err_k * self.dt
            cost += a_k.T @ Q_a @ a_k * self.dt
            cost += F_err_k.T @ ca.diag(R_F) @ F_err_k * self.dt

            # ************ Dynamics and integration ************* #
            if k == 0:
                q_prev = q_0
                v_prev = v_0
                a_prev = ca.MX(self.N_V, 1)
            else:
                q_prev = q_traj[:, k-1]
                v_prev = v_traj[:, k-1]
                a_prev = a_traj[:, k-1]
            v_int = v_prev + a_prev * self.dt
            q_integrated = self.model.get_integrated_states(q_prev, v_int * self.dt)
            q_gap = self.model.get_state_error(q_k, q_integrated)
            self.formulation.add_equality_constraint(q_gap, f'pos_integration_{k}')
            tau_inertial_k = self.model.get_inverse_dynamics(q_k, v_k, a_k)[:6] # M*qdd + h(q, qd)
            tau_external_k = self.model.get_external_wrench(q_k, F_k*self.model.bodyweight,
                                                            self.end_eff_frames)[:6]
            self.formulation.add_equality_constraint(tau_inertial_k - tau_external_k, f'dynamics_{k}')

            # ************ Foot swing ************* #
            p_contact_k = ca.vertcat(fwd_kin_traj['p_right_toe'][:, k],
                                     fwd_kin_traj['p_left_toe'][:, k],
                                     fwd_kin_traj['p_right_heel'][:, k],
                                     fwd_kin_traj['p_left_heel'][:, k]
                                     )
            h_contact_des_k = h_contact_des_traj[:, k]
            h_contact_k = p_contact_k[self.contact_z_idx] # ? How to deal with this for a reference?
            self.formulation.add_equality_constraint(h_contact_k - h_contact_des_k, f"foot_swing_{k}")

            # ************ Contact ************* #
            contact_k = contact_traj[:, k]
            F_x = F_k[self.contact_x_idx]
            F_y = F_k[self.contact_y_idx]
            F_z = F_k[self.contact_z_idx]
            v_contact_k = ca.vertcat(
                fwd_kin_traj['v_right_toe'][:, k],
                fwd_kin_traj['v_left_toe'][:, k],
                fwd_kin_traj['v_right_heel'][:, k],
                fwd_kin_traj['v_left_heel'][:, k],
            )
            v_foot_x_k = v_contact_k[self.contact_x_idx]
            v_foot_y_k = v_contact_k[self.contact_y_idx]
            mu = 0.7
            self.formulation.add_inequality_constraint(
                F_x, f'friction_x_{k}', lb=-mu*F_z/np.sqrt(2), ub=mu*F_z/np.sqrt(2))
            self.formulation.add_inequality_constraint(
                F_y, f'friction_y_{k}', lb=-mu*F_z/np.sqrt(2), ub=mu*F_z/np.sqrt(2))
            self.formulation.add_inequality_constraint(
                F_z, f'nonnegative_{k}', lb=ca.MX(self.N_CONTACT, 1))
            self.formulation.add_equality_constraint(F_z * (1.0  - contact_k), f'contact_{k}')
            self.formulation.add_equality_constraint(v_foot_x_k * contact_k, f'contact_x_vel_{k}')
            self.formulation.add_equality_constraint(v_foot_y_k * contact_k, f'contact_y_vel_{k}')

            # ************ Lateral foot spacing in yaw-aligned body frame ************* #
            q_fb_k = q_k[:7]
            qx, qy, qz, qw = q_fb_k[3], q_fb_k[4], q_fb_k[5], q_fb_k[6]
            siny_cosp = 2 * (qw * qz + qx * qy)
            cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
            body_yaw = ca.atan2(siny_cosp, cosy_cosp)
            c_yaw = ca.cos(body_yaw)
            s_yaw = ca.sin(body_yaw)
            b_R_w = ca.vertcat(
                ca.horzcat(c_yaw, s_yaw, 0),
                ca.horzcat(-s_yaw, c_yaw, 0),
                ca.horzcat(0, 0, 1),
            )

            p_right_toe_body = b_R_w @ p_contact_k[self.right_toe_idx]
            p_left_toe_body = b_R_w @ p_contact_k[self.left_toe_idx]
            p_right_heel_body = b_R_w @ p_contact_k[self.right_heel_idx]
            p_left_heel_body = b_R_w @ p_contact_k[self.left_heel_idx]

            dist_toe_lateral = (
                p_left_toe_body[1] - p_right_toe_body[1]
                - self.collision_radius**2
                - self.offset_lateral_stance
            )
            dist_heel_lateral = (
                p_left_heel_body[1] - p_right_heel_body[1]
                - self.collision_radius**2
                - self.offset_lateral_stance
            )
            self.formulation.add_inequality_constraint(
                dist_toe_lateral, f'toe_lateral_spacing_{k}', lb=ca.MX(1, 1))
            self.formulation.add_inequality_constraint(
                dist_heel_lateral, f'heel_lateral_spacing_{k}', lb=ca.MX(1, 1))

        self.formulation.set_objective(cost)
        self.formulation.post_process()
        self.problem_built = True

    def build_bezier_swing_fn(self):
        t_swing = ca.MX.sym('t_swing', 1, 1)
        v_TO = ca.MX.sym('v_TO', 1, 1)
        v_TD = ca.MX.sym('v_TD', 1, 1)
        h_swing = ca.MX.sym('h_swing', 1, 1)
        P_sym = ca.MX(5, 1)
        P_sym[2] = v_TO/4
        P_sym[3] = 8*(h_swing - v_TO/16 + v_TD/16)/3
        P_sym[4] = -v_TD/4
        fn_bezier = lambda t, P: sum(
            math.comb(P.shape[0] - 1, i) * (1 - t) ** (P.shape[0] - 1 - i) * t ** i * P[i]
            for i in range(P.shape[0])
        )
        B_t = fn_bezier(t_swing, P_sym)
        B_dot_t = ca.jacobian(B_t, t_swing)
        B_ddot_t = ca.jacobian(B_dot_t, t_swing)
        fn_swing_traj = ca.Function('fn_swing_traj',
            [t_swing, v_TO, v_TD, h_swing], [B_t, B_dot_t, B_ddot_t],
            ['t_swing', 'v_TO', 'v_TD', 'h_swing'], ['B_t', 'B_dot_t', 'B_ddot_t'],
            self.fn_opts)
        fn_swing_traj = fn_swing_traj.map(self.N_HORIZON, 'serial')

        t_traj = ca.MX.sym('t_traj', self.N_CONTACT, self.N_HORIZON)
        h_swing_traj = ca.MX(self.N_CONTACT, self.N_HORIZON)
        v_swing_traj = ca.MX(self.N_CONTACT, self.N_HORIZON)
        a_swing_traj = ca.MX(self.N_CONTACT, self.N_HORIZON)

        for i in range(self.N_CONTACT):
            [B_traj, B_dot_traj, B_ddot_traj] = fn_swing_traj(t_traj[i, :], v_TO, v_TD, h_swing)
            h_swing_traj[i, :] = B_traj
            v_swing_traj[i, :] = B_dot_traj
            a_swing_traj[i, :] = B_ddot_traj
        self.fn_swing_traj = ca.Function('fn_swing_traj',
            [t_traj, v_TO, v_TD, h_swing], [h_swing_traj, v_swing_traj, a_swing_traj],
            ['t_traj', 'v_TO', 'v_TD', 'h_swing'], ['h_swing_traj', 'v_swing_traj', 'a_swing_traj'],
            self.fn_opts)
        return self.fn_swing_traj

    def build_contact_schedule_fn(self):
        """Build CasADi function for computing contact schedule and swing phase over the horizon.
          - Advances phase each step by dt/t_period using the active gait.
          - Switches from current_gait to queued_gait at the first horizon step
            where t_horizon >= t_period_remaining.
          - At that transition step (if the gait changed), resets phase to the
            queued gait's phase_offset rather than continuing to advance.

        Inputs
        ------
        phase               (N_CONTACT,)   current phase per contact [0, 1)
        t_horizon           (N_HORIZON,)   cumulative time at each horizon step [s]
        t_period_remaining  scalar         time remaining in current gait period [s]
        t_period_curr       scalar         period of current gait [s]
        phase_offset_curr   (N_CONTACT,)   phase offset of current gait
        phase_switch_curr   (N_CONTACT,)   contact-to-swing switch phase of current gait
        t_period_queue      scalar         period of queued gait [s]
        phase_offset_queue  (N_CONTACT,)   phase offset of queued gait
        phase_switch_queue  (N_CONTACT,)   contact-to-swing switch phase of queued gait
        gait_changed        scalar         1 if current != queued gait, else 0

        Outputs
        -------
        contact_schedule    (N_CONTACT * N_HORIZON,)  flattened column-major, 1=contact 0=swing
        phase_swing         (N_CONTACT * N_HORIZON,)  flattened column-major, swing progress in [0,1]
        """
        N = self.N_HORIZON
        N_GC = self.N_CONTACT

        # ---- symbolic inputs ----
        phase = ca.MX.sym('phase', N_GC, 1)
        t_horizon = ca.MX.sym('t_horizon', N, 1)
        t_period_remaining = ca.MX.sym('t_period_remaining')

        t_period_curr = ca.MX.sym('t_period_curr')
        phase_offset_curr = ca.MX.sym('phase_offset_curr', N_GC, 1)
        phase_switch_curr = ca.MX.sym('phase_switch_curr', N_GC, 1)

        t_period_queue = ca.MX.sym('t_period_queue')
        phase_offset_queue = ca.MX.sym('phase_offset_queue', N_GC, 1)
        phase_switch_queue = ca.MX.sym('phase_switch_queue', N_GC, 1)

        gait_changed = ca.MX.sym('gait_changed')  # 1 if gaits differ

        # dt_horizon = diff([0; t_horizon])  (MATLAB 1-indexed, same result)
        dt_horizon = ca.vertcat(t_horizon[0], t_horizon[1:] - t_horizon[:-1])

        contact_cols = []
        swing_cols = []
        phase_prev = phase

        for j in range(N):
            t_j = t_horizon[j]
            dt_j = dt_horizon[j]

            # Select gait parameters for this step
            use_current = t_j < t_period_remaining
            t_period_j      = ca.if_else(use_current, t_period_curr,      t_period_queue)
            phase_offset_j  = ca.if_else(use_current, phase_offset_curr,  phase_offset_queue)
            phase_switch_j  = ca.if_else(use_current, phase_switch_curr,  phase_switch_queue)

            # Advance phase
            phase_j = phase_prev + dt_j / t_period_j

            # Detect the first transition step: t_horizon[j] >= t_period_remaining
            # AND t_horizon[j-1] < t_period_remaining  (or j == 0)
            if j == 0:
                is_first_transition = t_j >= t_period_remaining
            else:
                is_first_transition = ca.logic_and(
                    t_j >= t_period_remaining,
                    t_horizon[j - 1] < t_period_remaining
                )

            # At the transition, reset to queued gait's phase_offset if gait changed
            reset_phase = ca.logic_and(is_first_transition, gait_changed > 0.5)
            phase_j_mod = ca.fmod(phase_j, 1.0)
            phase_schedule_j = ca.if_else(reset_phase, phase_offset_j, phase_j_mod)

            # Per-contact swing phase and contact state
            contact_list = []
            swing_list = []
            for i in range(N_GC):
                p_i  = phase_schedule_j[i]
                ps_i = phase_switch_j[i]
                in_swing = p_i >= ps_i
                contact_list.append(ca.if_else(in_swing, 0.0, 1.0))
                swing_progress = (p_i - ps_i) / (1.0 - ps_i)
                swing_list.append(ca.if_else(in_swing, swing_progress, 1.0))

            contact_cols.append(ca.vertcat(*contact_list))
            swing_cols.append(ca.vertcat(*swing_list))
            phase_prev = phase_schedule_j

        contact_schedule = ca.horzcat(*contact_cols)  # N_GC x N
        phase_swing      = ca.horzcat(*swing_cols)    # N_GC x N

        self.fn_contact_schedule = ca.Function(
            'fn_contact_schedule',
            [phase, t_horizon, t_period_remaining,
             t_period_curr, phase_offset_curr, phase_switch_curr,
             t_period_queue, phase_offset_queue, phase_switch_queue,
             gait_changed],
            [ca.vec(contact_schedule), ca.vec(phase_swing)],
            ['phase', 't_horizon', 't_period_remaining',
             't_period_curr', 'phase_offset_curr', 'phase_switch_curr',
             't_period_queue', 'phase_offset_queue', 'phase_switch_queue',
             'gait_changed'],
            ['contact_schedule', 'phase_swing'],
            self.fn_opts
        )
        return self.fn_contact_schedule

    def build_desired_trajectory_fn(self):
        """Build CasADi function for a simple command-tracking desired trajectory.

        The command is ``[z_des, v_x_body, v_y_body, w_z]``. The desired base
        pose is propagated over the horizon in the horizontal plane while using
        a quaternion representation for orientation. Joint references are built
        from the nominal posture with light arm shaping, and contact forces are
        regularized around the previous solution at the first step and then
        evenly distributed across active contacts afterward.
        """

        q_0_sym = ca.MX.sym('q_0', self.N_Q, 1)
        cmd_sym = ca.MX.sym('cmd', 4, 1)
        dt_sym = ca.MX.sym('dt', self.N_HORIZON, 1)
        contact_sym = ca.MX.sym('contact_traj', self.N_CONTACT * self.N_HORIZON, 1)
        F_prev_sym = ca.MX.sym('F_prev', self.N_F, 1)

        q_des_traj = ca.MX(ca.DM(self.q_nominal_traj))
        qd_des_traj = ca.MX.zeros(self.N_V, self.N_HORIZON)
        F_des_traj = ca.MX.zeros(self.N_F, self.N_HORIZON)
        contact_traj = ca.reshape(contact_sym, self.N_CONTACT, self.N_HORIZON)

        # Posture shaping about the nominal configuration.
        q_des_traj[2, :] = cmd_sym[0]
        q_des_traj[[17, 21], :] = 0.2
        q_des_traj[[20, 24], :] = -0.5

        # Free-flyer tangent coordinates in Pinocchio are [v_xyz, w_xyz, ...].
        qd_des_traj[0, :] = cmd_sym[1]
        qd_des_traj[1, :] = cmd_sym[2]
        qd_des_traj[5, :] = cmd_sym[3]

        for k in range(self.N_HORIZON):
            if k == 0:
                x_prev = q_0_sym[0]
                y_prev = q_0_sym[1]
                qx_prev = q_0_sym[3]
                qy_prev = q_0_sym[4]
                qz_prev = q_0_sym[5]
                qw_prev = q_0_sym[6]
            else:
                x_prev = q_des_traj[0, k - 1]
                y_prev = q_des_traj[1, k - 1]
                qx_prev = q_des_traj[3, k - 1]
                qy_prev = q_des_traj[4, k - 1]
                qz_prev = q_des_traj[5, k - 1]
                qw_prev = q_des_traj[6, k - 1]

            yaw_prev = ca.atan2(
                2 * (qw_prev * qz_prev + qx_prev * qy_prev),
                1 - 2 * (qy_prev * qy_prev + qz_prev * qz_prev),
            )
            yaw_k = yaw_prev + cmd_sym[3] * dt_sym[k]

            c_yaw = ca.cos(yaw_prev)
            s_yaw = ca.sin(yaw_prev)
            x_k = x_prev + dt_sym[k] * (cmd_sym[1] * c_yaw - cmd_sym[2] * s_yaw)
            y_k = y_prev + dt_sym[k] * (cmd_sym[1] * s_yaw + cmd_sym[2] * c_yaw)

            q_des_traj[0, k] = x_k
            q_des_traj[1, k] = y_k
            q_des_traj[3, k] = 0.0
            q_des_traj[4, k] = 0.0
            q_des_traj[5, k] = ca.sin(yaw_k / 2)
            q_des_traj[6, k] = ca.cos(yaw_k / 2)

            contact_k = contact_traj[:, k]
            Fz_k = self.model.bodyweight/2 * contact_k
            F_des_traj[self.contact_z_idx, k] = Fz_k

        F_des_traj[:, 0] = F_prev_sym

        self.fn_desired_trajectory = ca.Function(
            'fn_desired_trajectory',
            [q_0_sym, cmd_sym, dt_sym, contact_sym, F_prev_sym],
            [ca.vec(q_des_traj), ca.vec(qd_des_traj), ca.vec(F_des_traj)],
            ['q_0', 'cmd', 'dt', 'contact_traj', 'F_prev'],
            ['q_des', 'qd_des', 'F_des'],
            self.fn_opts,
        )
        return self.fn_desired_trajectory

    def build_torque_interpolation_fn(self):
        """Build CasADi function for torque interpolation between MPC nodes.

        Computes actuator torques, states, and forces at elapsed time ``t`` within
        the first MPC interval [0, dt_0], using linear interpolation between the
        current state and the first MPC horizon step, with finite-difference
        acceleration for the RNEA call.

        Compared to the MATLAB version, ``q_nom`` is an additional input because
        the Python MPC solution stores delta-q (tangent-space offsets from q_nom)
        rather than absolute joint configurations.  Forces in the solution are
        stored normalized by bodyweight and are re-scaled internally.

        Inputs
        ------
        t           scalar               elapsed time since MPC solve [s]
        dt          (N_HORIZON,)         per-step MPC timesteps [s]
        x_0         (N_Q + N_V,)         current state [q_0; v_0]
        q_nom       (N_Q,)               nominal q used during the MPC solve
        soln_MPC    (N_STAGE*N_HORIZON,) flattened column-major MPC solution

        Outputs
        -------
        tau     (N_JOINTS,)  actuated joint torques (floating-base DOFs excluded)
        q       (N_JOINTS,)  interpolated joint positions (indices 7: of q_interp)
        qd      (N_JOINTS,)  interpolated joint velocities (indices 6: of v_interp)
        F       (N_F,)       interpolated contact forces (normalized by bodyweight)
        qdd_0   (N_V,)       finite-difference generalized acceleration at t=0
        """

        # ---- symbolic inputs ----
        t_sym        = ca.MX.sym('t')
        dt_sym       = ca.MX.sym('dt', self.N_HORIZON, 1)
        x_0_sym      = ca.MX.sym('x_0', self.N_Q + self.N_V, 1)
        q_nom_sym    = ca.MX.sym('q_nom', self.N_Q, 1)
        soln_MPC_sym = ca.MX.sym('soln_MPC', self.N_STAGE * self.N_HORIZON, 1)

        # Reshape flat solution → (N_STAGE × N_HORIZON), column-major
        soln_MPC_traj = ca.reshape(soln_MPC_sym, self.N_STAGE, self.N_HORIZON)

        dt_0 = dt_sym[0]
        q_0  = x_0_sym[:self.N_Q]
        v_0  = x_0_sym[self.N_Q:]

        # Extract dq, velocity, and forces from the first two horizon steps
        dq_0 = soln_MPC_traj[:self.N_V, 0]            # tangent offset at step 0
        v_1  = soln_MPC_traj[self.N_V:2*self.N_V, 0]  # MPC velocity at step 0 (= qd_1)
        F_0  = soln_MPC_traj[2*self.N_V:, 0]           # normalized forces at step 0
        F_1  = soln_MPC_traj[2*self.N_V:, 1]           # normalized forces at step 1

        # q_1: integrate nominal q along the MPC delta-q for horizon step 0
        q_1 = self.model.get_integrated_states(q_nom_sym, dq_0)

        # Finite-difference acceleration (consistent with MPC dynamics constraint)
        qdd_0 = (v_1 - v_0) / dt_0

        # Linear interpolation between current state and first MPC step
        F_interp  = F_0 + t_sym * (F_1 - F_0) / dt_0
        qd_interp = v_0 + t_sym * (v_1 - v_0) / dt_0
        q_interp  = q_0 + t_sym * (q_1 - q_0) / dt_0

        # tau = RNEA(q, v, a) - Jc^T * F_contact
        # Forces in the solution are normalized; scale back by bodyweight
        tau_inertial = self.model.get_inverse_dynamics(q_interp, qd_interp, qdd_0)
        tau_contact  = self.model.get_external_wrench(
            q_interp, F_interp * self.model.bodyweight, self.end_eff_frames)
        tau_interp = tau_inertial - tau_contact

        self.fn_torque_interpolation = ca.Function(
            'fn_torque_interpolation',
            [t_sym, dt_sym, x_0_sym, q_nom_sym, soln_MPC_sym],
            [tau_interp[6:], q_interp[7:], qd_interp[6:], F_interp, qdd_0],
            ['t', 'dt', 'x_0', 'q_nom', 'soln_MPC'],
            ['tau', 'q', 'qd', 'F', 'qdd_0'],
            self.fn_opts
        )
        return self.fn_torque_interpolation

    def setup_solver(self, qp_solver, sqp_cfg, qp_cfg):
        # self.solver = self.formulation.setup_solver('ipopt')
        self.solver = self.formulation.setup_solver('sqp',
                                                    qp_solver=qp_solver,
                                                    sqp_cfg=sqp_cfg,
                                                    qp_cfg=qp_cfg
                                                    )
        self.x_default = np.zeros([self.formulation.n_x, 1])
        self.p_default = np.zeros([self.formulation.n_par, 1])
        self.last_soln = self.x_default.copy()

    def solve(self, x=None, p=None):
        x_guess = self.x_default if self.last_soln is None else self.last_soln
        x_eval = x_guess if x is None else x
        p_eval = self.p_default if p is None else p
        soln = self.solver.solve(x_eval, p_eval)
        self.last_soln = soln.copy()
        return soln

    def visualize(self):
        self.visualizer = Visualizer3D()
        self.visualizer.add_urdf(self.urdf_filepath, "mit_humanoid")
        self.visualizer.update_urdf("mit_humanoid",
                                    self.q_nominal[:3],
                                    self.q_nominal[3:7],
                                    self.q_nominal[7:],
        )



def plot_solution_summary(q_traj, F_traj, dt, n_contact):
    time = np.arange(q_traj.shape[1]) * dt
    fig_base, ax_base = plt.subplots(1, 1, figsize=(8, 2.6), constrained_layout=True)

    ax_base.plot(time, q_traj[0, :], label='x', linewidth=1.6)
    ax_base.plot(time, q_traj[1, :], label='y', linewidth=1.6)
    ax_base.plot(time, q_traj[2, :], label='z', linewidth=1.6)
    ax_base.set_ylabel('Base [m]')
    ax_base.set_xlabel('Time [s]')
    ax_base.set_title('Base position', fontsize=10)
    ax_base.legend(loc='best', ncol=3, fontsize=8)

    idx_FR = [0, 1, 2]
    idx_FL = [3, 4, 5]
    idx_BR = [6, 7, 8]
    idx_BL = [9, 10, 11]

    F_FR = F_traj[idx_FR, :]
    F_FL = F_traj[idx_FL, :]
    F_BR = F_traj[idx_BR, :]
    F_BL = F_traj[idx_BL, :]

    fig_grf, axs = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axs[0, 0].plot(F_FL.T)
    axs[0, 0].set_title("Front Left")
    axs[0, 1].plot(F_FR.T)
    axs[0, 1].set_title("Front Right")
    axs[1, 0].plot(F_BL.T)
    axs[1, 0].set_title("Back Left")
    axs[1, 1].plot(F_BR.T)
    axs[1, 1].set_title("Back Right")

    for ax in axs.flat:
        ax.set_ylabel("Force [N]")
        ax.grid(True)

    axs[1, 0].set_xlabel("Time step")
    axs[1, 1].set_xlabel("Time step")
    axs[0, 0].legend(["Fx", "Fy", "Fz"], loc="upper right")
    plt.suptitle("Ground Reaction Forces")
    plt.tight_layout()
    plt.show()
    return fig_base, fig_grf



# TODO: run with args (--compile for different behavior, visualize solution otherwise)



if __name__ == "__main__":
    controller = MITHumanoidModelPredictiveController('../../../extensions/mit_humanoid/assets/urdf/humanoid_full_sf.urdf')
    controller.build()
    controller.setup_solver(
        qp_solver='osqp',
        qp_cfg={'max_iter': 25},
        sqp_cfg={'max_iter': 1}
    )
    controller.formulation.set_parameter("q_0", controller.q_nominal)
    controller.formulation.set_parameter("q_nom", controller.q_nominal)
    controller.formulation.set_parameter("q_des", controller.q_nominal_traj)
    controller.formulation.set_parameter("contact_traj", np.ones((controller.N_CONTACT, controller.N_HORIZON)))
    p_eval = controller.formulation.get_parameter_vec()

    import sys
    import torch
    from cusadi.parallelization import parallelize_functions
    BATCH_SIZE = 1
    controller_fns = controller.solver.setup_parallelization('cudss',
                                                             batch_size=BATCH_SIZE,
                                                             precision='float',
                                                             dynamic_batching=False)
    controller_pfns = parallelize_functions(controller_fns,
                                            batch_size=1,
                                            precision='float',
                                            dynamic_batching=False)
    controller.solver.set_cusadi_functions(controller_pfns)
    x_tensor = torch.zeros(BATCH_SIZE, controller.N_DV, device='cuda', dtype=torch.float)
    p_tensor = torch.tensor(p_eval.T, device='cuda', dtype=torch.float)
    p_tensor = torch.tile(p_tensor, (BATCH_SIZE, 1)).contiguous().float()
    torch.set_printoptions(precision=20, threshold=sys.maxsize)
    out = controller.solver.solve_parallelized(x_tensor, p_tensor)
    print(out)

    # soln_opt = controller.solver.qp_backend.solve_with_osqp(np.zeros((120, 1)), p_eval)
    # # print(soln_opt.T)
    # controller.solver.qp_backend.build_parallel_fns('ldl')
    # soln_opt = controller.solver.solve(np.zeros((120, 1)), p_eval, solve_method='custom')
    # print("SOLN OPT: ", soln_opt.T)

    # import torch
    # controller.solver.qp_backend.linsys_method = 'cudss'
    # print(out[0, :])

    # soln_opt = controller.solve(x=None, p=p_eval)
    # soln_traj = soln_opt.reshape(-1, controller.N_HORIZON, order='F')
    # dq_traj = soln_traj[controller.dq_idx, :]
    # v_traj = soln_traj[controller.v_idx, :]
    # F_traj = soln_traj[controller.F_idx, :]
    # q_traj = controller.model.get_integrated_states(controller.q_nominal, dq_traj).toarray()
    # plot_solution_summary(q_traj, F_traj, controller.dt, controller.N_CONTACT)

    # pos_traj = q_traj[:3, :]
    # ori_traj = q_traj[3:7, :]
    # jnt_traj = q_traj[7:, :]
    # controller.visualize()
    # controller.visualizer.add_urdf_trajectory(
    #     'mit_humanoid', pos_traj, ori_traj, jnt_traj, dt=controller.dt)
    
    # while True:
    #     pass
