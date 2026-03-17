import numpy as np
import casadi as ca
import pinocchio as pin
import pinocchio.casadi as cpin
from .model import RobotModel

class PinocchioModel(RobotModel):
    urdf_filepath: str
    pin_model: pin.Model
    cpin_model: cpin.Model
    end_eff_frame_ids = []
    fn_opts = {}

    def __init__(self, urdf_filepath: str, is_floating: bool) -> None:
        super().__init__()
        self.urdf_filepath = urdf_filepath
        self.is_floating = is_floating
        if self.is_floating:
            self.pin_model = pin.buildModelFromUrdf(
                urdf_filepath, pin.JointModelFreeFlyer())
            self.joint_labels = [self.pin_model.names[i] for i in \
                                range(2, len(self.pin_model.names))]
        else:
            self.pin_model = pin.buildModelFromUrdf(urdf_filepath)
            self.joint_labels = [self.pin_model.names[i] for i in \
                                range(1, len(self.pin_model.names))]
        self.pin_data = pin.Data(self.pin_model)
        self.cpin_model = cpin.Model(self.pin_model)
        self.cpin_data = cpin.Data(self.cpin_model)

        # Useful quantities
        self.q_neutral = pin.neutral(self.pin_model)
        self.NJ = len(self.joint_labels)
        self.mass = 0
        for i in self.pin_model.inertias:
            self.mass += i.mass
        self.bodyweight = self.mass * 9.81
        # print("Pinocchio model built.")
        # print("Joint labels:", self.joint_labels)
    
    def set_end_effector_frames(self, frames: list|str):
        frame_ids, _ = self._parse_frame_labels(frames)
        self.end_eff_frame_ids = frame_ids
    
    def get_frame_names(self):
        names = []
        for f in self.pin_model.frames:
            names.append(f.name)
        return names

    def get_joint_labels(self):
        return self.joint_labels

    def get_forward_kinematics(self, q, v=None, a=None, frames=None):
        q, is_symbolic = self._preprocess_vec(q)
        if v is not None: v, _ = self._preprocess_vec(v)
        if a is not None: a, _ = self._preprocess_vec(a)
        model, data, pin_backend = self._get_backend(is_symbolic)
        frame_ids, frame_names = self._parse_frame_labels(frames)
        return_v = (v is not None) or (a is not None)
        return_a = a is not None
        zeros = ca.DM.zeros if is_symbolic else np.zeros
        if v is None: v = zeros(model.nv, 1) if is_symbolic else zeros(model.nv)
        if a is None: a = zeros(model.nv, 1) if is_symbolic else zeros(model.nv)
        build_function = not hasattr(self, 'fn_forward_kinematics') or \
                         getattr(self, '_fk_frame_names', None) != tuple(frame_names)

        if is_symbolic:
            if build_function:
                q_sym = ca.SX.sym('q', self.cpin_model.nq, 1)
                v_sym = ca.SX.sym('v', self.cpin_model.nv, 1)
                a_sym = ca.SX.sym('a', self.cpin_model.nv, 1)
                pin_backend.forwardKinematics(model, data, q_sym, v_sym, a_sym)
                pin_backend.updateFramePlacements(model, data)
                p_out = [data.oMf[f].translation for f in frame_ids]
                R_out = [ca.reshape(data.oMf[f].rotation, 9, 1) for f in frame_ids]
                v_out, w_out, a_out, alpha_out = [], [], [], []
                for f in frame_ids:
                    vel = pin_backend.getFrameVelocity(model, data, f, pin_backend.ReferenceFrame.WORLD)
                    acc = pin_backend.getFrameAcceleration(model, data, f, pin_backend.ReferenceFrame.WORLD)
                    v_out.append(vel.linear); w_out.append(vel.angular)
                    a_out.append(acc.linear); alpha_out.append(acc.angular)
                sym_out = [*p_out, *R_out, *v_out, *w_out, *a_out, *alpha_out]
                labels_out = sum(([f'{tag}_{name}' for name in frame_names]
                                  for tag in ('p', 'R', 'v', 'w', 'lin_acc', 'ang_acc')), [])
                self.fn_forward_kinematics = ca.Function(
                    'forward_kinematics', [q_sym, v_sym, a_sym], sym_out,
                    ['q', 'v', 'a'], labels_out, self.fn_opts)
                self._fk_frame_names = tuple(frame_names)

            out = self.fn_forward_kinematics.call({'q': q, 'v': v, 'a': a})
            if not return_v: return {k: val for k, val in out.items() if k.startswith(('p_', 'R_'))}
            if not return_a: return {k: val for k, val in out.items() if not k.startswith(('a_', 'alpha_'))}
            return out

        if return_a: pin_backend.forwardKinematics(model, data, q, v, a)
        elif return_v: pin_backend.forwardKinematics(model, data, q, v)
        else: pin_backend.forwardKinematics(model, data, q)
        pin_backend.updateFramePlacements(model, data)
        pos_frames = {name: data.oMf[i] for name, i in zip(frame_names, frame_ids)}
        if not return_v: return pos_frames
        vel_frames = {name: pin_backend.getFrameVelocity(
            model, data, f, pin_backend.ReferenceFrame.WORLD)
            for name, f in zip(frame_names, frame_ids)}
        if return_a:
            acc_frames = {name: pin_backend.getFrameAcceleration(
                model, data, f, pin_backend.ReferenceFrame.WORLD)
                for name, f in zip(frame_names, frame_ids)}
            return pos_frames, vel_frames, acc_frames
        return pos_frames, vel_frames

    def get_integrated_states(self, q, dq):
        q, is_symbolic = self._preprocess_vec(q)
        dq, _ = self._preprocess_vec(dq)
        model, data, pin_backend = self._get_backend(is_symbolic)
        build_function = not hasattr(self, 'fn_integration')
        if build_function:
            q_sym = ca.SX.sym('q', model.nq, 1)
            dq_sym = ca.SX.sym('dq', model.nv, 1)
            q_next = pin_backend.integrate(model, q_sym, dq_sym)
            self.fn_integration = ca.Function(
                'integration', [q_sym, dq_sym],
                [q_next], ['q', 'dq'], ['q_next'],
                self.fn_opts)
        return self.fn_integration(q, dq)

    def get_state_error(self, q_1, q_2):
        q_1, is_symbolic = self._preprocess_vec(q_1)
        q_2, _ = self._preprocess_vec(q_2)
        model, data, pin_backend = self._get_backend(is_symbolic)
        build_function = not hasattr(self, 'fn_state_error')
        if is_symbolic:
            if build_function:
                q_1_sym = ca.SX.sym('q_1', model.nq, 1)
                q_2_sym = ca.SX.sym('q_2', model.nq, 1)
                q_err = pin_backend.difference(model, q_1_sym, q_2_sym)
                self.fn_state_error = ca.Function(
                    'state_error', [q_1_sym, q_2_sym],
                    [q_err], ['q_1', 'q_2'], ['q_err'],
                    self.fn_opts)
            return self.fn_state_error(q_1, q_2)
    
    def get_jacobian(self, q, frames=None):
        frame_ids, frame_names = self._parse_frame_labels(frames)
        q, is_symbolic = self._preprocess_vec(q)
        build_function = not hasattr(self, 'fn_jacobian')
        # if is_symbolic:
        if build_function:
            model, data, pin_backend = self._get_backend(True)
            q_sym = ca.SX.sym('q', model.nq, 1)
            J = []
            for frame in frame_ids:
                J_frame = pin_backend.computeFrameJacobian(
                    model, data, q_sym, frame, pin_backend.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                # J_frame = pin_backend.computeFrameJacobian(
                #     model, data, q_sym, frame, pin_backend.ReferenceFrame.WORLD)
                J_frame = J_frame[:3, :]
                J.append(J_frame)
            J_kin = ca.sparsify(ca.vertcat(*J))
            self.fn_jacobian = ca.Function(
                'jacobian', [q_sym],
                [J_kin], ['q'], ['J_kin'],
                self.fn_opts)
        return self.fn_jacobian(q)

    def get_external_wrench(self, q, f_ext, frames=None):
        frame_ids, frame_names = self._parse_frame_labels(frames)
        q, is_symbolic = self._preprocess_vec(q)
        f_ext, _ = self._preprocess_vec(f_ext)
        model, data, pin_backend = self._get_backend(is_symbolic)
        build_function = not hasattr(self, 'fn_external_wrench')
        if is_symbolic:
            if build_function:
                q_sym = ca.SX.sym('q', model.nq, 1)
                f_ext_sym = ca.SX.sym('f_ext', 3*len(frame_ids), 1)
                J = []
                for frame in frame_ids:
                    J_frame = pin_backend.computeFrameJacobian(
                        model, data, q_sym, frame, pin_backend.ReferenceFrame.LOCAL_WORLD_ALIGNED)
                    # J_frame = pin_backend.computeFrameJacobian(
                    #     model, data, q_sym, frame, pin_backend.ReferenceFrame.WORLD)
                    J_frame = J_frame[:3, :]
                    J.append(J_frame)
                tau_ext = ca.vertcat(*J).T @ f_ext_sym
                self.fn_external_wrench = ca.Function(
                    'external_wrench', [q_sym, f_ext_sym],
                    [tau_ext], ['q', 'f_ext'], ['tau_ext'],
                    self.fn_opts)
            return self.fn_external_wrench(q, f_ext)

    def get_rotation_error(self, R_1, R_2):
        build_function = not hasattr(self, 'fn_rotation_error')
        if build_function:
            R_1_sym = ca.SX.sym('R_1', 3, 3)
            R_2_sym = ca.SX.sym('R_2', 3, 3)
            phi = cpin.log3(R_1_sym @ R_2_sym.T)
            self.fn_rotation_error = ca.Function(
                'rotation_error', [R_1_sym, R_2_sym],
                [phi], ['R_1', 'R_2'], ['phi'],
                self.fn_opts)
        return self.fn_rotation_error(R_1, R_2)

    def get_inverse_dynamics(self, q, v, a):
        q, is_symbolic = self._preprocess_vec(q)
        v, _ = self._preprocess_vec(v)
        a, _ = self._preprocess_vec(a)
        model, data, pin_backend = self._get_backend(is_symbolic)
        build_function = not hasattr(self, 'fn_inverse_dynamics')
        if is_symbolic:
            if build_function:
                q_sym = ca.SX.sym('q', self.cpin_model.nq, 1)
                v_sym = ca.SX.sym('v', self.cpin_model.nv, 1)
                a_sym = ca.SX.sym('a', self.cpin_model.nv, 1)
                tau_sym = pin_backend.rnea(model, data, q_sym, v_sym, a_sym)
                labels_in = ['q', 'v', 'a']
                labels_out = ['tau_inertial']
                self.fn_inverse_dynamics = ca.Function(
                    'inverse_dynamics', [q_sym, v_sym, a_sym],
                    [tau_sym], labels_in, labels_out, self.fn_opts)
            return self.fn_inverse_dynamics(q, v, a)
        else:
            return pin_backend.rnea(model, data, q, v, a)

    ##### Utility functions #####
    def _preprocess_vec(self, vec):
        is_symbolic = isinstance(vec, (ca.SX, ca.MX))
        if not is_symbolic:
            if isinstance(vec, list):
                vec = np.array(vec)
            if len(vec.shape) == 2:
                vec = vec.squeeze()
        return vec, is_symbolic

    def _get_backend(self, is_symbolic):
        """Return the correct model, data, and backend module."""
        if is_symbolic:
            return self.cpin_model, self.cpin_data, cpin
        else:
            return self.pin_model, self.pin_data, pin

    def _parse_frame_labels(self, frames):
        if frames is None:
            frame_ids = self.end_eff_frame_ids
        else:
            frame_labels = [frames] if isinstance(frames, str) else frames
            frame_ids = [self.pin_model.getFrameId(label) for label in frame_labels]
        frame_names = [self.pin_model.frames[i].name for i in frame_ids]
        return frame_ids, frame_names