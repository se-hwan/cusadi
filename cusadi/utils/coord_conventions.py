import numpy as np
import casadi as ca
from .orientation import orientation_map
from .symbolic import auto_concat

class CoordConvention:
    name: str
    fb_pos_order: list      # {'lin', 'ang'}
    fb_vel_order: list      # {'lin', 'ang'}
    fb_vel_frame: list      # {'base', 'world'}
    orientation: str        # {'quat_xyzw', 'quat_wxyz', 'rpy', 'R_mat'}
    is_floating: bool
    fb_dim: int             # Floating base dimension
    joint_order: list       # List of joint names in order
    joint_dim: int          # Number of joints

    def __init__(self, name,
                 fb_pos_order=['lin', 'ang'],
                 fb_vel_order=['lin', 'ang'],
                 fb_vel_frame=['base', 'base'],
                 orientation='quat_xyzw',
                 is_floating=True):
        self.name = name
        self.is_floating = is_floating
        self.joint_order = None
        self.fb_pos_order = fb_pos_order
        self.fb_vel_order = fb_vel_order
        self.fb_order = {
            'gc': self.fb_pos_order,
            'gv': self.fb_vel_order}
        self.fb_vel_info = {
            fb_vel_order[0]: fb_vel_frame[0],
            fb_vel_order[1]: fb_vel_frame[1]}
        self.orientation = orientation
        self.set_fb_dim()
        self.set_fb_indices()
    
    def set_fb_dim(self):
        is_quat = self.orientation.startswith('quat')
        is_rotmat = self.orientation == 'R_mat'
        self.fb_pos_dim = {'lin': 3}
        if is_rotmat:
            self.fb_dim = 9 + 3  # rotmat + lin
            self.fb_pos_dim['ang'] = 9
        elif is_quat:
            self.fb_dim = 4 + 3  # quat + lin
            self.fb_pos_dim['ang'] = 4
        else:
            self.fb_dim = 6
            self.fb_pos_dim['ang'] = 3

    def set_fb_indices(self):
        # Compute contiguous index ranges based on order
        self.gc_idxs = {}
        self.gv_idxs = {}
        start_pos = 0
        start_vel = 0
        for key in self.fb_pos_order:
            self.gc_idxs[key] = list(range(start_pos, start_pos + self.fb_pos_dim[key]))
            self.gv_idxs[key] = list(range(start_vel, start_vel + 3))
            start_pos += self.fb_pos_dim[key]            
            start_vel += 3

    def set_joint_order(self, joint_order):
        self.joint_order = joint_order
        self.joint_dim = len(joint_order)
        self.gc_dim = self.fb_dim + self.joint_dim
        self.gv_dim = 6 + self.joint_dim

    def set_floating(self, is_floating):
        self.is_floating = is_floating


def convert_floating_base(vec_src, vec_type, src_key: CoordConvention,
                        dst_key: CoordConvention, is_symbolic=False):
    assert src_key.fb_order is not None, "Source floating base order not set"
    assert dst_key.fb_order is not None, "Destination floating base order not set"

    fb_dst_dict = {}
    if vec_type == 'gc':
        src_ori_convention = src_key.orientation
        dst_ori_convention = dst_key.orientation
        src_ori = vec_src[src_key.gc_idxs['ang']]
        fb_dst_dict['ang'] = orientation_map.map(src_ori, is_symbolic,
                                      src_ori_convention, dst_ori_convention)
        fb_dst_dict['lin'] = vec_src[src_key.gc_idxs['lin']]
        fb_dst = [fb_dst_dict[key] for key in dst_key.fb_pos_order]

    elif vec_type == 'gv':
        src_lin_vel = vec_src[src_key.gv_idxs['lin']]
        src_ang_vel = vec_src[src_key.gv_idxs['ang']]
        fb_dst_dict['ang'] = src_ang_vel # Frame conversion logic
        fb_dst_dict['lin'] = src_lin_vel
        if src_key.fb_vel_info['lin'] != dst_key.fb_vel_info['lin']:
            raise NotImplementedError("Linear velocity frame conversion not implemented")
        if src_key.fb_vel_info['ang'] != dst_key.fb_vel_info['ang']:
            raise NotImplementedError("Angular velocity frame conversion not implemented")
        fb_dst = [fb_dst_dict[key] for key in dst_key.fb_vel_order] # Reorder

    return auto_concat(fb_dst, is_symbolic)

def convert_joint_order(vec_src, src_key: CoordConvention,
                        dst_key: CoordConvention, is_symbolic=False):
    assert src_key.joint_order is not None, "Source joint order not set"
    assert dst_key.joint_order is not None, "Destination joint order not set"
    assert vec_src.shape[0] == src_key.joint_dim, \
        "Input vector size does not match source joint dimension"
    assert set(src_key.joint_order) == set(dst_key.joint_order), \
        "Source and destination joint orders do not match"

    joint_idx_reordering = [src_key.joint_order.index(a) 
                            for a in dst_key.joint_order]
    joint_dst = vec_src[joint_idx_reordering]
    return joint_dst

def convert_coordinates(vec_src, vec_type, src_key: CoordConvention,
                        dst_key: CoordConvention, is_symbolic=False):
    if src_key.is_floating:
        jnt_idx = vec_src.shape[0] - src_key.joint_dim
        fb_src = vec_src[0:jnt_idx]
        joint_src = vec_src[jnt_idx:]
        fb_dst = convert_floating_base(fb_src, vec_type, src_key, dst_key, is_symbolic)
        joint_dst = convert_joint_order(joint_src, src_key, dst_key, is_symbolic)
        return auto_concat([fb_dst, joint_dst], is_symbolic)
    else:
        joint_dst = convert_joint_order(vec_src, src_key, dst_key, is_symbolic)
        return joint_dst



PIN_CONVENTION = CoordConvention(
    # q = ([x, y, z], [qx, qy, qz, qw], [joint_angles])
    # v = ([vx, vy, vz] (base), [wx, wy, wz] (base), [joint_vels])
    name='pinocchio',
    fb_pos_order=['lin', 'ang'],
    fb_vel_order=['lin', 'ang'],
    fb_vel_frame=['base', 'base'],
    orientation='quat_xyzw'
)

RS_CONVENTION = CoordConvention(
    # q = ([x, y, z], [r, p, y], [joint_angles])
    # v = ([wx, wy, wz], vx, vy, vz, [joint_vels]) in base frame
    name='robot-software',
    fb_pos_order=['lin', 'ang'],
    fb_vel_order=['ang', 'lin'],
    fb_vel_frame=['base', 'base'],
    orientation='rpy'
)

VISER_CONVENTION = CoordConvention(
    # q = ([x, y, z], [qw, qx, qy, qz], [joint_angles])
    # v = ([vx, vy, vz] (base), [wx, wy, wz] (base), [joint_vels])
    name='viser',
    fb_pos_order=['lin', 'ang'],
    fb_vel_order=['lin', 'ang'],
    fb_vel_frame=['base', 'base'],
    orientation='quat_wxyz'
)

MUJOCO_CONVENTION = CoordConvention(
    # q = ([x, y, z], [qw, qx, qy, qz], [joint_angles])
    # v = ([vx, vy, vz] (world), [wx, wy, wz] (base), [joint_vels])
    name='mujoco',
    fb_pos_order=['lin', 'ang'],
    fb_vel_order=['lin', 'ang'],
    fb_vel_frame=['world', 'base'],
    orientation='quat_wxyz'
)

ISAAC_CONVENTION = CoordConvention(
    # q = ([x, y, z], [qw, qx, qy, qz], [joint_angles])
    # v = ([vx, vy, vz] (world), [wx, wy, wz] (world), [joint_vels])
    name='isaac',
    fb_pos_order=['lin', 'ang'],
    fb_vel_order=['lin', 'ang'],
    fb_vel_frame=['world', 'world'],
    orientation='quat_wxyz'
)