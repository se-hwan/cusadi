import scipy
import numpy as np
import casadi as ca
from .symbolic import auto_concat

def identity(ori, is_symbolic):
    return ori

def quat_xyzw_to_rot(quat, is_symbolic=None):
    x, y, z, w = quat

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

    wx = w * x
    wy = w * y
    wz = w * z
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=float)


def rot_to_quat_xyzw(R, is_symbolic=None):
    """
    Parameters:
        R (np.ndarray): A 3x3 rotation matrix.
    Returns:
        np.ndarray: A quaternion (x, y, z, w).
    """
    assert R.shape == (3, 3), "Input must be a 3x3 matrix"
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    else:
        # Find the major diagonal element
        if m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    quat = np.array([x, y, z, w])
    quat /= np.linalg.norm(quat)
    return quat

"""
# ! RPY convention means R = Rz(yaw) * Ry(pitch) * Rx(roll)
Convert roll-pitch-yaw (rpy) to quaternions.
Works with numpy arrays or CasADi SX/MX types.
"""
def rpy_to_quat_xyzw(rpy, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(rpy, (ca.SX, ca.MX))
    math = ca if is_symbolic else np

    # Half angles
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]

    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)

    # Quaternion components (xyzw)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy

    # # Normalize quaternion
    # norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    # qw /= norm
    # qx /= norm
    # qy /= norm
    # qz /= norm

    quat = auto_concat([qx, qy, qz, qw], is_symbolic)
    return quat

# ! Do NOT trust this function yet, needs testing
def rpy_to_quat_wxyz(rpy, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(pitch, (ca.SX, ca.MX))
    math = ca if is_symbolic else np

    # Half angles
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)

    # Quaternion (w, x, y, z)
    qw = cy * cp * cr + sy * sp * sr
    qx = sy * cp * cr - cy * sp * sr
    qy = cy * sp * cr + sy * cp * sr
    qz = cy * cp * sr - sy * sp * cr

    quat = auto_concat([qw, qx, qy, qz], is_symbolic)
    return quat

'''
Convert quaternions to roll-pitch-yaw.
Works with numpy arrays or CasADi SX/MX types.
'''
def quat_xyzw_to_rpy(quat, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(pitch, (ca.SX, ca.MX))
    math = ca if is_symbolic else np

    qx, qy, qz, qw = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    rpy = auto_concat([roll, pitch, yaw], is_symbolic)
    return rpy

def quat_wxyz_to_rpy(quat, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(pitch, (ca.SX, ca.MX))
    math = ca if is_symbolic else np

    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    rpy = auto_concat([roll, pitch, yaw], is_symbolic)
    return rpy

def rpy_to_rotmat(rpy, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(rpy, (ca.SX, ca.MX))
    roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
    Rz = rz(yaw, is_symbolic)
    Ry = ry(pitch, is_symbolic)
    Rx = rx(roll, is_symbolic)
    return Rz.T @ Ry.T @ Rx.T

def rx(roll, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(roll, (ca.SX, ca.MX))
    math = ca if is_symbolic else np
    c = math.cos(roll)
    s = math.sin(roll)
    if is_symbolic:
        sym_matrix = ca.SX if isinstance(roll, ca.SX) else ca.MX
        R = sym_matrix(3,3)
        R[0, 0] = 1
        R[1, 1] = c
        R[1, 2] = s
        R[2, 1] = -s
        R[2, 2] = c
    else:
        R = np.array([
            [1, 0, 0],
            [0, c, s],
            [0, -s, c]])
    return R

def ry(pitch, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(pitch, (ca.SX, ca.MX))
    math = ca if is_symbolic else np
    c = math.cos(pitch)
    s = math.sin(pitch)
    if is_symbolic:
        sym_matrix = ca.SX if isinstance(pitch, ca.SX) else ca.MX
        R = sym_matrix(3,3)
        R[0, 0] = c
        R[0, 2] = -s
        R[1, 1] = 1
        R[2, 0] = s
        R[2, 2] = c
    else:
        R = np.array([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]])
    return R

def rz(yaw, is_symbolic=None):
    if is_symbolic is None:
        is_symbolic = isinstance(yaw, (ca.SX, ca.MX))
    math = ca if is_symbolic else np
    c = math.cos(yaw)
    s = math.sin(yaw)
    if is_symbolic:
        sym_matrix = ca.SX if isinstance(yaw, ca.SX) else ca.MX
        R = sym_matrix(3,3)
        R[0, 0] = c
        R[0, 1] = s
        R[1, 0] = -s
        R[1, 1] = c
        R[2, 2] = 1
    else:
        R = np.array([
            [c, s, 0],
            [-s, c, 0],
            [0, 0, 1]])
    return R

def b_inv(rpy, is_symbolic=None):
    '''
    omega = B*[r_dot, p_dot, y_dot], converts Euler angle rates to angular velocity
    Computes inverse of B matrix
    '''
    if is_symbolic is None:
        is_symbolic = isinstance(rpy, (ca.SX, ca.MX))
    math = ca if is_symbolic else np

    c_psi = math.cos(rpy[2]); s_psi = math.sin(rpy[0])
    c_th = math.cos(rpy[1]); s_th = math.sin(rpy[1])
    t_th = math.atan2(s_th, c_th)
    if is_symbolic:
        sym_matrix = ca.SX if isinstance(rpy, ca.SX) else ca.MX
        B_inv = sym_matrix(3,3)
        B_inv[0, 0] = c_psi / c_th
        B_inv[0, 1] = s_psi / c_th
        B_inv[1, 0] = -s_psi
        B_inv[1, 1] = c_psi
        B_inv[2, 0] = c_psi * t_th
        B_inv[2, 1] = s_psi * t_th
        B_inv[2, 2] = 1
    else:
        B_inv = np.array([[c_psi/c_th, s_psi/c_th,      0],
                          [-s_psi,          c_psi,      0],
                          [c_psi*t_th, s_psi*t_th,      1]])
    return B_inv


class OrientationMap:
    info: dict
    def __init__(self):
        self.graph = {}

    def add_edge(self, a, b, func_ab, func_ba=None):
        """Add bidirectional connection between a and b.
        If func_ba is None, assumes symmetric mapping (same function both ways)."""
        if func_ba is None:
            func_ba = func_ab
        self.graph.setdefault(a, {})[b] = func_ab
        self.graph.setdefault(b, {})[a] = func_ba
        self.graph.setdefault(a, {})[a] = identity
        self.graph.setdefault(b, {})[b] = identity

    def get(self, a, b):
        """Get mapping function from a → b, or None if not found."""
        return self.graph.get(a, {}).get(b, None)

    def map(self, ori, is_symbolic, map_src, map_dst):
        """Apply mapping function from a → b."""
        func = self.get(map_src, map_dst)
        if func is None:
            raise NotImplementedError(f"No mapping from {map_src} → {map_dst}")
        return func(ori, is_symbolic)

    def neighbors(self, a):
        """Return all connected nodes."""
        return list(self.graph.get(a, {}).keys())


orientation_map = OrientationMap()
orientation_map.add_edge('rpy', 'quat_xyzw', rpy_to_quat_xyzw, quat_xyzw_to_rpy)
orientation_map.add_edge('rpy', 'quat_wxyz', rpy_to_quat_wxyz, quat_wxyz_to_rpy)